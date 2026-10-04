"""Supervisor that runs the whole local backend: API server, iMessage sidecar, Postgres.

    uv run atlas up            # start in the background, print status
    uv run atlas up --foreground
    uv run atlas status | logs [name] [-f] | down

Each service runs in its own process group, logs to data/logs/<name>.log and restarts with
exponential backoff (1s, 2s, 4s ... 60s; reset after 60s of uptime). State for `atlas status`
lives in data/run/state.json, the supervisor pid in data/run/daemon.pid.

Gmail sync runs inside the API server (atlas/notify.py polls atlas.ingest.sync.sync_new every
POLL_SECONDS) because watch alerts need the new ids and the in-memory index has to see the
appended vectors. The daemon reports it but does not run a second sync process.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from atlas import config

RUN = config.DATA / "run"
LOGS = config.DATA / "logs"
PID_FILE = RUN / "daemon.pid"
STATE_FILE = RUN / "state.json"
MAX_BACKOFF = 60
STABLE_SECONDS = 60


def env(name, default=""):
    return os.getenv(name) or config.env(name, default) or default


def ports():
    return int(env("PORT", "8765")), int(env("SIDECAR_PORT", "8766"))


def spectrum_configured():
    return bool(env("SPECTRUM_PROJECT_ID") and env("SPECTRUM_PROJECT_SECRET"))


def gmail_configured():
    return bool(env("GMAIL_USER") and env("GMAIL_APP_PASSWORD"))


def pg_wanted():
    return env("ATLAS_DB").lower() in ("pg", "postgres") and (config.ROOT / "docker-compose.yml").exists()


def alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    try:  # a zombie child still answers kill(0)
        out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        return bool(out) and not out.startswith("Z")
    except OSError:
        return True


def read_pid():
    try:
        return int(PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return None


def read_state():
    try:
        return json.loads(STATE_FILE.read_text())
    except (OSError, ValueError):
        return {}


def tail(path: Path, n=20) -> list[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 64 * 1024))
            return f.read().decode(errors="replace").splitlines()[-n:]
    except OSError:
        return []


def last_error_line(path: Path) -> str | None:
    for line in reversed(tail(path, 80)):
        if any(w in line for w in ("Error", "error", "Traceback", "failed", "FAILED", "Exception", "could not", "cannot")):
            return line.strip()[:300]
    return None


# ---------- services ----------

@dataclass
class Service:
    name: str
    cmd: list[str]
    cwd: Path
    env: dict = field(default_factory=dict)
    proc: subprocess.Popen | None = None
    restarts: int = 0
    failures: int = 0
    started: float = 0.0
    next_start: float = 0.0
    last_exit: int | None = None
    last_error: str | None = None
    note: str = ""

    @property
    def log(self) -> Path:
        return LOGS / f"{self.name}.log"

    def start(self):
        LOGS.mkdir(parents=True, exist_ok=True)
        f = open(self.log, "ab", buffering=0)
        f.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} starting {' '.join(self.cmd)}\n".encode())
        self.proc = subprocess.Popen(self.cmd, cwd=self.cwd, env={**os.environ, **self.env}, stdout=f,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        f.close()
        self.started = time.time()

    def poll(self):
        """Restart with backoff if the process died. Returns True if it is running."""
        now = time.time()
        if self.proc and self.proc.poll() is None:
            if now - self.started > STABLE_SECONDS:
                self.failures = 0
            return True
        if self.proc is not None:  # just died
            self.last_exit = self.proc.returncode
            self.last_error = last_error_line(self.log) or f"exited with code {self.last_exit}"
            self.failures += 1
            self.restarts += 1
            self.proc = None
            self.next_start = now + min(MAX_BACKOFF, 2 ** (self.failures - 1))
        if now >= self.next_start:
            try:
                self.start()
            except OSError as e:
                self.last_error = f"cannot start: {e}"
                self.failures += 1
                self.next_start = now + min(MAX_BACKOFF, 2 ** (self.failures - 1))
                return False
            return True
        return False

    def stop(self, timeout=8.0):
        p = self.proc
        if not p or p.poll() is not None:
            return
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except OSError:
            pass
        try:
            p.wait(timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                pass
            p.wait(3)

    def info(self):
        running = bool(self.proc and self.proc.poll() is None)
        return {"pid": self.proc.pid if running else None, "running": running, "restarts": self.restarts,
                "started": self.started if running else None, "last_exit": self.last_exit,
                "last_error": self.last_error, "log": str(self.log), "note": self.note,
                "next_start": self.next_start if not running else None}


def sidecar_mode():
    """'spectrum' with credentials, 'mock' if ATLAS_SIDECAR=mock, else None (not started)."""
    forced = env("ATLAS_SIDECAR").lower()
    if forced in ("off", "0", "no"):
        return None
    if forced == "mock":
        return "mock"
    return "spectrum" if spectrum_configured() else None


def build_services() -> tuple[list[Service], dict]:
    port, sport = ports()
    api_url = f"http://127.0.0.1:{port}"
    side_url = f"http://127.0.0.1:{sport}"
    base_env = {"PORT": str(port), "SIDECAR_PORT": str(sport), "SIDECAR_URL": side_url, "ATLAS_URL": api_url,
                "PYTHONUNBUFFERED": "1"}
    services = [Service("api", [sys.executable, str(config.ROOT / "server.py")], config.ROOT, base_env)]
    skipped = {}
    mode = sidecar_mode()
    node = shutil.which("node")
    if not mode:
        skipped["imessage"] = "set SPECTRUM_PROJECT_ID and SPECTRUM_PROJECT_SECRET (or ATLAS_SIDECAR=mock)"
    elif not node:
        skipped["imessage"] = "node is not installed (brew install node)"
    else:
        cmd = [node, str(config.ROOT / "imessage" / "src" / "index.js")]
        if mode == "mock":
            cmd += ["--dry-run", "--no-repl"]
        s = Service("imessage", cmd, config.ROOT / "imessage", base_env)
        s.note = mode
        services.append(s)
    return services, skipped


def ensure_node_modules(log):
    d = config.ROOT / "imessage"
    if (d / "node_modules" / "spectrum-ts").exists():
        return True
    npm = shutil.which("npm")
    if not npm:
        return False
    log("installing imessage/ node modules (npm ci)")
    with open(LOGS / "imessage.log", "ab") as f:
        r = subprocess.run([npm, "ci", "--no-audit", "--no-fund"], cwd=d, stdout=f, stderr=subprocess.STDOUT)
    return r.returncode == 0


def compose_cmd() -> list[str] | None:
    """`docker compose` (plugin) if it works, else the standalone `docker-compose` binary."""
    docker = shutil.which("docker")
    if docker:
        try:
            if subprocess.run([docker, "compose", "version"], capture_output=True, timeout=10).returncode == 0:
                return [docker, "compose"]
        except (OSError, subprocess.TimeoutExpired):
            pass
    dc = shutil.which("docker-compose") or next(
        (p for p in ("/opt/homebrew/bin/docker-compose", "/usr/local/bin/docker-compose") if Path(p).exists()), None)
    return [dc] if dc else None


def compose(*args, timeout=180):
    base = compose_cmd()
    if not base:
        return False, "docker compose not found (install Docker Desktop or docker-compose)"
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / "db.log", "ab") as f:
        f.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(base + list(args))}\n".encode())
        try:
            r = subprocess.run([*base, *args], cwd=config.ROOT, stdout=f, stderr=subprocess.STDOUT,
                               timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as e:
            return False, str(e)
    return r.returncode == 0, None if r.returncode == 0 else (last_error_line(LOGS / "db.log") or f"exit {r.returncode}")


# ---------- supervisor ----------

class Supervisor:
    def __init__(self, log=print):
        self.log = log
        self.stopping = False
        self.services, self.skipped = build_services()
        self.db = {"wanted": pg_wanted(), "up": False, "last_error": None}

    def write_state(self, stopped=False):
        port, sport = ports()
        st = {"pid": os.getpid(), "updated": time.time(), "started": getattr(self, "t0", time.time()),
              "stopped": stopped, "port": port, "sidecar_port": sport,
              "services": {s.name: s.info() for s in self.services}, "skipped": self.skipped, "db": self.db,
              "gmail": "in api notifier" if gmail_configured() else "set GMAIL_USER and GMAIL_APP_PASSWORD",
              "token": bool(env("ATLAS_TOKEN"))}
        RUN.mkdir(parents=True, exist_ok=True)
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=1))
        tmp.replace(STATE_FILE)

    def _signal(self, *_):
        self.stopping = True

    def run(self):
        other = read_pid()
        if other and other != os.getpid() and alive(other):
            self.log(f"atlas daemon already running (pid {other})")
            return 1
        RUN.mkdir(parents=True, exist_ok=True)
        LOGS.mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text(str(os.getpid()))
        self.t0 = time.time()
        signal.signal(signal.SIGTERM, self._signal)
        signal.signal(signal.SIGINT, self._signal)
        signal.signal(signal.SIGHUP, self._signal)
        try:
            if self.db["wanted"]:
                self.log("starting Postgres (docker compose up -d)")
                ok, err = compose("up", "-d")
                self.db.update(up=ok, last_error=err)
            if any(s.name == "imessage" for s in self.services) and not ensure_node_modules(self.log):
                self.skipped["imessage"] = "npm install failed in imessage/, see data/logs/imessage.log"
                self.services = [s for s in self.services if s.name != "imessage"]
            for s in self.services:
                s.start()
                self.log(f"started {s.name} (pid {s.proc.pid}), log {s.log}")
            while not self.stopping:
                for s in self.services:
                    s.poll()
                self.write_state()
                for _ in range(10):
                    if self.stopping:
                        break
                    time.sleep(0.1)
        finally:
            self.log("stopping services")
            for s in reversed(self.services):
                s.stop()
            self.write_state(stopped=True)
            try:
                if read_pid() == os.getpid():
                    PID_FILE.unlink()
            except OSError:
                pass
        return 0


# ---------- commands ----------

def start_background(wait=30.0) -> int:
    pid = read_pid()
    if pid and alive(pid):
        print(f"atlas is already running (pid {pid}). `atlas status` for details.")
        return 0
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / "daemon.log", "ab") as f:
        p = subprocess.Popen([sys.executable, "-m", "atlas.cli", "up", "--foreground"], cwd=config.ROOT,
                             stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"atlas daemon started (pid {p.pid}), waiting for the API...")
    port, _ = ports()
    t0 = time.time()
    while time.time() - t0 < wait:
        if p.poll() is not None:
            print("daemon exited early, see data/logs/daemon.log:")
            print("\n".join(tail(LOGS / "daemon.log", 15)))
            return 1
        if _get(f"http://127.0.0.1:{port}/api/health", timeout=1):
            break
        time.sleep(0.5)
    print()
    return status()


def stop(timeout=20.0) -> int:
    pid = read_pid()
    st = read_state()
    if pid and alive(pid):
        os.kill(pid, signal.SIGTERM)
        t0 = time.time()
        while alive(pid) and time.time() - t0 < timeout:
            time.sleep(0.2)
        if alive(pid):
            os.kill(pid, signal.SIGKILL)
        print(f"stopped atlas daemon (pid {pid})")
    else:
        print("atlas daemon is not running")
    # leftovers (daemon killed hard): kill recorded service process groups
    for name, s in (st.get("services") or {}).items():
        spid = s.get("pid")
        if spid and alive(spid):
            try:
                os.killpg(spid, signal.SIGTERM)
                print(f"killed leftover {name} (pid {spid})")
            except OSError:
                pass
    if (st.get("db") or {}).get("up") and env("ATLAS_DB_STOP", "0") == "1":
        compose("stop")
    try:
        PID_FILE.unlink()
    except OSError:
        pass
    return 0


def _get(url, timeout=2.0, token=None):
    import httpx
    t = token if token is not None else env("ATLAS_TOKEN")
    try:
        r = httpx.get(url, timeout=timeout, headers={"Authorization": f"Bearer {t}"} if t else {})
        return r.json() if r.status_code < 400 else None
    except Exception:  # noqa: BLE001
        return None


def _ago(ts):
    if not ts:
        return "never"
    d = int(time.time() - ts)
    return f"{d}s ago" if d < 120 else f"{d // 60}m ago" if d < 7200 else f"{d // 3600}h ago"


def index_count(encoder):
    try:
        ids = json.loads((config.INDEX_DIR / encoder / "ids.json").read_text())
        return len(ids)
    except (OSError, ValueError):
        for d in sorted(config.INDEX_DIR.glob("*/ids.json")):
            try:
                return len(json.loads(d.read_text()))
            except (OSError, ValueError):
                pass
    return None


def collect_status() -> dict:
    pid = read_pid()
    st = read_state()
    port = st.get("port") or ports()[0]
    sport = st.get("sidecar_port") or ports()[1]
    running = bool(pid and alive(pid))
    out = {"daemon": {"running": running, "pid": pid if running else None,
                      "uptime": f"started {_ago(st.get('started'))}" if running else "not running (uv run atlas up)"},
           "state": st, "port": port, "sidecar_port": sport}
    out["health"] = _get(f"http://127.0.0.1:{port}/api/health")
    out["notify"] = _get(f"http://127.0.0.1:{port}/api/notify/status")
    out["sidecar"] = _get(f"http://127.0.0.1:{sport}/health")
    enc = (out["health"] or {}).get("encoder") or env("ATLAS_ENCODER", "base")
    out["indexed"] = index_count(enc)
    return out


def status(as_json=False) -> int:
    s = collect_status()
    if as_json:
        print(json.dumps(s, indent=1, default=str))
        return 0
    st, d = s["state"], s["daemon"]
    ok = lambda b: "up  " if b else "DOWN"  # noqa: E731
    print(f"atlas daemon   {ok(d['running'])} pid {d['pid'] or '-'}, {d['uptime']}")
    svcs = st.get("services") or {}
    h = s["health"]
    api = svcs.get("api") or {}
    line = f"api            {ok(bool(h))} http://127.0.0.1:{s['port']}"
    if api.get("restarts"):
        line += f", {api['restarts']} restarts"
    print(line)
    if h:
        if h.get("auth_required") and "n_emails" not in h:
            print("               token rejected: set ATLAS_TOKEN in this shell to match the server")
        else:
            print(f"               {h.get('n_emails')} emails in mail.sqlite, {s['indexed'] if s['indexed'] is not None else 'no'} "
                  f"in the index, encoder {h.get('encoder')}, token {'on' if h.get('auth_required') else 'off'}")
    if api.get("last_error"):
        print(f"               last error: {api['last_error']}")

    side = svcs.get("imessage")
    sc = s["sidecar"]
    if side and not d["running"]:
        print("imessage       DOWN")
    elif side:
        mode = (sc or {}).get("mode") or side.get("note")
        print(f"imessage       {ok(bool(sc))} http://127.0.0.1:{s['sidecar_port']} ({mode} mode)"
              + (f", {side['restarts']} restarts" if side.get("restarts") else ""))
        if sc and sc.get("mode") == "spectrum":
            conn = sc.get("connected")
            print(f"               Photon {'connected' if conn else 'NOT connected'}, providers {', '.join(sc.get('providers') or ['imessage'])}, owner {sc.get('owner') or 'NOT SET (OWNER_PHONE)'}")
            if not sc.get("threads"):
                print("               one time step: text the Photon line once from your phone (say \"help\").")
                print("               Shared lines cannot start a chat, so the brief and alerts only arrive after that.")
            else:
                print(f"               known threads: {', '.join(sc.get('threads'))}")
        if side.get("last_error"):
            print(f"               last error: {side['last_error']}")
    else:
        print(f"imessage       off  {(st.get('skipped') or {}).get('imessage') or 'not started'}")

    db = st.get("db") or {}
    if db.get("wanted"):
        print(f"postgres       {ok(db.get('up'))} docker compose (ATLAS_DB=pg)" + (f", {db['last_error']}" if db.get("last_error") else ""))
    else:
        print("postgres       off  set ATLAS_DB=pg (needs docker-compose.yml from the tiger branch)")

    n = s["notify"] or {}
    on = gmail_configured()
    gm = f"polled by the api notifier every {(n or {}).get('poll_seconds') or 120}s" if on else "set GMAIL_USER and GMAIL_APP_PASSWORD"
    print(f"gmail sync     {'on  ' if on else 'off '} {gm}" + (f", last poll {_ago(n.get('last_poll'))}" if n and on else ""))
    if n:
        print(f"morning brief  {'on  ' if n.get('brief_enabled') else 'off '} at {n.get('brief_time')} {n.get('timezone')}, last sent {n.get('last_brief') or 'never'}")
        if n.get("last_error"):
            print(f"notifier error {n['last_error'][:200]}")
    print(f"logs           {LOGS}")
    return 0


def logs(name=None, follow=False, n=40) -> int:
    files = [LOGS / f"{name}.log"] if name else sorted(LOGS.glob("*.log"))
    files = [f for f in files if f.exists()]
    if not files:
        print(f"no logs in {LOGS}")
        return 1
    if follow:
        os.execvp("tail", ["tail", "-n", str(n), "-F", *map(str, files)])
    for f in files:
        print(f"==> {f.name} <==")
        print("\n".join(tail(f, n)))
    return 0
