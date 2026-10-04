"""`atlas` command: run and inspect the local backend.

    atlas up [--foreground]     start everything (API, iMessage sidecar, Postgres)
    atlas down                  stop everything
    atlas status [--json]       what is running, emails indexed, Photon, last brief
    atlas logs [name] [-f]      api | imessage | db | daemon
    atlas phone                 how to open Inbox Atlas on your phone (Tailscale)
    atlas install-service       macOS LaunchAgent so it starts at login and stays up
    atlas uninstall-service
"""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

from atlas import config, daemon

LABEL = "com.inboxatlas.daemon"
PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def cmd_up(a):
    if a.foreground:
        return daemon.Supervisor(log=lambda m: print(f"[atlas] {m}", flush=True)).run()
    return daemon.start_background()


def plist_dict():
    path = os.pathsep.join(dict.fromkeys(
        [str(Path(sys.executable).parent), "/opt/homebrew/bin", "/usr/local/bin", *os.environ.get("PATH", "").split(os.pathsep),
         "/usr/bin", "/bin", "/usr/sbin", "/sbin"]))
    return {
        "Label": LABEL,
        "ProgramArguments": [sys.executable, "-m", "atlas.cli", "up", "--foreground"],
        "WorkingDirectory": str(config.ROOT),
        "EnvironmentVariables": {"PATH": path, "PYTHONUNBUFFERED": "1"},
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 10,
        "ProcessType": "Interactive",
        "StandardOutPath": str(daemon.LOGS / "launchd.log"),
        "StandardErrorPath": str(daemon.LOGS / "launchd.log"),
    }


def cmd_install_service(a):
    daemon.LOGS.mkdir(parents=True, exist_ok=True)
    PLIST.parent.mkdir(parents=True, exist_ok=True)
    with open(PLIST, "wb") as f:
        plistlib.dump(plist_dict(), f)
    uid = os.getuid()
    print(f"wrote {PLIST}")
    print("It runs `atlas up --foreground` at login and launchd restarts it if it dies.")
    print("Stop any running copy first (`uv run atlas down`), then load it:")
    print(f"  launchctl bootstrap gui/{uid} {PLIST}")
    print("Check it:")
    print(f"  launchctl print gui/{uid}/{LABEL} | head -20")
    print("Note: with the service loaded, `atlas down` stops the services but launchd starts them again;")
    print("use `atlas uninstall-service` to stop for good.")
    return 0


def cmd_uninstall_service(a):
    uid = os.getuid()
    print("Unload it first if loaded:")
    print(f"  launchctl bootout gui/{uid}/{LABEL}")
    if PLIST.exists():
        PLIST.unlink()
        print(f"removed {PLIST}")
    else:
        print(f"{PLIST} not present")
    return 0


def tailscale_bin():
    for p in (shutil.which("tailscale"), "/usr/local/bin/tailscale",
              "/Applications/Tailscale.app/Contents/MacOS/Tailscale"):
        if p and Path(p).exists():
            return p
    return None


def tailscale_info():
    ts = tailscale_bin()
    if not ts:
        return None
    try:
        out = subprocess.run([ts, "status", "--json"], capture_output=True, text=True, timeout=5).stdout
        d = json.loads(out)
        me = d.get("Self") or {}
        return {"bin": ts, "dns": (me.get("DNSName") or "").rstrip("."), "ips": me.get("TailscaleIPs") or [],
                "state": d.get("BackendState")}
    except Exception:  # noqa: BLE001
        return {"bin": ts, "dns": "", "ips": [], "state": "unknown"}


def cmd_phone(a):
    port, _ = daemon.ports()
    token = daemon.env("ATLAS_TOKEN")
    ts = tailscale_info()
    print("Inbox Atlas on your phone\n")
    print("1. iMessage (Photon): text the Photon line. `atlas status` shows if the sidecar is connected.")
    print("   First time only: send any text (\"help\") to the line, shared lines cannot start a chat.\n")
    print("2. Web app over Tailscale (phone and Mac on the same tailnet, Tailscale app on the phone):")
    if not ts:
        print("   tailscale CLI not found. Install Tailscale on the Mac, then rerun `atlas phone`.")
        return 1
    if ts["state"] != "Running":
        print(f"   tailscale is {ts['state']}. Open the Tailscale app and log in, then rerun.")
    host = ts["dns"] or "mccrispy"
    serve = f"{ts['bin']} serve --bg {port}"
    print(f"   HTTPS on the tailnet (recommended, works with the API bound to localhost):")
    print(f"     {serve}")
    print(f"   then open on the phone:  https://{host}/" + (f"?token={token}" if token and a.show_token else ""))
    print(f"   undo with: {ts['bin']} serve --https=443 off   (or `tailscale serve reset`)")
    try:
        cur = subprocess.run([ts["bin"], "serve", "status"], capture_output=True, text=True, timeout=5).stdout.strip()
        print(f"   current serve config: {cur.splitlines()[0] if cur else 'none'}")
    except Exception:  # noqa: BLE001
        pass
    ip = next((i for i in ts["ips"] if "." in i), "100.64.0.7")
    print(f"\n   Plain HTTP by tailnet IP: http://{ip}:{port}/")
    if token:
        print("   (works now: ATLAS_TOKEN is set, so the API listens on all interfaces and asks for the token)")
    else:
        print("   (only if you set ATLAS_TOKEN in .env and restart; without a token the API listens on localhost only)")
    print("\n   On the phone: Share > Add to Home Screen for an app icon.")
    if token:
        print("   The page asks for the token once and remembers it." + ("" if a.show_token else " (`atlas phone --show-token` prints a link with it)"))
    if a.serve:
        print(f"\nrunning: {serve}")
        return subprocess.run(serve.split()).returncode
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="atlas", description="Inbox Atlas local backend")
    sub = ap.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("up", help="start the API, iMessage sidecar and Postgres")
    up.add_argument("--foreground", "-f", action="store_true", help="run the supervisor in this terminal")
    sub.add_parser("down", help="stop everything")
    st = sub.add_parser("status", help="show what is running")
    st.add_argument("--json", action="store_true")
    lg = sub.add_parser("logs", help="show logs")
    lg.add_argument("name", nargs="?", help="api | imessage | db | daemon")
    lg.add_argument("-f", "--follow", action="store_true")
    lg.add_argument("-n", type=int, default=40)
    ph = sub.add_parser("phone", help="phone access over Tailscale")
    ph.add_argument("--serve", action="store_true", help="also run `tailscale serve --bg PORT`")
    ph.add_argument("--show-token", action="store_true")
    sub.add_parser("install-service", help="write the macOS LaunchAgent")
    sub.add_parser("uninstall-service", help="remove the macOS LaunchAgent")
    a = ap.parse_args(argv)
    if a.cmd == "up":
        return cmd_up(a)
    if a.cmd == "down":
        return daemon.stop()
    if a.cmd == "status":
        return daemon.status(as_json=a.json)
    if a.cmd == "logs":
        return daemon.logs(a.name, a.follow, a.n)
    if a.cmd == "phone":
        return cmd_phone(a)
    if a.cmd == "install-service":
        return cmd_install_service(a)
    if a.cmd == "uninstall-service":
        return cmd_uninstall_service(a)
    return 2


if __name__ == "__main__":
    sys.exit(main())
