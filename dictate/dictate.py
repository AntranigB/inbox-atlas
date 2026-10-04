"""System-wide Whisperflow-style dictation for macOS.

Hold Right Option, speak, release. The audio goes to the running Inbox Atlas server
(/api/dictate) or straight to Grok if the server is down, and the cleaned text is pasted
into whatever app has focus. Your clipboard is restored afterwards.

    uv sync --extra dictate
    uv run python dictate/dictate.py [--key alt_r] [--style plain|email|message] [--no-paste]
"""

import argparse
import asyncio
import io
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import httpx
import numpy as np
import pyperclip
import sounddevice as sd
from pynput import keyboard

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RATE = 16000
MIN_SECONDS = 0.35
SOUNDS = Path("/System/Library/Sounds")


def beep(name: str):
    f = SOUNDS / f"{name}.aiff"
    if f.exists():
        subprocess.Popen(["afplay", "-v", "0.4", str(f)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        print("\a", end="", flush=True)


def to_wav(frames: list[np.ndarray]) -> bytes:
    pcm = np.concatenate(frames).astype(np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    return buf.getvalue()


def transcribe_via_server(server: str, wav: bytes, style: str) -> dict:
    r = httpx.post(f"{server}/api/dictate", files={"audio": ("dictation.wav", wav, "audio/wav")},
                   data={"style": style}, timeout=30)
    r.raise_for_status()
    return r.json()


def transcribe_direct(wav: bytes, style: str) -> dict:
    from atlas.voice.cleanup import cleanup
    from atlas.voice.stt import transcribe

    async def go():
        t0 = time.perf_counter()
        raw = (await transcribe(wav, "dictation.wav", "audio/wav")).get("text", "")
        text = await cleanup(raw, style)
        return {"raw": raw, "text": text, "ms": round((time.perf_counter() - t0) * 1000)}

    return asyncio.run(go())


def paste(text: str, kb: keyboard.Controller):
    try:
        old = pyperclip.paste()
    except Exception:
        old = None
    pyperclip.copy(text)
    time.sleep(0.05)
    with kb.pressed(keyboard.Key.cmd):
        kb.press("v")
        kb.release("v")
    time.sleep(0.35)  # let the target app read the clipboard before we restore it
    if old is not None:
        pyperclip.copy(old)


class Dictator:
    def __init__(self, args):
        self.args = args
        self.key = getattr(keyboard.Key, args.key, None) or keyboard.KeyCode.from_char(args.key)
        self.kb = keyboard.Controller()
        self.frames: list[np.ndarray] = []
        self.recording = False
        self.busy = False
        self.t0 = 0.0
        self.lock = threading.Lock()
        self.stream = sd.InputStream(samplerate=RATE, channels=1, dtype="int16", callback=self.on_audio)
        self.stream.start()

    def on_audio(self, indata, frames, t, status):
        if self.recording:
            self.frames.append(indata[:, 0].copy())

    def status(self, msg: str):
        print(f"\r\033[K{msg}", end="", flush=True)

    def on_press(self, key):
        if key != self.key or self.recording:
            return
        with self.lock:
            self.frames, self.recording, self.t0 = [], True, time.time()
        beep("Tink")
        self.status("recording... (release to send)")

    def on_release(self, key):
        if key != self.key or not self.recording:
            return
        with self.lock:
            self.recording = False
            frames, dur = self.frames, time.time() - self.t0
        beep("Pop")
        if dur < MIN_SECONDS or not frames:
            self.status("too short, ignored")
            return
        threading.Thread(target=self.finish, args=(frames, dur), daemon=True).start()

    def finish(self, frames, dur):
        wav = to_wav(frames)
        self.status(f"transcribing {dur:.1f}s...")
        via = "server"
        try:
            try:
                out = transcribe_via_server(self.args.server, wav, self.args.style)
            except httpx.HTTPError:
                via = "grok"
                out = transcribe_direct(wav, self.args.style)
        except Exception as e:
            self.status(f"failed: {e}\n")
            beep("Basso")
            return
        text = (out.get("text") or "").strip()
        if not text:
            self.status("heard nothing")
            return
        self.status(f"[{out.get('ms', '?')} ms via {via}] {text}\n")
        if self.args.verbose:
            print(f"  raw: {out.get('raw')}")
        if not self.args.no_paste:
            paste(text + (" " if self.args.trailing_space else ""), self.kb)
        self.status(f"ready. hold {self.args.key} to dictate")

    def run(self):
        print(f"Inbox Atlas dictate. Hold {self.args.key} to talk, release to paste. Ctrl+C quits.")
        print(f"server {self.args.server} (falls back to Grok directly), style {self.args.style}")
        self.status(f"ready. hold {self.args.key} to dictate")
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            try:
                listener.join()
            except KeyboardInterrupt:
                pass


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--key", default="alt_r", help="pynput Key name (alt_r, cmd_r, ctrl_r, f19) or a character")
    p.add_argument("--style", default="plain", choices=["plain", "email", "message", "query"])
    p.add_argument("--server", default="http://127.0.0.1:8765")
    p.add_argument("--no-paste", action="store_true", help="print only, do not paste")
    p.add_argument("--trailing-space", action="store_true", help="add a space after pasted text")
    p.add_argument("-v", "--verbose", action="store_true", help="also print the raw transcript")
    args = p.parse_args()
    try:
        Dictator(args).run()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
