"""Phone width (390px) screenshot of the running web UI, plus a list of elements that overflow.

Headless Chrome will not lay out a window narrower than 500px, so the page is loaded in a 390px
iframe inside a same-origin wrapper page (written to web/_wrap.html for the run, then removed).

    uv run python scripts/phone_shot.py --port 8765 --out docs/img/phone.png [--height 844] [--path "?q=hackathon"]
"""

import argparse
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
WRAP = """<!doctype html><html><body style="margin:0;background:#888">
<iframe id="f" src="/%s" style="width:390px;height:%dpx;border:0;display:block;margin:0 auto"></iframe>
<script>setTimeout(()=>{const d=document.getElementById('f').contentDocument;const o=[];
d.querySelectorAll('*').forEach(e=>{const r=e.getBoundingClientRect();if(r.right>392)o.push(e.tagName+'.'+e.className+'#'+e.id+' w='+Math.round(r.width)+' r='+Math.round(r.right))});
const p=document.createElement('pre');p.id='DBG';p.textContent='scrollWidth='+d.documentElement.scrollWidth+'\\n'+o.join('\\n');document.body.append(p)},%d)</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="8765")
    ap.add_argument("--out")
    ap.add_argument("--height", type=int, default=844)
    ap.add_argument("--path", default="")
    ap.add_argument("--wait", type=int, default=4000)
    a = ap.parse_args()
    wrap = ROOT / "web" / "_wrap.html"
    wrap.write_text(WRAP % (a.path, a.height, a.wait))
    base = [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--virtual-time-budget={a.wait + 4000}"]
    url = f"http://127.0.0.1:{a.port}/_wrap.html"
    try:
        out = subprocess.run(base + ["--window-size=500,900", "--dump-dom", url], capture_output=True, text=True).stdout
        m = re.search(r'<pre id="DBG">(.*?)</pre>', out, re.S)
        print(m.group(1) if m else "no result")
        if a.out:
            subprocess.run(base + [f"--window-size=500,{a.height}", f"--screenshot={a.out}", url], capture_output=True)
            subprocess.run(["sips", "-c", str(a.height), "390", a.out], capture_output=True)
            print("saved", a.out)
    finally:
        os.remove(wrap)


if __name__ == "__main__":
    main()
