"""Talk to mcp/server.py over stdio like an agent would and print what the demo tools return.

    uv run python scripts/mcp_check.py              # the two demo questions
    uv run python scripts/mcp_check.py "question"   # any question

Prints the token count, answerable/related flag and the context string for atlas_context and
atlas_points_of_interest. Exit code 1 if the server does not list both tools.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path = [p for p in sys.path if Path(p or ".").resolve() != ROOT / "mcp"]

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

QUESTIONS = ["what pulse widths did I use for the arm servos?", "did I book a yacht charter?"]


def payload(res):
    if getattr(res, "structuredContent", None):
        r = res.structuredContent
        return r.get("result", r) if isinstance(r, dict) and set(r) == {"result"} else r
    for c in res.content:
        if getattr(c, "text", None):
            try:
                return json.loads(c.text)
            except ValueError:
                return {"text": c.text}
    return {}


async def main(questions):
    params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "mcp" / "server.py")], cwd=str(ROOT))
    quiet = open(os.devnull, "w")  # model loading logs go to the server's stderr
    async with stdio_client(params, errlog=quiet) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        names = {t.name for t in (await s.list_tools()).tools}
        print(f"tools: {', '.join(sorted(names))}")
        if not {"atlas_context", "atlas_points_of_interest"} <= names:
            return 1
        for q in questions:
            print(f"\n=== {q}")
            for tool in ("atlas_points_of_interest", "atlas_context"):
                p = payload(await s.call_tool(tool, {"question": q}))
                flag = p.get("related", p.get("answerable"))
                print(f"--- {tool}: {p.get('tokens')} tokens, {'related' if 'related' in p else 'answerable'}={flag}")
                print(p.get("context") or p.get("reason") or "")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:] or QUESTIONS)))
