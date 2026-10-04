"""Move real Gmail rows out of the demo database into their own profile.

data/mail.sqlite           demo profile: fixture inbox + synthetic vault (safe for a projector)
data/personal/mail.sqlite  personal profile: real Gmail (run with ATLAS_DATA=data/personal ATLAS_DB=sqlite)

Raw .eml files stay in data/raw/gmail and are never touched.
"""

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atlas import store  # noqa: E402

DEMO = ROOT / "data" / "mail.sqlite"
PERSONAL = ROOT / "data" / "personal" / "mail.sqlite"
REAL = ("gmail",)


def main():
    src = sqlite3.connect(DEMO)
    src.row_factory = sqlite3.Row
    marks = ",".join("?" * len(REAL))
    rows = [dict(r) for r in src.execute(f"select * from emails where source in ({marks})", REAL)]
    print(f"real mail rows in demo db: {len(rows)}")
    if not rows:
        return
    PERSONAL.parent.mkdir(parents=True, exist_ok=True)
    dst = store.connect(PERSONAL)
    for i in range(0, len(rows), 2000):
        store.upsert_emails(dst, rows[i:i + 2000])
    n = dst.execute("select count(*) from emails").fetchone()[0]
    print(f"personal db now has {n} emails at {PERSONAL}")
    src.execute(f"delete from emails where source in ({marks})", REAL)
    src.commit()
    left = dict(src.execute("select source, count(*) from emails group by source").fetchall())
    print(f"demo db left with: {left}")


if __name__ == "__main__":
    main()
