"""Demo mode: load tests/fixtures/mailbox.jsonl into data/mail.sqlite (no credentials needed).

    uv run python -m atlas.ingest.load_fixture [path.jsonl]
"""

import sys

from atlas import config, store


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    conn = store.connect()
    rows = store.load_fixture(conn, argv[0] if argv else None)
    print(f"loaded {len(rows)} fixture emails into {config.DB_PATH}")
    print("next: uv run python -m atlas.index.build")
    return 0


if __name__ == "__main__":
    sys.exit(main())
