"""Build the embedding index for every email in data/mail.sqlite.

    uv run python -m atlas.index.build --encoder base
    uv run python -m atlas.index.build --encoder models/atlas-embed --no-grok
"""

import argparse
import sys
import time

from atlas import config, store
from atlas.index import build_index
from atlas.model.encoder import load_encoder


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default=None, help="hash | base | path to a trained checkpoint (default ATLAS_ENCODER)")
    ap.add_argument("--no-map", action="store_true", help="skip UMAP/HDBSCAN")
    ap.add_argument("--no-grok", action="store_true", help="do not label clusters with Grok")
    a = ap.parse_args(argv)
    conn = store.connect()
    n = len(store.all_ids(conn))
    if not n:
        print(f"no emails in {config.DB_PATH}. Run atlas.ingest.gmail_export or atlas.ingest.load_fixture first.")
        return 1
    t0 = time.time()
    enc = load_encoder(a.encoder)
    idx = build_index(enc, conn, with_map=not a.no_map, grok=not a.no_grok and bool(config.XAI_API_KEY), verbose=True)
    print(f"index {enc.name}: {len(idx)} emails, dim {idx.dim}, {len(idx.map.get('clusters', []))} clusters, "
          f"{time.time() - t0:.0f}s -> {config.INDEX_DIR / enc.name}")
    for c in idx.map.get("clusters", [])[:15]:
        print(f"  [{c['id']:>3}] {c['size']:>5}  {c['label']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
