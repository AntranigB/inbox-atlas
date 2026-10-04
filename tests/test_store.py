from atlas import store
from atlas.model.encoder import load_encoder


def test_fixture_and_fts():
    conn = store.connect(":memory:")
    rows = store.load_fixture(conn)
    assert len(store.all_ids(conn)) == len(rows)
    hits = store.fts_search(conn, "Codeforces round")
    assert hits and store.get_email(conn, hits[0][0])["from_name"] == "Codeforces"


def test_hash_encoder_normalized():
    E = load_encoder("hash").encode_docs(["a b c", "hello world"])
    assert E.shape == (2, 256) and abs((E ** 2).sum(1) - 1).max() < 1e-5
