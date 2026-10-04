"""Write one row to query_log (a TimescaleDB hypertable in Postgres) per question. Never raises."""

import logging
import time
from contextlib import contextmanager

log = logging.getLogger("atlas.qlog")


def write(channel, query, n_facets=None, region_size=None, tokens_returned=None, latency_ms=None):
    try:
        from atlas.search.engine import get_engine

        get_engine().backend.log_query(channel, query, n_facets=n_facets, region_size=region_size,
                                       tokens_returned=tokens_returned, latency_ms=latency_ms)
    except Exception as e:
        log.debug("query_log write skipped: %s", e)


@contextmanager
def timed(channel, query):
    """with timed("web", q) as row: ...; row.update(region_size=..., tokens_returned=...)"""
    row, t0 = {}, time.time()
    try:
        yield row
    finally:
        write(channel, query, latency_ms=(time.time() - t0) * 1000, **row)
