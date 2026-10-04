-- Inbox Atlas on Tiger Data (Postgres + pgvector). Idempotent: safe to run on every connect.
-- Works on the local timescale/timescaledb-ha image and on Tiger Cloud unchanged.
create extension if not exists vector;

-- Emails mirror the SQLite store, plus a pointer back to the raw .eml on disk.
create table if not exists emails(
  id text primary key,
  thread_id text,
  from_addr text,
  from_name text,
  to_addrs text,
  date bigint,                -- unix seconds UTC
  subject text,
  body text,
  snippet text,
  labels jsonb,
  source text,                -- 'gmail' | 'fixture' | 'enron'
  raw_path text,              -- data/raw/gmail/<id>.eml for gmail rows, else null
  tsv tsvector generated always as (
    setweight(to_tsvector('english', coalesce(subject, '')), 'A') ||
    setweight(to_tsvector('english', coalesce(from_name, '')), 'B') ||
    setweight(to_tsvector('english', coalesce(body, '')), 'C')) stored
);
create index if not exists emails_tsv_gin on emails using gin(tsv);
create index if not exists emails_date on emails(date);

-- One row per (email, encoder). Every encoder lives in the same vector(768) column:
-- Matryoshka-truncated encoders (dim 256 or 64) and the 256-d hash encoder are zero-padded
-- to 768. Both sides are L2-normalized and the padding is zero, so cosine distance is
-- exactly the cosine in the truncated space. `dim` records the real size.
create table if not exists email_vectors(
  email_id text not null references emails(id) on delete cascade,
  encoder text not null,
  dim int not null,
  embedding vector(768) not null,
  primary key(email_id, encoder)
);
create index if not exists email_vectors_hnsw on email_vectors using hnsw (embedding vector_cosine_ops);
-- No btree on encoder: it would win over HNSW for the filtered knn. pgvector 0.8 iterative
-- scans (set per connection in pg.py) keep the HNSW scan going until k rows pass the filter.
drop index if exists email_vectors_encoder;

-- Chunks of long documents. A chunk points at an email (doc_id = email id, uri = raw .eml path)
-- or a note (source = 'obsidian', doc_id = note path, uri = path#heading).
-- span_start / span_end are character offsets into the document text.
create table if not exists chunks(
  id text primary key,
  doc_id text not null,
  source text not null,
  uri text,
  heading text,
  span_start int,
  span_end int,
  text text,
  created timestamptz not null default now(),
  tsv tsvector generated always as (to_tsvector('english', coalesce(heading, '') || ' ' || coalesce(text, ''))) stored
);
create index if not exists chunks_doc on chunks(source, doc_id);
create index if not exists chunks_tsv_gin on chunks using gin(tsv);

create table if not exists chunk_vectors(
  chunk_id text not null references chunks(id) on delete cascade,
  encoder text not null,
  dim int not null,
  embedding vector(768) not null,
  primary key(chunk_id, encoder)
);
create index if not exists chunk_vectors_hnsw on chunk_vectors using hnsw (embedding vector_cosine_ops);

-- Grok chat history, shared by web, voice and iMessage.
create table if not exists chat_sessions(
  id text primary key,
  channel text,
  user_handle text,
  title text,
  created timestamptz not null default now(),
  updated timestamptz not null default now()
);
create table if not exists chat_messages(
  id bigserial primary key,
  session_id text not null references chat_sessions(id) on delete cascade,
  role text not null,          -- user | assistant | tool
  content text,
  tool_name text,
  tool_args jsonb,
  hits jsonb,                  -- email ids the turn returned
  created timestamptz not null default now()
);
create index if not exists chat_messages_session on chat_messages(session_id, id);

-- Mirrors of the SQLite events and watches tables.
create table if not exists events(id text primary key, title text, start bigint, "end" bigint, location text, source text);
create table if not exists watches(id text primary key, name text, positive text, negative text, created bigint, last_checked bigint);

-- Token-efficiency metrics. Turned into a TimescaleDB hypertable by pg.init_db when the
-- extension is available, plain table otherwise.
create table if not exists query_log(
  ts timestamptz not null default now(),
  channel text,
  query text,
  n_facets int,
  region_size int,
  tokens_returned int,
  latency_ms double precision
);
create index if not exists query_log_ts on query_log(ts desc);
