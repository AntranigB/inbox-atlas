// Thin client for the Inbox Atlas API server (server.py, port 8765).

export class AtlasError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.status = status;
  }
}

export class AtlasClient {
  constructor(baseUrl = 'http://localhost:8765', fetchImpl = globalThis.fetch, timeoutMs = 90000) {
    this.base = baseUrl.replace(/\/$/, '');
    this.fetch = fetchImpl;
    this.timeoutMs = timeoutMs;
  }

  async call(method, path, body) {
    let res;
    try {
      res = await this.fetch(this.base + path, {
        method,
        headers: body ? { 'content-type': 'application/json' } : {},
        body: body ? JSON.stringify(body) : undefined,
        signal: AbortSignal.timeout(this.timeoutMs),
      });
    } catch (err) {
      throw new AtlasError(`cannot reach Inbox Atlas at ${this.base} (${err.message}). Is server.py running?`, 0);
    }
    if (!res.ok) {
      let detail = '';
      try {
        detail = (await res.text()).slice(0, 200);
      } catch {}
      throw new AtlasError(`${method} ${path} returned ${res.status} ${detail}`.trim(), res.status);
    }
    const txt = await res.text();
    try {
      return JSON.parse(txt);
    } catch {
      return txt;
    }
  }

  ask(text, history = []) {
    return this.call('POST', '/api/ask', { text, channel: 'imessage', history });
  }

  // Served by atlas/api/messaging.py: todays_agenda when search-agent is merged, else a local fallback.
  agenda(date) {
    return this.call('GET', `/api/notify/agenda${date ? `?date=${encodeURIComponent(date)}` : ''}`);
  }

  watches() {
    return this.call('GET', '/api/notify/watches');
  }

  addWatch(topic) {
    return this.call('POST', '/api/notify/watches', { name: topic, topic });
  }

  stopWatch(name) {
    return this.call('DELETE', `/api/notify/watches/${encodeURIComponent(name)}`);
  }

  brief(enabled) {
    return this.call('POST', '/api/notify/brief', { enabled });
  }
}
