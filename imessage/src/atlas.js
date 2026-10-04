// Thin client for the Inbox Atlas API server (server.py, port 8765).

export class AtlasError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.status = status;
  }
}

export class AtlasClient {
  constructor(baseUrl = 'http://localhost:8765', fetchImpl = globalThis.fetch, timeoutMs = 90000, token = '') {
    this.base = baseUrl.replace(/\/$/, '');
    this.fetch = fetchImpl;
    this.timeoutMs = timeoutMs;
    this.token = token;
    this.noChat = false;
  }

  async call(method, path, body) {
    let res;
    try {
      res = await this.fetch(this.base + path, {
        method,
        headers: {
          ...(body ? { 'content-type': 'application/json' } : {}),
          ...(this.token ? { authorization: `Bearer ${this.token}` } : {}),
        },
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

  // With a session id the server keeps the history (POST /api/chat, shared with the web UI's
  // session list). Older servers without /api/chat get the local history on /api/ask.
  async ask(text, history = [], sessionId = null) {
    if (sessionId && !this.noChat) {
      try {
        return await this.call('POST', '/api/chat', { text, channel: 'imessage', session_id: sessionId });
      } catch (err) {
        if (!(err instanceof AtlasError) || (err.status !== 404 && err.status !== 405)) throw err;
        this.noChat = true;
      }
    }
    return this.call('POST', '/api/ask', { text, channel: 'imessage', history });
  }

  resetSession(sessionId) {
    return this.call('DELETE', `/api/chat/sessions/${encodeURIComponent(sessionId)}`).catch(() => null);
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
