// HTTP API of the sidecar (port 8766): POST /send, GET /health, and in mock mode POST /mock/inbound.

import http from 'node:http';

function readJson(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', (c) => {
      data += c;
      if (data.length > 1e6) reject(new Error('body too large'));
    });
    req.on('end', () => {
      if (!data) return resolve({});
      try {
        resolve(JSON.parse(data));
      } catch {
        reject(new Error('invalid JSON'));
      }
    });
    req.on('error', reject);
  });
}

function reply(res, status, obj) {
  res.writeHead(status, { 'content-type': 'application/json' });
  res.end(JSON.stringify(obj));
}

export function createServer({ transport, owner, atlas = null }) {
  return http.createServer(async (req, res) => {
    const url = new URL(req.url, 'http://localhost');
    try {
      if (req.method === 'GET' && url.pathname === '/health') {
        return reply(res, 200, {
          ok: true,
          mode: transport.mode,
          owner: owner || null,
          connected: transport.mode === 'mock' ? true : !!transport.connected,
          providers: transport.providers || [transport.mode === 'mock' ? 'mock' : 'imessage'],
          threads: transport.threads ? transport.threads() : [],
          last_error: transport.lastError || null,
          atlas: atlas?.base || null,
        });
      }
      if (req.method === 'POST' && url.pathname === '/send') {
        const body = await readJson(req);
        const text = String(body.text || '').trim();
        if (!text) return reply(res, 400, { ok: false, error: 'text is required' });
        const r = await transport.send(text, body.to, body.platform);
        return reply(res, 200, r);
      }
      if (req.method === 'POST' && url.pathname === '/mock/inbound' && transport.mode === 'mock') {
        const body = await readJson(req);
        const parts = await transport.inbound(String(body.text || ''), body.from || undefined);
        return reply(res, 200, { ok: true, parts, reply: parts.join('\n') });
      }
      return reply(res, 404, { ok: false, error: 'not found' });
    } catch (err) {
      return reply(res, 502, { ok: false, error: err?.message || String(err) });
    }
  });
}
