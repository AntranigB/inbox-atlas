import assert from 'node:assert/strict';
import { PassThrough } from 'node:stream';
import test from 'node:test';
import { AtlasClient } from '../src/atlas.js';
import { Bot, formatAgenda, withAgendaHint } from '../src/bot.js';
import { HistoryStore } from '../src/history.js';
import { MockTransport } from '../src/mock.js';
import { createServer } from '../src/server.js';
import { normalizeHandle, plain, splitMessage } from '../src/text.js';

const OWNER = '+15555550100';

// Fake fetch that routes to handlers keyed by "METHOD /path".
function fakeFetch(routes, calls = []) {
  return async (url, init = {}) => {
    const u = new URL(url);
    const key = `${init.method || 'GET'} ${u.pathname}`;
    const body = init.body ? JSON.parse(init.body) : undefined;
    calls.push({ key, body, search: u.search });
    const h = routes[key];
    if (!h) return new Response('{"detail":"Not Found"}', { status: 404 });
    const out = await h(body, u);
    return new Response(JSON.stringify(out), { status: 200, headers: { 'content-type': 'application/json' } });
  };
}

function makeBot(routes, calls) {
  const atlas = new AtlasClient('http://atlas.test', fakeFetch(routes, calls));
  const log = { warn() {}, error() {}, log() {} };
  return new Bot({ atlas, history: new HistoryStore(null), owners: [OWNER], tz: 'America/New_York', log });
}

test('normalizeHandle', () => {
  assert.equal(normalizeHandle('(555) 555-0100'), OWNER);
  assert.equal(normalizeHandle('1 555 555 0100'), OWNER);
  assert.equal(normalizeHandle('+1 (555) 555-0100'), OWNER);
  assert.equal(normalizeHandle('Me@Example.com'), 'me@example.com');
});

test('splitMessage keeps chunks under 600 and loses nothing', () => {
  const para = 'Sentence number one is here. '.repeat(30);
  const text = [para, para, 'short tail'].join('\n\n');
  const parts = splitMessage(text, 600);
  assert.ok(parts.length > 2);
  for (const p of parts) assert.ok(p.length <= 600, `chunk too long: ${p.length}`);
  assert.equal(parts.join(' ').replace(/\s+/g, ' '), text.replace(/\s+/g, ' ').trim());
  assert.deepEqual(splitMessage('hi'), ['hi']);
  assert.deepEqual(splitMessage(''), []);
});

test('plain strips markdown and dashes', () => {
  const out = plain('## Today\n**Spin** at 5pm \u2014 Helen Newman\n* demo day');
  assert.ok(!/[*#\u2014\u2013]/.test(out), out);
  assert.match(out, /Spin at 5pm, Helen Newman/);
});

test('ignores strangers, answers owner', async () => {
  const calls = [];
  const bot = makeBot({ 'POST /api/ask': () => ({ reply: 'hi there' }) }, calls);
  assert.equal(await bot.handle({ threadId: 't', sender: '+19999999999', text: 'hello' }), null);
  assert.equal(calls.length, 0);
  assert.equal(await bot.handle({ threadId: 't', sender: '5555550100', text: 'hello' }), 'hi there');
  bot.allowAny = true;
  assert.equal(await bot.handle({ threadId: 't', sender: '+19999999999', text: 'hello' }), 'hi there');
});

test('ask sends channel imessage and rolling history', async () => {
  const calls = [];
  let n = 0;
  const bot = makeBot({ 'POST /api/ask': () => ({ reply: `r${++n}` }) }, calls);
  for (let i = 0; i < 10; i++) await bot.handle({ threadId: 't', sender: OWNER, text: `q${i}` });
  const last = calls.at(-1).body;
  assert.equal(last.channel, 'imessage');
  assert.equal(last.text, 'q9');
  assert.equal(last.history.length, 16); // 8 turns
  assert.deepEqual(last.history.at(-1), { role: 'assistant', content: 'r9' });
});

test('today asks the agent for todays_agenda with the local date', async () => {
  const calls = [];
  const bot = makeBot({ 'POST /api/ask': () => ({ reply: 'Spin at 5pm' }) }, calls);
  const r = await bot.handle({ threadId: 't', sender: OWNER, text: 'Today' });
  assert.equal(r, 'Spin at 5pm');
  assert.match(calls[0].body.text, /What do I have today\?/);
  assert.match(calls[0].body.text, /todays_agenda/);
  assert.match(calls[0].body.text, /\d{4}-\d{2}-\d{2}/);
});

test('agenda hint uses the right date and tomorrow', () => {
  const now = new Date('2026-10-04T13:00:00Z');
  assert.match(withAgendaHint('what do I have today?', 'America/New_York', now), /date=2026-10-04/);
  assert.match(withAgendaHint('anything tomorrow?', 'America/New_York', now), /date=2026-10-05/);
  assert.equal(withAgendaHint('did I get anything about internships', 'America/New_York', now), 'did I get anything about internships');
});

test('ask 404 gives a clear message, today falls back to /api/agenda', async () => {
  const calls = [];
  const bot = makeBot(
    {
      'GET /api/notify/agenda': () => ({
        events: [],
        emails: [{ from_name: 'Cornell Fitness', subject: 'Gym class booking confirmed' }],
      }),
    },
    calls,
  );
  const r1 = await bot.handle({ threadId: 't', sender: OWNER, text: 'any internship emails?' });
  assert.match(r1, /not wired up/);
  const r2 = await bot.handle({ threadId: 't', sender: OWNER, text: 'today' });
  assert.match(r2, /Gym class booking confirmed/);
});

test('unreachable server gives a clear message', async () => {
  const atlas = new AtlasClient('http://atlas.test', async () => {
    throw new Error('ECONNREFUSED');
  });
  const bot = new Bot({ atlas, history: new HistoryStore(null), owners: [OWNER], log: { error() {} } });
  assert.match(await bot.handle({ threadId: 't', sender: OWNER, text: 'hi' }), /cannot reach Inbox Atlas/);
});

test('watch, watches, stop, brief commands hit the notify API', async () => {
  const calls = [];
  const watches = [];
  const bot = makeBot(
    {
      'POST /api/notify/watches': (b) => {
        watches.push({ id: 'w1', name: b.name });
        return { id: 'w1', name: b.name };
      },
      'GET /api/notify/watches': () => watches,
      'DELETE /api/notify/watches/internships': () => ({ removed: 1 }),
      'POST /api/notify/brief': (b) => ({ enabled: b.enabled, time: '08:00' }),
    },
    calls,
  );
  const h = (text) => bot.handle({ threadId: 't', sender: OWNER, text });
  assert.match(await h('watches'), /No watches/);
  assert.match(await h('watch internships'), /Watching "internships"/);
  assert.match(await h('watches'), /- internships/);
  assert.match(await h('stop internships'), /Stopped/);
  assert.match(await h('brief on'), /on.*08:00/);
  assert.match(await h('brief off'), /off/);
  assert.match(await h('help'), /watch <topic>/);
});

test('sidecar HTTP /send and /mock/inbound in mock mode', async () => {
  const out = new PassThrough();
  let printed = '';
  out.on('data', (d) => (printed += d));
  const bot = makeBot({ 'POST /api/ask': (b) => ({ reply: `echo: ${b.text}` }) });
  const transport = new MockTransport({ bot, owner: OWNER, output: out, repl: false });
  const server = createServer({ transport, owner: OWNER });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const base = `http://127.0.0.1:${server.address().port}`;
  try {
    let r = await fetch(`${base}/send`, { method: 'POST', body: JSON.stringify({ text: 'New email in internships: Jane: Offer' }) });
    assert.equal(r.status, 200);
    assert.equal((await r.json()).to, OWNER);
    assert.match(printed, /atlas -> \+15555550100: New email in internships/);
    r = await fetch(`${base}/send`, { method: 'POST', body: '{}' });
    assert.equal(r.status, 400);
    r = await fetch(`${base}/mock/inbound`, { method: 'POST', body: JSON.stringify({ text: 'hello' }) });
    assert.equal((await r.json()).reply, 'echo: hello');
    r = await fetch(`${base}/health`);
    assert.equal((await r.json()).mode, 'mock');
  } finally {
    server.close();
  }
});

test('formatAgenda', () => {
  const s = formatAgenda(
    { events: [{ title: 'Demo day', start: 1791118800, location: 'PSB' }], emails: [] },
    'America/New_York',
  );
  assert.match(s, /Demo day @ PSB/);
  assert.match(formatAgenda({}, 'UTC'), /Nothing/);
});
