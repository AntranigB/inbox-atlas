import assert from 'node:assert/strict';
import test from 'node:test';
import { SpectrumTransport } from '../src/spectrum.js';

const OWNER = '+15555550100';
const quiet = { log() {}, warn() {}, error() {} };

function fakeSpace(id) {
  const s = { id, sent: [], typing: 0 };
  s.send = async (t) => {
    s.sent.push(t);
    return { id: `m${s.sent.length}` };
  };
  s.responding = async (fn) => {
    s.typing++;
    return fn();
  };
  return s;
}

function fakeMessage(sender, text, extra = {}) {
  return {
    direction: 'inbound',
    content: { type: 'text', text },
    sender: { id: sender },
    read: async () => {
      fakeMessage.reads++;
    },
    ...extra,
  };
}
fakeMessage.reads = 0;

function fakeBot(reply) {
  return {
    allowed: (s) => s === OWNER,
    handle: async ({ sender }) => (sender === OWNER ? reply : null),
  };
}

test('inbound owner message: read receipt, typing, split reply in same space', async () => {
  const long = 'word '.repeat(300).trim();
  const t = new SpectrumTransport({ bot: fakeBot(long), owner: OWNER, log: quiet });
  const space = fakeSpace('any;-;' + OWNER);
  await t.onMessage(space, fakeMessage(OWNER, 'hi'));
  assert.equal(space.typing, 1);
  assert.equal(fakeMessage.reads, 1);
  assert.ok(space.sent.length >= 3);
  assert.ok(space.sent.every((c) => c.length <= 600));
  assert.equal(t.spaces.get(OWNER), space);
});

test('strangers, outbound echoes and non text are ignored', async () => {
  const t = new SpectrumTransport({ bot: fakeBot('x'), owner: OWNER, log: quiet });
  const space = fakeSpace('s');
  await t.onMessage(space, fakeMessage('+19999999999', 'hi'));
  await t.onMessage(space, fakeMessage(OWNER, 'hi', { direction: 'outbound' }));
  await t.onMessage(space, fakeMessage(OWNER, '', { content: { type: 'reaction' } }));
  assert.deepEqual(space.sent, []);
});

test('outbound send reuses the remembered space, cold start error is explained', async () => {
  const t = new SpectrumTransport({ bot: fakeBot('x'), owner: OWNER, log: quiet });
  const space = fakeSpace('dm');
  t.im = {
    space: {
      get: async () => {
        throw new Error('nope');
      },
      create: async () => ({
        id: 'new',
        send: async () => {
          throw new Error('Target not allowed');
        },
      }),
    },
    user: async (h) => ({ id: h }),
  };
  await assert.rejects(t.send('hello'), /texted the line first/);
  t.remember(OWNER, space);
  const r = await t.send('hello');
  assert.equal(r.spaceId, 'dm');
  assert.deepEqual(space.sent, ['hello']);
});
