// Inbox Atlas iMessage sidecar. Run: node src/index.js  (or --dry-run / MOCK=1 for the terminal mock)

import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { AtlasClient } from './atlas.js';
import { Bot } from './bot.js';
import { HistoryStore } from './history.js';
import { MockTransport } from './mock.js';
import { createServer } from './server.js';
import { SpectrumTransport, spectrumConfigured } from './spectrum.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
try {
  process.loadEnvFile(path.join(ROOT, '.env'));
} catch {}

const env = process.env;
const args = new Set(process.argv.slice(2));
const mock = args.has('--dry-run') || args.has('--mock') || env.MOCK === '1';
const port = Number(env.SIDECAR_PORT || 8766);
const owner = env.OWNER_PHONE || '';
const dataDir = env.ATLAS_DATA || path.join(ROOT, 'data');

const atlas = new AtlasClient(env.ATLAS_URL || 'http://localhost:8765');
const history = new HistoryStore(path.join(dataDir, mock ? 'imessage_history.mock.json' : 'imessage_history.json'));
const owners = [owner, ...(env.OWNER_HANDLES || '').split(',')].filter(Boolean);
const bot = new Bot({
  atlas,
  history,
  owners: mock && !owners.length ? ['+15555550100'] : owners,
  allowAny: env.ALLOW_ANY === '1',
  tz: env.TIMEZONE || 'America/New_York',
});

let transport;
if (mock) {
  transport = new MockTransport({ bot, owner: owner || '+15555550100', repl: !args.has('--no-repl') });
} else {
  if (!spectrumConfigured(env)) {
    console.error('[imessage] SPECTRUM_PROJECT_ID / SPECTRUM_PROJECT_SECRET not set. See docs/IMESSAGE.md, or run with --dry-run.');
    process.exit(1);
  }
  if (!owner) console.warn('[imessage] OWNER_PHONE not set: every inbound message will be ignored (set ALLOW_ANY=1 to answer anyone).');
  transport = new SpectrumTransport({ bot, owner, stateFile: path.join(dataDir, 'imessage_state.json') });
}

const server = createServer({ transport, owner });
server.listen(port, '127.0.0.1', async () => {
  console.log(`[imessage] sidecar on http://127.0.0.1:${port} (${transport.mode} mode), Atlas at ${atlas.base}`);
  try {
    await transport.start();
  } catch (err) {
    console.error('[imessage] could not start transport:', err?.message || err);
    process.exit(1);
  }
});

const shutdown = async () => {
  server.close();
  await transport.stop().catch(() => {});
  process.exit(0);
};
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
