// Mock iMessage transport: a terminal REPL that plays both sides of the chat.
// Lines you type are iMessages from OWNER_PHONE. Outbound texts (POST /send)
// print as if they arrived on your phone.

import readline from 'node:readline';
import { normalizeHandle, splitMessage } from './text.js';

export class MockTransport {
  constructor({ bot, owner, input = process.stdin, output = process.stdout, repl = true }) {
    this.bot = bot;
    this.owner = normalizeHandle(owner) || '+15555550100';
    this.input = input;
    this.output = output;
    this.repl = repl;
    this.mode = 'mock';
    this.sent = [];
    this.threadId = `mock;-;${this.owner}`;
  }

  print(line) {
    this.output.write(line + '\n');
  }

  bubble(from, text) {
    const parts = splitMessage(text);
    parts.forEach((p, i) => {
      const tag = parts.length > 1 ? ` (${i + 1}/${parts.length})` : '';
      this.print(`${from}${tag}: ${p.replace(/\n/g, '\n    ')}`);
    });
    return parts;
  }

  async start() {
    this.print(`[mock iMessage] you are ${this.owner}. Type a message, "help" for commands, Ctrl-C to quit.`);
    if (!this.repl) return;
    const rl = readline.createInterface({ input: this.input, terminal: false });
    // Handle lines one at a time so piped input keeps conversation order.
    let chain = Promise.resolve();
    rl.on('line', (line) => {
      chain = chain.then(() => this.inbound(line, this.owner).catch((e) => this.print(`[error] ${e.message}`)));
    });
  }

  // Simulate an inbound iMessage. Returns the reply parts.
  async inbound(text, from = this.owner) {
    if (!String(text).trim()) return [];
    if (!this.repl || !this.input.isTTY) this.print(`you: ${text}`);
    if (this.bot.allowed(from)) this.print('atlas is typing...');
    const reply = await this.bot.handle({ threadId: `mock;-;${normalizeHandle(from)}`, sender: from, text });
    if (!reply) {
      this.print(`[ignored message from ${from}]`);
      return [];
    }
    return this.bubble('atlas', reply);
  }

  async send(text, to) {
    const handle = normalizeHandle(to || this.owner);
    const parts = this.bubble(`atlas -> ${handle}`, text);
    this.sent.push({ to: handle, text });
    return { ok: true, to: handle, parts: parts.length, mock: true };
  }

  async stop() {}
}
