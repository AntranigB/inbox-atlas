// Real iMessage transport over Photon Spectrum (spectrum-ts).
//
// Shared Spectrum lines will not cold start a chat. The owner must text the
// assigned line once first (or be added under Dashboard > Users and open the
// chat). After that, outbound sends land in that thread.

import fs from 'node:fs';
import path from 'node:path';
import { normalizeHandle, splitMessage } from './text.js';

export function spectrumConfigured(env = process.env) {
  return !!(env.SPECTRUM_PROJECT_ID && env.SPECTRUM_PROJECT_SECRET);
}

export class SpectrumTransport {
  constructor({ bot, owner, env = process.env, stateFile = null, log = console }) {
    this.bot = bot;
    this.owner = normalizeHandle(owner);
    this.env = env;
    this.stateFile = stateFile;
    this.log = log;
    this.spaces = new Map(); // handle -> Space seen inbound
    this.spaceIds = {}; // handle -> space id, persisted
    if (stateFile) {
      try {
        this.spaceIds = JSON.parse(fs.readFileSync(stateFile, 'utf8')).spaceIds || {};
      } catch {}
    }
    this.mode = 'spectrum';
  }

  async start() {
    const { Spectrum } = await import('spectrum-ts');
    const { imessage } = await import('spectrum-ts/providers/imessage');
    this.imessage = imessage;
    this.app = await Spectrum({
      projectId: this.env.SPECTRUM_PROJECT_ID,
      projectSecret: this.env.SPECTRUM_PROJECT_SECRET,
      providers: [imessage.config()],
    });
    this.im = imessage(this.app);
    this.log.log('[imessage] connected to Spectrum, listening for iMessages');
    this.loop = this.listen();
  }

  async listen() {
    try {
      for await (const [space, message] of this.app.messages) {
        // Do not block the stream on a slow agent call.
        this.onMessage(space, message).catch((err) => this.log.error('[imessage] handler failed:', err));
      }
    } catch (err) {
      this.log.error('[imessage] message stream ended:', err?.message || err);
    }
  }

  async onMessage(space, message) {
    if (message.direction && message.direction !== 'inbound') return;
    if (message.content?.type !== 'text') return;
    const sender = message.sender?.id || '';
    const text = message.content.text || '';
    this.log.log(`[imessage] <- ${sender}: ${text}`);
    if (!this.bot.allowed(sender)) {
      await this.bot.handle({ threadId: space.id, sender, text }); // logs the ignore
      return;
    }
    this.remember(sender, space);
    message.read().catch(() => {});
    const reply = await space.responding(() => this.bot.handle({ threadId: space.id, sender, text }));
    if (!reply) return;
    for (const chunk of splitMessage(reply)) {
      await space.send(chunk);
    }
    this.log.log(`[imessage] -> ${sender}: ${reply.slice(0, 120)}${reply.length > 120 ? '...' : ''}`);
  }

  remember(handle, space) {
    const h = normalizeHandle(handle);
    this.spaces.set(h, space);
    if (this.spaceIds[h] !== space.id) {
      this.spaceIds[h] = space.id;
      if (this.stateFile) {
        try {
          fs.mkdirSync(path.dirname(this.stateFile), { recursive: true });
          fs.writeFileSync(this.stateFile, JSON.stringify({ spaceIds: this.spaceIds }, null, 1));
        } catch {}
      }
    }
  }

  async spaceFor(handle) {
    if (this.spaces.has(handle)) return this.spaces.get(handle);
    const known = this.spaceIds[handle];
    for (const id of [known, `any;-;${handle}`].filter(Boolean)) {
      try {
        return await this.im.space.get(id);
      } catch {}
    }
    const user = await this.im.user(handle);
    return this.im.space.create(user);
  }

  async send(text, to) {
    const handle = normalizeHandle(to || this.owner);
    if (!handle) throw new Error('no recipient: set OWNER_PHONE or pass "to"');
    if (!this.im) throw new Error('Spectrum is not connected yet');
    try {
      const space = await this.spaceFor(handle);
      const chunks = splitMessage(text);
      for (const c of chunks) await space.send(c);
      return { ok: true, to: handle, spaceId: space.id, parts: chunks.length };
    } catch (err) {
      const detail = err?.message || String(err);
      if (/not allowed|cold|not registered|no chat/i.test(detail)) {
        const msg =
          `cannot start a chat with ${handle}: shared Spectrum lines only message people who texted the line first. ` +
          `Text the Photon number once from ${handle} (or add it under Dashboard > Users), then retry. Original error: ${detail}`;
        this.log.error('[imessage] ' + msg);
        throw new Error(msg);
      }
      throw err;
    }
  }

  async stop() {
    await this.app?.stop?.();
  }
}
