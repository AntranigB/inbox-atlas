// Real messaging transport over Photon Spectrum (spectrum-ts).
//
// SPECTRUM_PROVIDERS picks the platforms (comma list, default "imessage"):
//   imessage            Photon iMessage line (project id + secret only)
//   whatsapp            WhatsApp Business: WHATSAPP_ACCESS_TOKEN + WHATSAPP_PHONE_NUMBER_ID
//                       (+ optional WHATSAPP_APP_SECRET), or enable it in the Photon dashboard
//                       and leave them empty (cloud mode)
//   telegram            Telegram bot: TELEGRAM_BOT_TOKEN (+ optional TELEGRAM_WEBHOOK_SECRET)
//   slack, terminal     also shipped by spectrum-ts; slack is configured in the Photon dashboard
// Inbound from every provider arrives on the same app.messages stream and goes to the same Bot.
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

const ALIASES = { whatsapp: 'whatsapp-business', whatsapp_business: 'whatsapp-business', wa: 'whatsapp-business', tg: 'telegram' };

export function providerNames(env = process.env) {
  const raw = (env.SPECTRUM_PROVIDERS || 'imessage').split(',').map((s) => s.trim().toLowerCase()).filter(Boolean);
  return [...new Set(raw.map((n) => ALIASES[n] || n))];
}

// Platform name as it appears on message.platform, used to prefix session ids.
export function platformKey(name) {
  return name === 'whatsapp-business' ? 'whatsapp_business' : name;
}

// Build [name, config] pairs for Spectrum({providers}). Unknown or unconfigured ones are skipped with a warning.
export async function loadProviders(env = process.env, log = console, importer = (p) => import(p)) {
  const out = [];
  for (const name of providerNames(env)) {
    try {
      if (name === 'imessage') {
        const { imessage } = await importer('spectrum-ts/providers/imessage');
        out.push({ name, mod: imessage, config: imessage.config() });
      } else if (name === 'whatsapp-business') {
        const { whatsappBusiness } = await importer('spectrum-ts/providers/whatsapp-business');
        const direct = env.WHATSAPP_ACCESS_TOKEN && env.WHATSAPP_PHONE_NUMBER_ID;
        const cfg = direct
          ? { accessToken: env.WHATSAPP_ACCESS_TOKEN, phoneNumberId: env.WHATSAPP_PHONE_NUMBER_ID, ...(env.WHATSAPP_APP_SECRET ? { appSecret: env.WHATSAPP_APP_SECRET } : {}) }
          : {};
        out.push({ name, mod: whatsappBusiness, config: whatsappBusiness.config(cfg) });
      } else if (name === 'telegram') {
        if (!env.TELEGRAM_BOT_TOKEN) {
          log.warn('[imessage] telegram provider skipped: set TELEGRAM_BOT_TOKEN');
          continue;
        }
        const { telegram } = await importer('spectrum-ts/providers/telegram');
        const cfg = { botToken: env.TELEGRAM_BOT_TOKEN, ...(env.TELEGRAM_WEBHOOK_SECRET ? { webhookSecret: env.TELEGRAM_WEBHOOK_SECRET } : {}) };
        out.push({ name, mod: telegram, config: telegram.config(cfg) });
      } else if (name === 'slack' || name === 'terminal') {
        const mod = (await importer(`spectrum-ts/providers/${name}`))[name];
        out.push({ name, mod, config: mod.config() });
      } else {
        log.warn(`[imessage] unknown provider "${name}" in SPECTRUM_PROVIDERS, skipped`);
      }
    } catch (err) {
      log.error(`[imessage] provider ${name} failed to load: ${err?.message || err}`);
    }
  }
  return out;
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
    this.connected = false;
    this.providers = [];
    this.lastError = null;
  }

  async start() {
    const { Spectrum } = await import('spectrum-ts');
    const loaded = await loadProviders(this.env, this.log);
    if (!loaded.length) throw new Error('no Spectrum providers configured (SPECTRUM_PROVIDERS)');
    this.app = await Spectrum({
      projectId: this.env.SPECTRUM_PROJECT_ID,
      projectSecret: this.env.SPECTRUM_PROJECT_SECRET,
      providers: loaded.map((p) => p.config),
    });
    this.providers = loaded.map((p) => p.name);
    const im = loaded.find((p) => p.name === 'imessage');
    if (im) this.im = im.mod(this.app);
    this.connected = true;
    this.log.log(`[imessage] connected to Spectrum (${this.providers.join(', ')}), listening`);
    this.loop = this.listen();
  }

  threads() {
    return Object.keys(this.spaceIds);
  }

  async listen() {
    try {
      for await (const [space, message] of this.app.messages) {
        // Do not block the stream on a slow agent call.
        this.onMessage(space, message).catch((err) => this.log.error('[imessage] handler failed:', err));
      }
    } catch (err) {
      this.lastError = err?.message || String(err);
      this.log.error('[imessage] message stream ended:', this.lastError);
    }
    this.connected = false;
    // index.js exits so the atlas daemon restarts us with a fresh connection.
    if (!this.stopping) this.onEnd?.();
  }

  async onMessage(space, message) {
    if (message.direction && message.direction !== 'inbound') return;
    if (message.content?.type !== 'text') return;
    const sender = message.sender?.id || '';
    const text = message.content.text || '';
    const platform = platformKey(message.platform || space.__platform || 'imessage');
    this.log.log(`[imessage] <- ${platform !== 'imessage' ? platform + ' ' : ''}${sender}: ${text}`);
    if (!this.bot.allowed(sender)) {
      await this.bot.handle({ threadId: space.id, sender, text, platform }); // logs the ignore
      return;
    }
    this.remember(platform === 'imessage' ? sender : `${platform}:${sender}`, space);
    message.read?.().catch?.(() => {});
    const handle = () => this.bot.handle({ threadId: space.id, sender, text, platform });
    const reply = typeof space.responding === 'function' ? await space.responding(handle) : await handle();
    if (!reply) return;
    for (const chunk of splitMessage(reply)) {
      await space.send(chunk);
    }
    this.log.log(`[imessage] -> ${sender}: ${reply.slice(0, 120)}${reply.length > 120 ? '...' : ''}`);
  }

  remember(handle, space) {
    const h = handle.includes(':') ? handle : normalizeHandle(handle);
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

  async send(text, to, platform) {
    platform = platformKey(platform || 'imessage');
    if (platform !== 'imessage') {
      // Other providers: only reply into a chat the person opened with us.
      const key = `${platform}:${to || ''}`;
      const space = this.spaces.get(key);
      if (!space) throw new Error(`no open ${platform} chat with ${to || '(nobody)'}: they have to message the bot first`);
      const chunks = splitMessage(text);
      for (const c of chunks) await space.send(c);
      return { ok: true, to, platform, spaceId: space.id, parts: chunks.length };
    }
    const handle = normalizeHandle(to || this.owner);
    if (!handle) throw new Error('no recipient: set OWNER_PHONE or pass "to"');
    if (!this.im) throw new Error(this.providers.length ? 'imessage is not in SPECTRUM_PROVIDERS' : 'Spectrum is not connected yet');
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
    this.stopping = true;
    await this.app?.stop?.();
  }
}
