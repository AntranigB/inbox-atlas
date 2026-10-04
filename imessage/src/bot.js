// Transport independent brain of the sidecar: commands, history, calls to Inbox Atlas.

import { AtlasError } from './atlas.js';
import { localDate, normalizeHandle, plain } from './text.js';

export const HELP = [
  'Inbox Atlas on iMessage. Text me anything about your email, like "did I get anything about internships?"',
  'today: what you have today',
  'watch <topic>: text you when new mail about it arrives',
  'watches: list your watches',
  'stop <name>: remove a watch',
  'brief on / brief off: morning brief',
  'reset: forget this conversation',
].join('\n');

const TODAY_Q = 'What do I have today?';
const DAY_WORDS = /\b(today|tonight|this (morning|afternoon|evening)|my day|schedule|agenda|calendar|tomorrow)\b/i;

// Add a hint so the agent reaches for todays_agenda with the right local date.
export function withAgendaHint(text, tz, now = new Date()) {
  if (!DAY_WORDS.test(text)) return text;
  const tomorrow = /\btomorrow\b/i.test(text);
  const date = localDate(tz, tomorrow ? 1 : 0, now);
  return `${text}\n[context: the user's local date ${tomorrow ? 'tomorrow is' : 'today is'} ${date} (${tz}). Use the todays_agenda tool with date=${date} and answer in short plain text.]`;
}

function fmtTime(v, tz) {
  if (v == null || v === '') return '';
  const d = typeof v === 'number' ? new Date(v * 1000) : new Date(v);
  if (Number.isNaN(d.getTime())) return String(v);
  return d.toLocaleTimeString('en-US', { timeZone: tz, hour: 'numeric', minute: '2-digit' });
}

export function formatAgenda(a, tz) {
  const events = a?.events || [];
  const emails = a?.emails || [];
  if (!events.length && !emails.length) return 'Nothing on the calendar and no emails mention today.';
  const lines = [];
  if (events.length) {
    lines.push('Calendar:');
    for (const e of events) {
      const loc = e.location ? ` @ ${e.location}` : '';
      lines.push(`- ${fmtTime(e.start, tz)} ${e.title || e.summary || 'event'}${loc}`.replace('-  ', '- '));
    }
  }
  if (emails.length) {
    lines.push('From your email:');
    for (const m of emails.slice(0, 6)) {
      const who = m.from_name || m.from || m.from_addr || '';
      lines.push(`- ${m.subject || '(no subject)'}${who ? ` (${who})` : ''}`);
    }
  }
  return lines.join('\n');
}

function explain(err) {
  if (err instanceof AtlasError) {
    if (err.status === 404 || err.status === 405) {
      return 'The Inbox Atlas agent is not wired up on the server yet (404). Pull main with the search-agent branch merged and restart server.py.';
    }
    if (err.status === 0) return `I cannot reach Inbox Atlas right now: ${err.message}`;
    return `Inbox Atlas hit an error: ${err.message}`;
  }
  return `Something broke: ${err?.message || err}`;
}

export class Bot {
  constructor({ atlas, history, owners = [], allowAny = false, tz = 'America/New_York', log = console }) {
    this.atlas = atlas;
    this.history = history;
    this.owners = new Set(owners.map(normalizeHandle).filter(Boolean));
    this.allowAny = allowAny;
    this.tz = tz;
    this.log = log;
  }

  allowed(sender) {
    if (this.allowAny) return true;
    if (!this.owners.size) return false;
    return this.owners.has(normalizeHandle(sender));
  }

  // Returns the reply text, or null when the message should be ignored.
  async handle({ threadId, sender, text }) {
    const body = String(text || '').trim();
    if (!body) return null;
    if (!this.allowed(sender)) {
      this.log.warn?.(`[imessage] ignoring message from ${sender} (not OWNER_PHONE; set ALLOW_ANY=1 to allow)`);
      return null;
    }
    const reply = plain(await this.route(threadId, body));
    return reply || 'I have nothing to say to that.';
  }

  async route(threadId, body) {
    const lower = body.toLowerCase().replace(/[.!?]+$/, '').trim();
    try {
      if (lower === 'help' || lower === '?' || lower === 'commands') return HELP;
      if (lower === 'reset') {
        this.history.clear(threadId);
        return 'Okay, fresh start.';
      }
      if (lower === 'today') return await this.ask(threadId, TODAY_Q, true);
      if (lower === 'watches') return await this.listWatches();
      const watch = body.match(/^watch\s+(.+)$/i);
      if (watch) {
        const topic = watch[1].trim();
        const r = await this.atlas.addWatch(topic);
        return `Watching "${r?.name || topic}". I will text you when new mail about it lands. Text "stop ${r?.name || topic}" to end it.`;
      }
      const stop = body.match(/^(stop|unwatch)\s+(.+)$/i);
      if (stop) {
        const r = await this.atlas.stopWatch(stop[2].trim());
        return r?.removed ? `Stopped watching "${stop[2].trim()}".` : `No watch called "${stop[2].trim()}". Text "watches" to see them.`;
      }
      const brief = lower.match(/^brief\s+(on|off)$/);
      if (brief) {
        const r = await this.atlas.brief(brief[1] === 'on');
        return r?.enabled
          ? `Morning brief is on. You will get it at ${r.time || '08:00'} every day.`
          : 'Morning brief is off.';
      }
      return await this.ask(threadId, body, false);
    } catch (err) {
      this.log.error?.('[imessage] error:', err.message);
      return explain(err);
    }
  }

  async listWatches() {
    const ws = await this.atlas.watches();
    const list = Array.isArray(ws) ? ws : ws?.watches || [];
    if (!list.length) return 'No watches yet. Text "watch <topic>" to add one.';
    return 'Your watches:\n' + list.map((w) => `- ${w.name}`).join('\n');
  }

  async ask(threadId, text, isToday) {
    const history = this.history.get(threadId);
    let reply;
    try {
      const r = await this.atlas.ask(withAgendaHint(text, this.tz), history);
      reply = typeof r === 'string' ? r : r?.reply;
    } catch (err) {
      // Agent missing or down: "today" can still be answered from /api/agenda.
      if (!(isToday || DAY_WORDS.test(text))) throw err;
      try {
        const day = localDate(this.tz, /\btomorrow\b/i.test(text) ? 1 : 0);
        reply = formatAgenda(await this.atlas.agenda(day), this.tz);
      } catch {
        throw err;
      }
    }
    reply = plain(reply || '');
    this.history.add(threadId, text, reply);
    return reply;
  }
}
