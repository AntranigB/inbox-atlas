// Text helpers: phone normalizing, markdown stripping, splitting long replies.

export const MAX_CHUNK = 600;

export function normalizeHandle(raw) {
  const s = String(raw || '').trim();
  if (!s) return '';
  if (s.includes('@')) return s.toLowerCase();
  if (s.startsWith('+')) return '+' + s.slice(1).replace(/\D/g, '');
  const digits = s.replace(/\D/g, '');
  if (digits.length === 10) return `+1${digits}`;
  if (digits.length === 11 && digits.startsWith('1')) return `+${digits}`;
  return digits ? `+${digits}` : s;
}

// iMessage is plain text: drop markdown and the dash characters we never send.
export function plain(text) {
  return String(text ?? '')
    .replace(/\r\n/g, '\n')
    .replace(/```[a-z]*\n?/gi, '')
    .replace(/\*\*(.+?)\*\*/g, '$1')
    .replace(/__(.+?)__/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/^#{1,6}\s+/gm, '')
    .replace(/^\s*[*]\s+/gm, '- ')
    .replace(/\[([^\]]+)\]\((https?:[^)]+)\)/g, '$1 $2')
    .replace(/\s*[\u2014\u2013]\s*/g, ', ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function splitBy(text, max, [pattern, sep]) {
  const parts = text.split(pattern);
  const out = [];
  let cur = '';
  for (const p of parts) {
    const next = cur ? cur + sep + p : p;
    if (next.length <= max) cur = next;
    else {
      if (cur) out.push(cur);
      cur = p;
    }
  }
  if (cur) out.push(cur);
  return out;
}

// Split at paragraph, then line, then sentence, then word boundaries.
export function splitMessage(text, max = MAX_CHUNK) {
  const t = String(text ?? '').trim();
  if (!t) return [];
  if (t.length <= max) return [t];
  const seps = [
    [/\n\n/, '\n\n'],
    [/\n/, '\n'],
    [/(?<=[.!?])\s+/, ' '],
    [/ /, ' '],
  ];
  const rec = (s, level) => {
    if (s.length <= max) return [s];
    if (level >= seps.length) {
      const out = [];
      for (let i = 0; i < s.length; i += max) out.push(s.slice(i, i + max));
      return out;
    }
    return splitBy(s, max, seps[level]).flatMap((c) => rec(c, level + 1));
  };
  return rec(t, 0).map((c) => c.trim()).filter(Boolean);
}

export function localDate(tz, offsetDays = 0, now = new Date()) {
  const d = new Date(now.getTime() + offsetDays * 86400000);
  // en-CA formats as YYYY-MM-DD
  return new Intl.DateTimeFormat('en-CA', { timeZone: tz, year: 'numeric', month: '2-digit', day: '2-digit' }).format(d);
}
