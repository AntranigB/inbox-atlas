// Per-thread short history, in memory and mirrored to a JSON file.

import fs from 'node:fs';
import path from 'node:path';

export const MAX_TURNS = 8;

export class HistoryStore {
  constructor(file = null, maxTurns = MAX_TURNS) {
    this.file = file;
    this.maxTurns = maxTurns;
    this.threads = {};
    if (file) {
      try {
        this.threads = JSON.parse(fs.readFileSync(file, 'utf8')) || {};
      } catch {
        this.threads = {};
      }
    }
  }

  get(threadId) {
    return [...(this.threads[threadId] || [])];
  }

  add(threadId, userText, assistantText) {
    const h = this.threads[threadId] || [];
    h.push({ role: 'user', content: userText }, { role: 'assistant', content: assistantText });
    this.threads[threadId] = h.slice(-this.maxTurns * 2);
    this.save();
  }

  clear(threadId) {
    delete this.threads[threadId];
    this.save();
  }

  save() {
    if (!this.file) return;
    try {
      fs.mkdirSync(path.dirname(this.file), { recursive: true });
      const tmp = this.file + '.tmp';
      fs.writeFileSync(tmp, JSON.stringify(this.threads, null, 1));
      fs.renameSync(tmp, this.file);
    } catch (err) {
      console.error('[imessage] could not save history:', err.message);
    }
  }
}
