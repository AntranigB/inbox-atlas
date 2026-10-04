/* Inbox Atlas voice: hold-to-dictate (Whisperflow style) and "Talk to inbox" realtime agent.
 * Self-contained. Mounts into #voice-slot if present, otherwise floats bottom right.
 * Events on window:
 *   atlas:query  {text, raw}            after dictation fills the search box
 *   atlas:hits   {hits, region, args}   when the voice agent's search returns
 *   atlas:voice  {role, text, final}    captions from the voice agent
 */
(() => {
  if (window.AtlasVoice) return;
  const RATE = 24000;
  const HOLD_MS = 280;

  const css = `
  .av-wrap{display:inline-flex;align-items:center;gap:8px;font:13px/1.2 system-ui,-apple-system,sans-serif;position:relative}
  .av-float{position:fixed;right:18px;bottom:18px;z-index:9999;background:rgba(20,22,28,.92);padding:8px 10px;border-radius:999px;box-shadow:0 6px 24px rgba(0,0,0,.25)}
  .av-btn{position:relative;border:0;cursor:pointer;border-radius:999px;height:36px;display:inline-flex;align-items:center;gap:6px;padding:0 12px;
    background:#23262f;color:#e8eaf0;font:inherit;transition:background .15s,transform .1s;user-select:none;-webkit-user-select:none;touch-action:none}
  .av-btn:hover{background:#2e323d}
  .av-btn:active{transform:scale(.97)}
  .av-btn svg{width:16px;height:16px;fill:currentColor}
  .av-mic{width:36px;padding:0;justify-content:center}
  .av-ring{position:absolute;inset:-3px;border-radius:999px;border:2px solid #ff5a5f;opacity:0;transform:scale(1);transition:opacity .1s;pointer-events:none}
  .av-rec .av-ring{opacity:1}
  .av-rec{background:#3a1f24!important;color:#ff8a8e}
  .av-busy{background:#2b2a1c!important;color:#f3d36b}
  .av-talk.av-on{background:#173a2c;color:#6be3a8}
  .av-talk.av-speaking{background:#1b3346;color:#7cc4ff}
  .av-dot{width:8px;height:8px;border-radius:50%;background:currentColor;opacity:.6}
  .av-on .av-dot{animation:av-pulse 1.2s infinite}
  @keyframes av-pulse{50%{opacity:.15}}
  .av-meter{width:46px;height:6px;border-radius:3px;background:#2a2d36;overflow:hidden;display:none}
  .av-meter i{display:block;height:100%;width:0;background:linear-gradient(90deg,#6be3a8,#f3d36b,#ff5a5f);transition:width .05s}
  .av-active .av-meter{display:block}
  .av-status{color:#9aa0ad;font-size:12px;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:220px}
  .av-cap{position:absolute;top:calc(100% + 8px);right:0;width:min(380px,90vw);max-height:220px;overflow:auto;display:none;
    background:rgba(20,22,28,.96);color:#e8eaf0;border-radius:12px;padding:10px 12px;box-shadow:0 8px 28px rgba(0,0,0,.3);z-index:9999;font-size:13px;line-height:1.4}
  .av-float .av-cap{top:auto;bottom:calc(100% + 10px)}
  .av-cap.av-show{display:block}
  .av-cap p{margin:0 0 6px}
  .av-cap .u{color:#9aa0ad}
  .av-cap .a{color:#e8eaf0}
  .av-toast{position:fixed;left:50%;bottom:22px;transform:translateX(-50%);background:#3a1f24;color:#ffb3b5;padding:8px 14px;border-radius:8px;font:13px system-ui;z-index:10000}
  `;
  const style = document.createElement("style");
  style.textContent = css;
  document.head.appendChild(style);

  const MIC = '<svg viewBox="0 0 24 24"><path d="M12 14a3 3 0 0 0 3-3V5a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-3.08A7 7 0 0 0 19 11h-2z"/></svg>';

  const wrap = document.createElement("div");
  wrap.className = "av-wrap";
  wrap.innerHTML = `
    <button class="av-btn av-mic" type="button" title="Hold to dictate (or hold Space)">${MIC}<span class="av-ring"></span></button>
    <span class="av-meter"><i></i></span>
    <button class="av-btn av-talk" type="button" title="Talk to your inbox"><span class="av-dot"></span><span class="av-tl">Talk to inbox</span></button>
    <span class="av-status"></span>
    <div class="av-cap" aria-live="polite"></div>`;
  const $ = (s) => wrap.querySelector(s);
  const micBtn = $(".av-mic"), talkBtn = $(".av-talk"), meterBar = $(".av-meter i"), statusEl = $(".av-status"), cap = $(".av-cap");

  function mount() {
    const slot = document.getElementById("voice-slot");
    if (slot) slot.appendChild(wrap);
    else { wrap.classList.add("av-float"); document.body.appendChild(wrap); }
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount); else mount();

  const setStatus = (t) => { statusEl.textContent = t || ""; };
  function toast(msg) {
    const d = document.createElement("div"); d.className = "av-toast"; d.textContent = msg;
    document.body.appendChild(d); setTimeout(() => d.remove(), 3500);
  }
  const isEditable = (el) => el && (el.tagName === "TEXTAREA" || (el.tagName === "INPUT" && /^(text|search|email|url|tel|)$/i.test(el.type || "")) || el.isContentEditable);

  let lastEditable = null;
  document.addEventListener("focusin", (e) => { if (isEditable(e.target)) lastEditable = e.target; });

  // ---------- level meter ----------
  let meterRAF = 0;
  function startMeter(ctx, source) {
    const an = ctx.createAnalyser(); an.fftSize = 512; source.connect(an);
    const buf = new Uint8Array(an.fftSize);
    const tick = () => {
      an.getByteTimeDomainData(buf);
      let s = 0; for (const v of buf) { const x = (v - 128) / 128; s += x * x; }
      const rms = Math.sqrt(s / buf.length);
      meterBar.style.width = Math.min(100, rms * 400) + "%";
      meterRAF = requestAnimationFrame(tick);
    };
    tick();
    return () => { cancelAnimationFrame(meterRAF); meterBar.style.width = "0"; try { source.disconnect(an); } catch {} };
  }

  // ---------- dictation ----------
  const dict = { state: "idle", rec: null, chunks: [], stream: null, ctx: null, stopMeter: null, target: null, t0: 0, toggle: false };

  async function startDictation(target) {
    if (dict.state !== "idle") return;
    if (talk.on) { toast("Turn off Talk to inbox to dictate"); return; }
    dict.state = "starting"; dict.target = target || null; dict.t0 = performance.now(); dict.toggle = false;
    try {
      dict.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
    } catch (e) { dict.state = "idle"; toast("Microphone blocked: " + e.message); return; }
    if (dict.state !== "starting") { dict.stream.getTracks().forEach((t) => t.stop()); dict.state = "idle"; return; }
    const mime = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"].find((m) => window.MediaRecorder && MediaRecorder.isTypeSupported(m)) || "";
    dict.rec = new MediaRecorder(dict.stream, mime ? { mimeType: mime } : undefined);
    dict.chunks = [];
    dict.rec.ondataavailable = (e) => e.data.size && dict.chunks.push(e.data);
    dict.rec.start(100);
    dict.ctx = new AudioContext();
    dict.stopMeter = startMeter(dict.ctx, dict.ctx.createMediaStreamSource(dict.stream));
    dict.state = "recording";
    micBtn.classList.add("av-rec"); wrap.classList.add("av-active");
    setStatus("Listening...");
  }

  async function stopDictation() {
    if (dict.state === "starting") { dict.state = "cancel"; return; }
    if (dict.state !== "recording") return;
    dict.state = "sending";
    const rec = dict.rec;
    const done = new Promise((r) => (rec.onstop = r));
    rec.stop();
    await done;
    dict.stream.getTracks().forEach((t) => t.stop());
    dict.stopMeter && dict.stopMeter(); dict.ctx && dict.ctx.close();
    micBtn.classList.remove("av-rec"); wrap.classList.remove("av-active");
    const dur = performance.now() - dict.t0;
    const type = rec.mimeType || "audio/webm";
    const blob = new Blob(dict.chunks, { type });
    if (dur < 350 || blob.size < 1500) { dict.state = "idle"; setStatus(""); return; }
    micBtn.classList.add("av-busy"); setStatus("Transcribing...");
    const ext = type.includes("ogg") ? "ogg" : type.includes("mp4") ? "m4a" : "webm";
    const fd = new FormData();
    fd.append("audio", blob, "dictation." + ext);
    const tgt = dict.target;
    fd.append("style", tgt && tgt.id !== "q" && tgt.tagName === "TEXTAREA" ? "plain" : "query");
    try {
      const r = await fetch("/api/dictate", { method: "POST", body: fd });
      if (!r.ok) throw new Error((await r.text()).slice(0, 160));
      const j = await r.json();
      insertText(tgt, j.text, j.raw);
      setStatus(j.text ? `${j.ms} ms` : "Heard nothing");
      setTimeout(() => setStatus(""), 2500);
    } catch (e) {
      setStatus(""); toast("Dictation failed: " + e.message);
    } finally {
      micBtn.classList.remove("av-busy"); dict.state = "idle";
    }
  }

  function insertText(target, text, raw) {
    if (!text) return;
    const q = document.getElementById("q");
    const el = target || q;
    if (!el) { window.dispatchEvent(new CustomEvent("atlas:query", { detail: { text, raw } })); return; }
    if (el.isContentEditable) {
      el.focus(); document.execCommand("insertText", false, text);
    } else if (el === q || !el.value) {
      el.value = text;
    } else {
      const s = el.selectionStart ?? el.value.length, e = el.selectionEnd ?? s;
      const pre = el.value.slice(0, s), sep = pre && !/\s$/.test(pre) ? " " : "";
      el.value = pre + sep + text + el.value.slice(e);
      const pos = (pre + sep + text).length; el.setSelectionRange && el.setSelectionRange(pos, pos);
    }
    el.dispatchEvent(new Event("input", { bubbles: true }));
    if (el === q || !target) {
      window.dispatchEvent(new CustomEvent("atlas:query", { detail: { text, raw } }));
      el.dispatchEvent(new CustomEvent("atlas:query", { bubbles: true, detail: { text, raw } }));
    }
    el.focus && el.focus();
  }

  // Mic button: hold to talk, or tap once to start and tap again to stop.
  let downAt = 0;
  micBtn.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    if (dict.state === "recording" && dict.toggle) { stopDictation(); return; }
    downAt = performance.now();
    const tgt = isEditable(document.activeElement) ? document.activeElement : lastEditable;
    startDictation(tgt && document.contains(tgt) ? tgt : null);
  });
  const micUp = () => {
    if (!downAt) return;
    const held = performance.now() - downAt; downAt = 0;
    if (held < HOLD_MS) { dict.toggle = true; setStatus("Listening... tap to stop"); }
    else stopDictation();
  };
  micBtn.addEventListener("pointerup", micUp);
  micBtn.addEventListener("pointerleave", () => { if (downAt && performance.now() - downAt >= HOLD_MS) micUp(); });

  // Space: hold anywhere except while typing in a field. Fills #q.
  let spaceDown = false;
  window.addEventListener("keydown", (e) => {
    if (e.code !== "Space" || e.repeat || e.metaKey || e.ctrlKey || e.altKey) { if (spaceDown && e.code === "Space") e.preventDefault(); return; }
    const a = document.activeElement;
    if (isEditable(a) || (a && /^(BUTTON|SELECT|A)$/.test(a.tagName) && a !== micBtn)) return;
    e.preventDefault(); spaceDown = true;
    startDictation(document.getElementById("q"));
  });
  window.addEventListener("keyup", (e) => {
    if (e.code !== "Space" || !spaceDown) return;
    e.preventDefault(); spaceDown = false; stopDictation();
  });
  window.addEventListener("blur", () => { if (spaceDown) { spaceDown = false; stopDictation(); } });

  // ---------- realtime talk ----------
  const WORKLET = `
  class Cap extends AudioWorkletProcessor {
    constructor(){ super(); this.buf = new Int16Array(1200); this.n = 0; }
    process(inputs){
      const ch = inputs[0] && inputs[0][0];
      if (ch) for (let i = 0; i < ch.length; i++) {
        const s = Math.max(-1, Math.min(1, ch[i]));
        this.buf[this.n++] = s < 0 ? s * 0x8000 : s * 0x7fff;
        if (this.n === this.buf.length) { this.port.postMessage(this.buf.buffer.slice(0)); this.n = 0; }
      }
      return true;
    }
  }
  registerProcessor('atlas-cap', Cap);`;

  const talk = { on: false, ws: null, ctx: null, stream: null, node: null, src: null, stopMeter: null,
                 playAt: 0, sources: new Set(), userLine: null, botLine: null, mutedUntil: 0 };

  function caption(role, text, final) {
    cap.classList.add("av-show");
    const key = role === "user" ? "userLine" : "botLine";
    if (!talk[key]) {
      talk[key] = document.createElement("p"); talk[key].className = role === "user" ? "u" : "a";
      cap.appendChild(talk[key]);
      while (cap.children.length > 8) cap.firstChild.remove();
    }
    talk[key].textContent = (role === "user" ? "You: " : "") + text;
    cap.scrollTop = cap.scrollHeight;
    if (final) talk[key] = null;
    window.dispatchEvent(new CustomEvent("atlas:voice", { detail: { role, text, final: !!final } }));
  }

  function playPCM(b64) {
    const bin = atob(b64), n = bin.length >> 1;
    if (!n) return;
    const f = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let v = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
      if (v >= 0x8000) v -= 0x10000;
      f[i] = v / 0x8000;
    }
    const ctx = talk.ctx;
    const buf = ctx.createBuffer(1, n, RATE); buf.copyToChannel(f, 0);
    const s = ctx.createBufferSource(); s.buffer = buf; s.connect(ctx.destination);
    const now = ctx.currentTime;
    if (talk.playAt < now + 0.03) talk.playAt = now + 0.06;
    s.start(talk.playAt); talk.playAt += buf.duration;
    talk.mutedUntil = talk.playAt;
    talk.sources.add(s); s.onended = () => { talk.sources.delete(s); if (!talk.sources.size) talkBtn.classList.remove("av-speaking"); };
    talkBtn.classList.add("av-speaking");
  }

  function stopPlayback() {
    for (const s of talk.sources) { try { s.stop(); } catch {} }
    talk.sources.clear(); talk.playAt = 0; talk.mutedUntil = 0;
    talkBtn.classList.remove("av-speaking");
  }

  function onEvent(ev) {
    switch (ev.type) {
      case "atlas.ready": setStatus("Listening. Ask about your inbox"); break;
      case "atlas.error": case "error":
        toast("Voice: " + (ev.message || (ev.error && ev.error.message) || "error")); break;
      case "input_audio_buffer.speech_started": stopPlayback(); setStatus("Hearing you..."); break;
      case "input_audio_buffer.speech_stopped": setStatus("Thinking..."); break;
      case "conversation.item.input_audio_transcription.updated":
      case "conversation.item.input_audio_transcription.delta":
        if (ev.transcript || ev.delta) caption("user", ev.transcript || ev.delta, false); break;
      case "conversation.item.input_audio_transcription.completed":
        if (ev.transcript) caption("user", ev.transcript, true); break;
      case "response.output_audio.delta": case "response.audio.delta":
        if (ev.delta) playPCM(ev.delta); setStatus("Speaking"); break;
      case "response.output_audio_transcript.delta": case "response.audio_transcript.delta": case "response.text.delta":
        talk.botText = (talk.botText || "") + (ev.delta || ""); caption("bot", talk.botText, false); break;
      case "response.output_audio_transcript.done": case "response.audio_transcript.done":
        caption("bot", ev.transcript || talk.botText || "", true); talk.botText = ""; break;
      case "response.function_call_arguments.done": setStatus("Searching your inbox..."); break;
      case "atlas.hits":
        window.dispatchEvent(new CustomEvent("atlas:hits", { detail: { hits: ev.hits || [], region: ev.region || null, args: ev.args || {}, tool: ev.tool } }));
        setStatus(`${(ev.hits || []).length} matches`); break;
      case "response.done": if (!talk.sources.size) setStatus("Listening"); break;
    }
  }

  async function startTalk() {
    if (talk.on) return;
    if (dict.state !== "idle") return;
    talk.on = true; talkBtn.classList.add("av-on"); talkBtn.querySelector(".av-tl").textContent = "Stop";
    setStatus("Connecting...");
    try {
      talk.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 } });
      talk.ctx = new AudioContext({ sampleRate: RATE });
      await talk.ctx.resume();
      const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
      await talk.ctx.audioWorklet.addModule(url);
      talk.src = talk.ctx.createMediaStreamSource(talk.stream);
      talk.node = new AudioWorkletNode(talk.ctx, "atlas-cap");
      talk.src.connect(talk.node);
      talk.stopMeter = startMeter(talk.ctx, talk.src);
      wrap.classList.add("av-active");
      const proto = location.protocol === "https:" ? "wss:" : "ws:";
      const ws = new WebSocket(`${proto}//${location.host}/ws/voice`);
      ws.binaryType = "arraybuffer";
      talk.ws = ws;
      ws.onmessage = (m) => { if (typeof m.data === "string") { try { onEvent(JSON.parse(m.data)); } catch (e) { console.warn(e); } } };
      ws.onclose = () => { if (talk.on) { setStatus("Disconnected"); stopTalk(); } };
      ws.onerror = () => toast("Voice connection failed");
      // Half duplex while the agent speaks so it does not hear itself on laptop speakers.
      talk.node.port.onmessage = (e) => {
        if (ws.readyState !== 1) return;
        if (talk.ctx.currentTime < talk.mutedUntil + 0.25) return;
        ws.send(e.data);
      };
    } catch (e) {
      toast("Talk failed: " + e.message); stopTalk();
    }
  }

  function stopTalk() {
    talk.on = false; talkBtn.classList.remove("av-on", "av-speaking"); talkBtn.querySelector(".av-tl").textContent = "Talk to inbox";
    stopPlayback();
    try { talk.ws && talk.ws.close(); } catch {}
    talk.stopMeter && talk.stopMeter();
    try { talk.node && talk.node.disconnect(); talk.src && talk.src.disconnect(); } catch {}
    talk.stream && talk.stream.getTracks().forEach((t) => t.stop());
    talk.ctx && talk.ctx.close();
    Object.assign(talk, { ws: null, ctx: null, stream: null, node: null, src: null, stopMeter: null, userLine: null, botLine: null });
    wrap.classList.remove("av-active");
    setStatus("");
    setTimeout(() => { if (!talk.on) cap.classList.remove("av-show"); }, 6000);
  }

  talkBtn.addEventListener("click", () => {
    if (!talk.on) return startTalk();
    if (talk.sources.size) { // first click while it talks interrupts, second click stops
      stopPlayback();
      talk.ws && talk.ws.readyState === 1 && talk.ws.send(JSON.stringify({ type: "response.cancel" }));
      setStatus("Listening"); return;
    }
    stopTalk();
  });

  window.AtlasVoice = {
    startDictation, stopDictation, startTalk, stopTalk,
    ask(text) { if (talk.ws && talk.ws.readyState === 1) talk.ws.send(JSON.stringify({ type: "atlas.text", text })); },
  };
})();
