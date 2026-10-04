/* Inbox Atlas web UI. Vanilla JS, no build step. */
(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };

  const state = {
    mode: "region", encoder: null, query: "",
    positive: null, negative: null, // null means let Grok expand
    res: null, map: null, hits: new Map(), members: new Set(), facetPts: [],
    view: { s: 1, tx: 0, ty: 0 }, hover: null, history: [], session: null,
  };

  async function api(path, opts) {
    const r = await fetch(path, opts && opts.body ? { method: opts.method || "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(opts.body) } : opts);
    if (!r.ok) throw new Error(`${path} ${r.status}`);
    return r.json();
  }

  const fmtDate = (ts) => {
    if (!ts) return "";
    if (typeof ts === "string") return ts;
    const d = new Date(ts * 1000);
    const now = new Date();
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: d.getFullYear() === now.getFullYear() ? undefined : "numeric" });
  };

  /* ---------- search ---------- */
  async function search(q, keepFacets) {
    if (typeof q === "string") { if (q !== state.query || !keepFacets) { state.positive = null; state.negative = null; } state.query = q; $("#q").value = q; }
    if (!state.query.trim()) return;
    $("#result-meta").textContent = state.mode === "keyword" ? "searching" : (state.positive ? "searching" : "Grok is expanding your query");
    const body = { query: state.query, mode: state.mode, k: 20 };
    if (state.encoder) body.encoder = state.encoder;
    if (state.positive) { body.positive = state.positive; body.negative = state.negative || []; }
    try {
      const res = await api("/api/search", { body });
      applyResult(res);
    } catch (e) {
      $("#result-meta").textContent = "search failed: " + e.message;
    }
  }

  function applyResult(res) {
    state.res = res;
    if (res.facets && res.region) {
      state.positive = res.facets.positive || [];
      state.negative = res.facets.negative || [];
    }
    state.hits = new Map((res.hits || []).map((h) => [h.id, h]));
    state.members = new Set((res.region && res.region.member_ids) || (res.hits || []).filter((h) => h.member).map((h) => h.id));
    state.facetPts = res.facet_points || [];
    renderVerdict(res);
    renderChips(res);
    renderResults(res);
    drawMap();
    document.dispatchEvent(new CustomEvent("atlas:result", { detail: res }));
  }

  function renderVerdict(res) {
    const v = $("#verdict");
    const r = res.region;
    if (!r) { v.hidden = true; return; }
    v.hidden = false;
    v.className = "verdict " + (r.related ? "yes" : "no");
    v.innerHTML = "";
    v.append(el("b", null, r.related ? "RELATED: YES" : "RELATED: NO"));
    const conf = Math.round((r.confidence || 0) * 100);
    v.append(el("span", null, r.related
      ? `${r.count} in region, strongest z=${r.max_z}, ${conf}% sure`
      : `nothing beyond noise, max z=${r.max_z}, ${conf}% sure`));
  }

  function renderChips(res) {
    const box = $("#chips");
    box.innerHTML = "";
    if (!res.region) return;
    const hits = res.region.facet_hits || {};
    const mk = (label, neg) => {
      const c = el("button", "chip" + (neg ? " neg" : "") + (!neg && hits[label] === 0 ? " zero" : ""));
      c.type = "button";
      c.title = neg ? "excluded facet, click to remove" : "facet, click to remove and re-run";
      c.append(el("span", null, label));
      if (!neg && label in hits) c.append(el("span", "n", hits[label]));
      c.append(el("span", "x", "x"));
      c.onclick = () => {
        if (neg) state.negative = state.negative.filter((x) => x !== label);
        else state.positive = state.positive.filter((x) => x !== label);
        search(null, true);
      };
      return c;
    };
    (state.positive || []).forEach((p) => box.append(mk(p, false)));
    (state.negative || []).forEach((n) => box.append(mk(n, true)));
    const add = el("button", "chip add", "+ facet");
    add.type = "button";
    add.onclick = () => {
      const inp = el("input", "chip");
      inp.placeholder = "new facet, enter";
      inp.style.width = "150px";
      add.replaceWith(inp);
      inp.focus();
      inp.onkeydown = (e) => {
        if (e.key === "Enter" && inp.value.trim()) { state.positive = [...(state.positive || []), inp.value.trim()]; search(null, true); }
        if (e.key === "Escape") renderChips(state.res);
      };
    };
    box.append(add);
  }

  function renderResults(res) {
    const ol = $("#results");
    ol.innerHTML = "";
    const hits = res.hits || [];
    const label = { keyword: "BM25 keyword", embed: "plain embedding cosine", region: "region, hub corrected z", hybrid: "region + BM25 (RRF)" }[res.mode] || res.mode;
    $("#result-meta").textContent = `${hits.length} shown, ${label}${res.encoder ? ", " + res.encoder : ""}`;
    if (!hits.length) { ol.append(el("li", "empty", "No matches.")); return; }
    for (const h of hits) {
      const li = el("li", "hit" + (res.region && !h.member ? " out" : ""));
      li.dataset.id = h.id;
      li.append(el("div", "from", h.from || h.from_addr || "unknown"), el("div", "date", fmtDate(h.date)),
        el("div", "subj", h.subject || "(no subject)"), el("div", "snip", h.snippet || ""));
      const sc = el("div", "score");
      if (h.z != null) {
        const bar = el("span", "zbar" + (h.member ? "" : " out"));
        const fill = el("i");
        fill.style.width = Math.max(3, Math.min(100, (h.z / 8) * 100)) + "%";
        bar.append(fill);
        sc.append(bar, el("span", null, `z ${h.z.toFixed(1)}`));
        if (h.prob != null) sc.append(el("span", null, `p ${h.prob.toFixed(2)}`));
        if (h.facet) sc.append(el("span", "facet-tag", h.facet));
      } else if (h.bm25 != null) sc.append(el("span", null, `bm25 ${h.bm25.toFixed(2)}`));
      else if (h.cos != null) sc.append(el("span", null, `cos ${h.cos.toFixed(3)}`));
      li.append(sc);
      li.onclick = () => toggleBody(li, h.id);
      li.onmouseenter = () => { state.hover = h.id; drawMap(); };
      li.onmouseleave = () => { state.hover = null; drawMap(); };
      ol.append(li);
    }
  }

  async function toggleBody(li, id) {
    const open = li.querySelector(".body");
    if (open) { open.remove(); return; }
    const b = el("div", "body", "loading");
    li.append(b);
    try { const e = await api(`/api/email/${encodeURIComponent(id)}`); b.textContent = e.body || "(empty)"; }
    catch (e) { b.textContent = "could not load email"; }
  }

  /* ---------- map ---------- */
  const cv = $("#map");
  const ctx = cv.getContext("2d");
  let pts = [], bounds = null, colors = {}, clusterCenters = [];

  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const hue = (i) => (i * 137.508) % 360;
  function clusterColor(c, a) {
    if (c == null || c < 0) return `rgba(128,128,128,${a})`;
    const dark = matchMedia("(prefers-color-scheme: dark)").matches && document.documentElement.dataset.theme !== "light" || document.documentElement.dataset.theme === "dark";
    return `hsla(${hue(c)}, ${dark ? 55 : 48}%, ${dark ? 62 : 46}%, ${a})`;
  }

  async function loadMap() {
    try {
      const m = await api("/api/map" + (state.encoder ? `?encoder=${encodeURIComponent(state.encoder)}` : ""));
      state.map = m;
    } catch (e) { state.map = null; }
    const m = state.map;
    pts = (m && m.points) || [];
    $("#map-empty").hidden = pts.length > 0;
    if (!pts.length) { $("#map-meta").textContent = ""; $("#legend").innerHTML = ""; drawMap(); return; }
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const p of pts) { x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x); y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y); }
    bounds = { x0, x1, y0, y1 };
    const sums = {};
    for (const p of pts) { const s = sums[p.cluster] || (sums[p.cluster] = { x: 0, y: 0, n: 0 }); s.x += p.x; s.y += p.y; s.n++; }
    clusterCenters = (m.clusters || []).filter((c) => sums[c.id]).map((c) => ({ ...c, x: sums[c.id].x / sums[c.id].n, y: sums[c.id].y / sums[c.id].n }));
    $("#map-meta").textContent = `${pts.length} emails, ${clusterCenters.length} clusters${m.fallback ? ", PCA preview (index not built)" : ""}`;
    const lg = $("#legend");
    lg.innerHTML = "";
    clusterCenters.slice().sort((a, b) => b.size - a.size).slice(0, 14).forEach((c) => {
      const s = el("span");
      const dot = el("i"); dot.style.background = clusterColor(c.id, 1);
      s.append(dot, document.createTextNode(`${c.label} (${c.size})`));
      s.onclick = () => focusOn(c.x, c.y, 2.2);
      lg.append(s);
    });
    resetView();
  }

  function size() {
    const r = cv.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    if (cv.width !== Math.round(r.width * dpr) || cv.height !== Math.round(r.height * dpr)) {
      cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr);
    }
    return { w: r.width, h: r.height, dpr };
  }

  function baseXY(x, y, w, h) {
    const pad = 28;
    const sx = (w - 2 * pad) / Math.max(bounds.x1 - bounds.x0, 1e-9);
    const sy = (h - 2 * pad) / Math.max(bounds.y1 - bounds.y0, 1e-9);
    const s = Math.min(sx, sy);
    const ox = (w - s * (bounds.x1 - bounds.x0)) / 2, oy = (h - s * (bounds.y1 - bounds.y0)) / 2;
    return [ox + (x - bounds.x0) * s, oy + (bounds.y1 - y) * s];
  }
  function toScreen(x, y, w, h) {
    const [bx, by] = baseXY(x, y, w, h);
    const v = state.view;
    return [bx * v.s + v.tx, by * v.s + v.ty];
  }

  function star(cx, cy, r) {
    ctx.beginPath();
    for (let i = 0; i < 10; i++) {
      const a = (Math.PI / 5) * i - Math.PI / 2, rr = i % 2 ? r * 0.45 : r;
      ctx.lineTo(cx + Math.cos(a) * rr, cy + Math.sin(a) * rr);
    }
    ctx.closePath();
  }

  let raf = 0;
  function drawMap() { cancelAnimationFrame(raf); raf = requestAnimationFrame(draw); }
  function draw() {
    const { w, h, dpr } = size();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!pts.length || !bounds) return;
    const glow = css("--glow"), ink = css("--ink"), paper = css("--paper");
    const active = state.members.size > 0 || state.hits.size > 0;
    const dim = active ? parseFloat(css("--dot-dim")) : 0.85;
    const r = Math.max(2, Math.min(4.5, 3.2 * Math.sqrt(state.view.s)));
    for (const p of pts) {
      const [x, y] = toScreen(p.x, p.y, w, h);
      if (x < -10 || y < -10 || x > w + 10 || y > h + 10) continue;
      ctx.fillStyle = clusterColor(p.cluster, dim);
      ctx.beginPath(); ctx.arc(x, y, r, 0, 6.283); ctx.fill();
    }
    // region glow
    if (active) {
      ctx.save();
      ctx.globalCompositeOperation = document.documentElement.dataset.theme === "dark" || matchMedia("(prefers-color-scheme: dark)").matches ? "lighter" : "multiply";
      for (const p of pts) {
        const inR = state.members.has(p.id), hit = state.hits.has(p.id);
        if (!inR && !hit) continue;
        const [x, y] = toScreen(p.x, p.y, w, h);
        const g = ctx.createRadialGradient(x, y, 0, x, y, inR ? 26 : 14);
        g.addColorStop(0, `rgba(${glow},${inR ? 0.35 : 0.15})`);
        g.addColorStop(1, `rgba(${glow},0)`);
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(x, y, inR ? 26 : 14, 0, 6.283); ctx.fill();
      }
      ctx.restore();
      for (const p of pts) {
        const inR = state.members.has(p.id), hit = state.hits.has(p.id);
        if (!inR && !hit) continue;
        const [x, y] = toScreen(p.x, p.y, w, h);
        ctx.fillStyle = inR ? `rgb(${glow})` : clusterColor(p.cluster, 0.9);
        ctx.strokeStyle = paper; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(x, y, inR ? r + 2 : r + 0.5, 0, 6.283); ctx.fill(); ctx.stroke();
        if (state.hover === p.id) { ctx.strokeStyle = ink; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, r + 7, 0, 6.283); ctx.stroke(); }
      }
    }
    // cluster labels
    ctx.font = `500 11px ${css("--mono") || "monospace"}`;
    ctx.textAlign = "center";
    for (const c of clusterCenters) {
      if (c.size < 3 && clusterCenters.length > 8) continue;
      let [x, y] = toScreen(c.x, c.y, w, h);
      const half = ctx.measureText(c.label).width / 2 + 4;
      if (half * 2 < w) x = Math.max(half, Math.min(w - half, x)); // keep labels inside on narrow screens
      ctx.lineWidth = 3; ctx.strokeStyle = paper; ctx.fillStyle = active ? css("--ink-3") : ink;
      ctx.strokeText(c.label, x, y); ctx.fillText(c.label, x, y);
    }
    // facet stars
    ctx.font = `600 12px ${css("--sans") || "sans-serif"}`;
    ctx.textAlign = "left";
    const placed = [];
    for (const f of state.facetPts) {
      const [x, y] = toScreen(f.x, f.y, w, h);
      star(x, y, 7);
      ctx.fillStyle = paper; ctx.strokeStyle = `rgb(${glow})`; ctx.lineWidth = 2; ctx.fill(); ctx.stroke();
      // skip labels that would collide with one already drawn
      let ly = y + 4;
      const tw = ctx.measureText(f.label).width;
      for (let tries = 0; tries < 4 && placed.some((b) => x + 10 < b.x1 && x + 10 + tw > b.x0 && Math.abs(ly - b.y) < 13); tries++) ly += 13;
      if (placed.some((b) => x + 10 < b.x1 && x + 10 + tw > b.x0 && Math.abs(ly - b.y) < 13)) continue;
      placed.push({ x0: x + 10, x1: x + 10 + tw, y: ly });
      ctx.lineWidth = 3; ctx.strokeStyle = paper; ctx.fillStyle = `rgb(${glow})`;
      ctx.strokeText(f.label, x + 10, ly); ctx.fillText(f.label, x + 10, ly);
    }
  }

  function resetView() { state.view = { s: 1, tx: 0, ty: 0 }; drawMap(); }
  function focusOn(x, y, s) {
    const { w, h } = size();
    const [bx, by] = baseXY(x, y, w, h);
    state.view = { s, tx: w / 2 - bx * s, ty: h / 2 - by * s };
    drawMap();
  }
  function zoomAt(mx, my, f) {
    const v = state.view;
    const ns = Math.max(0.5, Math.min(40, v.s * f));
    v.tx = mx - (mx - v.tx) * (ns / v.s); v.ty = my - (my - v.ty) * (ns / v.s); v.s = ns;
    drawMap();
  }
  function nearest(mx, my) {
    const { w, h } = size();
    let best = null, bd = 100;
    for (const p of pts) {
      const [x, y] = toScreen(p.x, p.y, w, h);
      const d = (x - mx) ** 2 + (y - my) ** 2;
      if (d < bd) { bd = d; best = p; }
    }
    return best;
  }

  const pointers = new Map();
  let lastPinch = 0, moved = false;
  cv.addEventListener("wheel", (e) => { e.preventDefault(); const b = cv.getBoundingClientRect(); zoomAt(e.clientX - b.left, e.clientY - b.top, Math.exp(-e.deltaY * 0.0015)); }, { passive: false });
  cv.addEventListener("pointerdown", (e) => { cv.setPointerCapture(e.pointerId); pointers.set(e.pointerId, [e.clientX, e.clientY]); moved = false; cv.classList.add("drag"); });
  cv.addEventListener("pointerup", async (e) => {
    pointers.delete(e.pointerId); lastPinch = 0; cv.classList.remove("drag");
    if (moved) return;
    const b = cv.getBoundingClientRect();
    const p = nearest(e.clientX - b.left, e.clientY - b.top);
    if (!p) return;
    const li = document.querySelector(`.hit[data-id="${CSS.escape(p.id)}"]`);
    if (li) { li.scrollIntoView({ behavior: "smooth", block: "center" }); li.classList.add("hover"); setTimeout(() => li.classList.remove("hover"), 1200); return; }
    try { const em = await api(`/api/email/${encodeURIComponent(p.id)}`); showTip(e, `${em.from_name || em.from_addr}: ${em.subject}`); } catch (_) {}
  });
  cv.addEventListener("pointercancel", (e) => { pointers.delete(e.pointerId); cv.classList.remove("drag"); });
  cv.addEventListener("pointermove", (e) => {
    const b = cv.getBoundingClientRect();
    if (pointers.has(e.pointerId)) {
      const [px, py] = pointers.get(e.pointerId);
      pointers.set(e.pointerId, [e.clientX, e.clientY]);
      if (pointers.size === 2) {
        const [a, c] = [...pointers.values()];
        const d = Math.hypot(a[0] - c[0], a[1] - c[1]);
        if (lastPinch) zoomAt((a[0] + c[0]) / 2 - b.left, (a[1] + c[1]) / 2 - b.top, d / lastPinch);
        lastPinch = d; moved = true; return;
      }
      const dx = e.clientX - px, dy = e.clientY - py;
      if (Math.abs(dx) + Math.abs(dy) > 1) moved = true;
      state.view.tx += dx; state.view.ty += dy; drawMap();
      $("#map-tip").hidden = true;
      return;
    }
    const p = nearest(e.clientX - b.left, e.clientY - b.top);
    if (!p) { $("#map-tip").hidden = true; return; }
    const h = state.hits.get(p.id);
    const cl = clusterCenters.find((c) => c.id === p.cluster);
    showTip(e, h ? `${h.from}: ${h.subject}` : (cl ? cl.label : "email"));
  });
  cv.addEventListener("pointerleave", () => { $("#map-tip").hidden = true; });
  function showTip(e, text) {
    const t = $("#map-tip"), b = cv.getBoundingClientRect();
    t.textContent = text; t.hidden = false;
    t.style.left = Math.min(e.clientX - b.left, b.width - 200) + "px"; t.style.top = (e.clientY - b.top) + "px";
  }
  $("#map-reset").onclick = resetView;
  window.addEventListener("resize", drawMap);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawMap);

  /* ---------- chat ---------- */
  function addMsg(role, text, extra) {
    const m = el("div", "msg " + role);
    if (role === "bot") {
      // tiny markdown: **bold** only, everything else stays plain text
      String(text).split(/(\*\*[^*]+\*\*)/).forEach((part) => {
        if (/^\*\*[^*]+\*\*$/.test(part)) m.append(el("b", null, part.slice(2, -2))); else m.append(document.createTextNode(part));
      });
    } else m.textContent = text;
    if (extra) m.append(el("span", "trace", extra));
    $("#chat-log").append(m);
    $("#chat-log").scrollTop = 1e9;
    return m;
  }

  /* ---------- chat sessions (server keeps history, POST /api/chat) ---------- */
  const SKEY = "atlas_session";
  const remember = (id) => { try { id ? localStorage.setItem(SKEY, id) : localStorage.removeItem(SKEY); } catch (_) {} };
  const sessionLabel = (s) => {
    const who = /^(imessage|telegram|whatsapp_business|slack):/.test(s.id) ? s.id.split(":")[0].replace("_business", "") + " " + s.id.split(":").slice(1).join(":") : "";
    const title = (s.title || "").replace(/\n\[context:[\s\S]*$/, "").slice(0, 40) || "chat";
    return who ? `${who}: ${title}` : title;
  };
  let chatApi = true;
  async function loadSessions() {
    const pick = $("#session-pick");
    let list = [];
    try { list = await api("/api/chat/sessions?limit=40"); } catch (_) { chatApi = false; pick.parentElement.hidden = true; return; }
    pick.innerHTML = "";
    pick.append(new Option("New chat", ""));
    for (const s of list) pick.append(new Option(sessionLabel(s), s.id));
    if (state.session && !list.some((s) => s.id === state.session)) pick.append(new Option("this chat", state.session));
    pick.value = state.session || "";
  }
  async function openSession(id) {
    state.session = id || null;
    state.history = [];
    remember(state.session);
    $("#chat-log").innerHTML = "";
    if (!id) { addMsg("bot", "New chat. Ask anything about your inbox."); return; }
    try {
      const r = await api(`/api/chat/sessions/${encodeURIComponent(id)}`);
      for (const m of r.messages || []) {
        addMsg(m.role === "user" ? "user" : "bot", m.role === "user" ? String(m.content).replace(/\n\[context:[\s\S]*$/, "") : m.content);
        state.history.push({ role: m.role, content: m.content });
      }
      if (!(r.messages || []).length) addMsg("bot", "Empty chat.");
    } catch (e) { addMsg("bot", "Could not load that chat: " + e.message); }
  }
  $("#session-pick").addEventListener("change", (e) => openSession(e.target.value));
  $("#session-new").addEventListener("click", () => { openSession(null); loadSessions(); $("#chat-in").focus(); });

  async function ask(text) {
    if (!text.trim()) return;
    addMsg("user", text);
    const wait = addMsg("bot thinking", "navigating your inbox");
    try {
      const res = chatApi
        ? await api("/api/chat", { body: { text, channel: "web", session_id: state.session } })
        : await api("/api/ask", { body: { text, channel: "web", history: state.history.slice(-10) } });
      if (res.session_id && res.session_id !== state.session) { state.session = res.session_id; remember(res.session_id); loadSessions(); }
      wait.remove();
      const tr = (res.trace || []).map((t) => t.tool).join(" > ");
      addMsg("bot", res.reply || "(no answer)", tr ? "tools: " + tr : "");
      state.history.push({ role: "user", content: text }, { role: "assistant", content: res.reply || "" });
      if (res.hits && res.hits.length) {
        const view = { mode: "region", query: text, hits: res.hits, region: res.region, facets: res.facets, encoder: state.res && state.res.encoder };
        applyResult(view);
      }
      return res;
    } catch (e) {
      wait.remove();
      addMsg("bot", "Something went wrong: " + e.message);
    }
  }

  /* ---------- wiring ---------- */
  $("#search-form").addEventListener("submit", (e) => { e.preventDefault(); search($("#q").value); });
  $("#chat-form").addEventListener("submit", (e) => { e.preventDefault(); const v = $("#chat-in").value; $("#chat-in").value = ""; ask(v); });
  document.querySelectorAll("#mode button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("#mode button").forEach((x) => x.classList.toggle("on", x === b));
    state.mode = b.dataset.mode;
    if (state.query) search(null, true);
  }));

  async function init() {
    try {
      const h = await api("/api/health");
      $("#status").textContent = "n_emails" in h ? `${h.n_emails} emails, ${h.encoder}` : "locked";
    } catch (e) { $("#status").textContent = "server offline"; }
    try {
      const enc = await api("/api/encoders");
      const avail = enc.available || [];
      if (avail.length >= 2) {
        const seg = $("#encoder");
        seg.hidden = false;
        avail.forEach((name) => {
          const b = el("button", name === enc.default || (state.encoder == null && name === avail[0] && !avail.includes(enc.default)) ? "on" : "", /^base/.test(name) ? "Base" : (/hash/.test(name) ? "Hash" : "Tuned"));
          b.title = name;
          b.onclick = () => {
            seg.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
            state.encoder = name; loadMap().then(() => state.query && search(null, true));
          };
          seg.append(b);
        });
      }
    } catch (_) {}
    try { state.session = new URLSearchParams(location.search).get("session") || localStorage.getItem(SKEY) || null; } catch (_) {}
    await loadSessions();
    if (state.session) await openSession(state.session);
    await loadMap();
    const q = new URLSearchParams(location.search).get("q");
    if (q) search(q);
  }

  window.atlas = { search: (q) => search(q), ask, applyResult, setQuery: (q) => { $("#q").value = q; }, state };
  init();
})();
