/* Optional ATLAS_TOKEN support. Loaded before app.js and voice.js: adds the token to every
   same-origin /api fetch and /ws websocket, asks for it once and keeps it in localStorage.
   A link like https://host/?token=abc stores the token and strips it from the address bar. */
(function () {
  "use strict";
  const KEY = "atlas_token";
  const store = {
    get() { try { return localStorage.getItem(KEY) || ""; } catch (_) { return ""; } },
    set(v) { try { v ? localStorage.setItem(KEY, v) : localStorage.removeItem(KEY); } catch (_) {} },
  };
  const url = new URL(location.href);
  if (url.searchParams.get("token")) {
    store.set(url.searchParams.get("token"));
    url.searchParams.delete("token");
    history.replaceState(null, "", url.pathname + (url.search || "") + url.hash);
  }

  const rawFetch = window.fetch.bind(window);
  const isApi = (u) => {
    try { const x = new URL(u, location.href); return x.origin === location.origin && x.pathname.startsWith("/api/"); }
    catch (_) { return false; }
  };
  function ask(again) {
    const v = window.prompt(again ? "That token did not work. Inbox Atlas token:" : "Inbox Atlas token (ATLAS_TOKEN on the Mac):", "");
    if (v) store.set(v.trim());
    return !!v;
  }
  function withToken(init) {
    const t = store.get();
    if (!t) return init;
    const headers = new Headers((init && init.headers) || {});
    headers.set("X-Atlas-Token", t);
    return Object.assign({}, init, { headers });
  }

  // First call decides whether the server wants a token at all.
  let ready = null;
  function check() {
    if (!ready) {
      ready = rawFetch("/api/health", withToken({})).then((r) => r.json()).then((h) => {
        if (h.auth_required && !("n_emails" in h)) {
          let tries = 0;
          const loop = () => tries++ < 3 && ask(tries > 1) ? rawFetch("/api/health", withToken({})).then((r) => r.json())
            .then((h2) => ("n_emails" in h2 ? null : loop())) : null;
          return loop();
        }
      }).catch(() => {});
    }
    return ready;
  }

  window.fetch = async function (input, init) {
    const u = typeof input === "string" ? input : input.url;
    if (!isApi(u)) return rawFetch(input, init);
    await check();
    let r = await rawFetch(input, withToken(init));
    if (r.status === 401 && ask(true)) r = await rawFetch(input, withToken(init));
    return r;
  };

  const RawWS = window.WebSocket;
  function WS(u, protocols) {
    let target = u;
    try {
      const x = new URL(u, location.href);
      if (x.host === location.host && x.pathname.startsWith("/ws/") && store.get()) {
        x.searchParams.set("token", store.get());
        target = x.toString();
      }
    } catch (_) {}
    return protocols === undefined ? new RawWS(target) : new RawWS(target, protocols);
  }
  WS.prototype = RawWS.prototype;
  ["CONNECTING", "OPEN", "CLOSING", "CLOSED"].forEach((k) => { WS[k] = RawWS[k]; });
  window.WebSocket = WS;
  window.atlasAuth = { token: store.get, setToken: store.set, forget: () => store.set("") };
})();
