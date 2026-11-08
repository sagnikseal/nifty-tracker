/* NIFTY engine dashboard. Vanilla JS, no build step, no network dependencies beyond the engine API.
 * Contract: docs/API.md. Every number shown comes from the API; missing values render as "—" with a
 * reason in the title attribute. Nothing is ever simulated in the browser. */
(function () {
  "use strict";

  // =========================================================================== utilities
  const $ = (id) => document.getElementById(id);
  const isNum = (v) => typeof v === "number" && isFinite(v);
  const MINUS = "−";
  const NF = {};
  const nf = (dp) => NF[dp] || (NF[dp] = new Intl.NumberFormat("en-IN", { minimumFractionDigits: dp, maximumFractionDigits: dp }));
  const num = (v, dp = 2) => (isNum(v) ? nf(dp).format(v).replace("-", MINUS) : "—");
  const signed = (v, dp = 2) => (isNum(v) ? (v > 0 ? "+" : v < 0 ? MINUS : "") + nf(dp).format(Math.abs(v)) : "—");
  const signedPct = (v, dp = 2) => (isNum(v) ? signed(v, dp) + "%" : "—");      // v already in percent
  const prob = (v, dp = 1) => (isNum(v) ? (v * 100).toFixed(dp) + "%" : "—");     // v in 0..1
  const inr = (v, dp = 0, sign = false) => {
    if (!isNum(v)) return "—";
    const s = nf(dp).format(Math.abs(v));
    if (v < 0) return MINUS + "₹" + s;
    return (sign && v > 0 ? "+" : "") + "₹" + s;
  };
  const pval = (v) => (!isNum(v) ? "—" : v < 0.001 ? "<0.001" : v.toFixed(3));
  const TIME_FMT = new Intl.DateTimeFormat("en-GB", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23" });
  const HM_FMT = new Intl.DateTimeFormat("en-GB", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
  const DATE_FMT = new Intl.DateTimeFormat("en-GB", { timeZone: "Asia/Kolkata", day: "2-digit", month: "short" });
  const YMD_FMT = new Intl.DateTimeFormat("en-CA", { timeZone: "Asia/Kolkata", year: "numeric", month: "2-digit", day: "2-digit" });
  const toDate = (x) => {
    if (x === null || x === undefined || x === "") return null;
    const d = typeof x === "number" ? new Date(x) : new Date(String(x));
    return isNaN(d.getTime()) ? null : d;
  };
  const timeIST = (x) => { const d = toDate(x); return d ? TIME_FMT.format(d) : "—"; };
  const hmIST = (x) => { const d = toDate(x); return d ? HM_FMT.format(d) : "—"; };
  const dateIST = (x) => { const d = toDate(x); return d ? DATE_FMT.format(d) : "—"; };
  const ymdIST = (x) => { const d = toDate(x); return d ? YMD_FMT.format(d) : null; };
  const todayIST = () => YMD_FMT.format(new Date());
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const safeUrl = (u) => (typeof u === "string" && /^https?:\/\//i.test(u) ? u : null);
  const parseMaybeJSON = (v) => { if (typeof v !== "string") return v; try { return JSON.parse(v); } catch (e) { return null; } };
  const ageText = (x) => {
    const d = toDate(x);
    if (!d) return "—";
    const m = Math.max(0, Math.round((Date.now() - d.getTime()) / 60000));
    if (m < 1) return "just now";
    if (m < 60) return m + " min ago";
    const h = Math.floor(m / 60);
    return h < 24 ? h + " h " + (m % 60) + " min ago" : Math.floor(h / 24) + " d ago";
  };

  /** Write text only when it changed (no layout churn on 2-second updates). */
  function setText(el, text, title) {
    if (typeof el === "string") el = $(el);
    if (!el) return;
    const t = text === null || text === undefined ? "—" : String(text);
    if (el.textContent !== t) el.textContent = t;
    if (title !== undefined) { if (title) el.title = title; else el.removeAttribute("title"); }
  }
  function setCls(el, base, extra) {
    if (typeof el === "string") el = $(el);
    if (!el) return;
    const c = extra ? base + " " + extra : base;
    if (el.className !== c) el.className = c;
  }
  function show(el, on) { if (typeof el === "string") el = $(el); if (el) el.hidden = !on; }
  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      const v = attrs[k];
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "text") el.textContent = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids) if (kid !== null && kid !== undefined) el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    return el;
  }
  /** Rich text: set plain text then underline glossary terms. Only re-renders when the text changed. */
  function setRich(el, text) {
    if (typeof el === "string") el = $(el);
    if (!el) return;
    const t = text === null || text === undefined ? "" : String(text);
    if (el.dataset.raw === t && el.dataset.gl === String(GLOSS_VERSION)) return;
    el.textContent = t;
    el.dataset.raw = t;
    el.dataset.gl = String(GLOSS_VERSION);
    glossify(el);
  }
  function setList(el, items) {
    if (typeof el === "string") el = $(el);
    if (!el) return;
    const arr = Array.isArray(items) ? items.map(String) : [];
    const key = arr.join("\u0001") + "|" + GLOSS_VERSION;
    if (el.dataset.raw === key) return;
    el.dataset.raw = key;
    el.textContent = "";
    if (!arr.length) { el.append(h("li", { class: "muted" }, "—")); return; }
    for (const s of arr) { const li = h("li", null, s); el.append(li); glossify(li); }
  }
  function toast(text, kind) {
    const t = h("div", { class: "toast " + (kind || ""), role: "status" }, text);
    $("toasts").append(t);
    setTimeout(() => t.remove(), 6000);
  }
  function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

  // =========================================================================== theme & tabs
  function applyTheme(t) {
    document.documentElement.dataset.theme = t;
    try { localStorage.setItem("nifty-theme", t); } catch (e) { /* ignore */ }
    Object.values(CHARTS).forEach((c) => c && c.requestDraw());
  }
  function initTheme() {
    let t = "dark";
    try { t = localStorage.getItem("nifty-theme") || "dark"; } catch (e) { /* ignore */ }
    const q = new URLSearchParams(location.search).get("theme");
    if (q === "light" || q === "dark") t = q;
    document.documentElement.dataset.theme = t;
    $("theme-toggle").addEventListener("click", () => applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
  }
  const TABS = ["live", "books", "reports", "system", "glossary"];
  function selectTab(name, focus) {
    for (const t of TABS) {
      const on = t === name;
      $("tab-" + t).setAttribute("aria-selected", on ? "true" : "false");
      $("tab-" + t).tabIndex = on ? 0 : -1;
      $("panel-" + t).hidden = !on;
    }
    if (focus) $("tab-" + name).focus();
    try { localStorage.setItem("nifty-tab", name); } catch (e) { /* ignore */ }
    if (name === "books") { CHARTS.eq && CHARTS.eq.resize(); fetchEquity(); fetchTrades(); }
    if (name === "live") { CHARTS.price && CHARTS.price.resize(); CHARTS.pred && CHARTS.pred.resize(); }
  }
  function initTabs() {
    TABS.forEach((t, i) => {
      const b = $("tab-" + t);
      b.addEventListener("click", () => selectTab(t));
      b.addEventListener("keydown", (e) => {
        if (e.key === "ArrowRight") selectTab(TABS[(i + 1) % TABS.length], true);
        else if (e.key === "ArrowLeft") selectTab(TABS[(i + TABS.length - 1) % TABS.length], true);
      });
    });
    let t = "live";
    try { t = localStorage.getItem("nifty-tab") || "live"; } catch (e) { /* ignore */ }
    const fromHash = (location.hash || "").replace("#", "");
    if (TABS.includes(fromHash)) t = fromHash;          // deep link, e.g. http://localhost:8010/#reports
    selectTab(TABS.includes(t) ? t : "live");
    window.addEventListener("hashchange", () => { const x = location.hash.replace("#", ""); if (TABS.includes(x)) selectTab(x); });
  }

  // =========================================================================== glossary & tooltips
  let GLOSS = {};
  let GLOSS_VERSION = 0;
  let RE_LONG = null, RE_SHORT = null;
  // Words too generic to underline automatically in free text (they still have tooltips on UI labels).
  const AUTO_SKIP = new Set(["long", "short", "target", "equity", "edge", "lot", "option", "books", "agent", "session",
    "candle", "gap", "confidence", "relevance", "magnitude", "surprise", "channel", "coverage", "accuracy", "charges",
    "regime", "momentum", "liquidity", "hedge", "pnl", "nifty", "nifty 50", "india-relevant", "price source",
    "against count", "extension rule", "guardrail", "llm", "ohlc", "bid", "ask", "skew", "pivot", "brokerage"]);
  const escRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

  async function loadGlossary() {
    for (const url of ["/api/glossary", "/static/glossary.json"]) {
      try {
        const r = await fetch(url, { cache: "no-store" });
        if (!r.ok) continue;
        const js = await r.json();
        if (js && typeof js === "object" && Object.keys(js).length) {
          GLOSS = {};
          for (const k in js) GLOSS[k.toLowerCase()] = js[k];
          buildGlossRegex();
          GLOSS_VERSION++;
          document.querySelectorAll("[data-term]").forEach((el) => { if (!el.hasAttribute("tabindex")) el.tabIndex = 0; });
          renderGlossaryTab();
          document.querySelectorAll(".gl-auto").forEach((el) => glossify(el));
          if (LAST) scheduleRender();
          return;
        }
      } catch (e) { /* try next */ }
    }
  }
  function buildGlossRegex() {
    const longT = [], shortT = [];
    for (const k of Object.keys(GLOSS)) {
      if (AUTO_SKIP.has(k)) continue;
      if (k.length <= 3) shortT.push(k.toUpperCase());       // RBI, CPI, ATM, PCR, IV ... match upper-case only
      else longT.push(k);
    }
    longT.sort((a, b) => b.length - a.length);
    shortT.sort((a, b) => b.length - a.length);
    RE_LONG = longT.length ? new RegExp("(?<![A-Za-z0-9])(" + longT.map(escRe).join("|") + ")(?![A-Za-z0-9])", "gi") : null;
    RE_SHORT = shortT.length ? new RegExp("(?<![A-Za-z0-9])(" + shortT.map(escRe).join("|") + ")(?![A-Za-z0-9])", "g") : null;
  }
  /** Wrap the first occurrence of each glossary term inside `root`'s text with a tooltip span. */
  function glossify(root) {
    if (!root || (!RE_LONG && !RE_SHORT)) return;
    const seen = new Set();
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(n) {
        const p = n.parentElement;
        if (!p || p.closest(".gl, [data-term], code, pre, script, style, button")) return NodeFilter.FILTER_REJECT;
        return n.nodeValue && n.nodeValue.trim().length > 1 ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
      },
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    for (const node of nodes) {
      const text = node.nodeValue;
      const hits = [];
      for (const re of [RE_LONG, RE_SHORT]) {
        if (!re) continue;
        re.lastIndex = 0;
        let m;
        while ((m = re.exec(text))) {
          const key = m[1].toLowerCase();
          if (seen.has(key) || !GLOSS[key]) continue;
          if (hits.some((x) => m.index < x.end && m.index + m[1].length > x.start)) continue;
          hits.push({ start: m.index, end: m.index + m[1].length, key });
          seen.add(key);
        }
      }
      if (!hits.length) continue;
      hits.sort((a, b) => a.start - b.start);
      const frag = document.createDocumentFragment();
      let pos = 0;
      for (const x of hits) {
        if (x.start > pos) frag.append(text.slice(pos, x.start));
        frag.append(h("span", { class: "gl", "data-gl": x.key, tabindex: "0" }, text.slice(x.start, x.end)));
        pos = x.end;
      }
      if (pos < text.length) frag.append(text.slice(pos));
      node.replaceWith(frag);
    }
  }
  function lookupTerm(key) {
    if (!key) return null;
    let k = key.toLowerCase().trim();
    if (GLOSS[k]) return { k, v: GLOSS[k] };
    if (k.endsWith("s") && GLOSS[k.slice(0, -1)]) return { k: k.slice(0, -1), v: GLOSS[k.slice(0, -1)] };
    return null;
  }
  let tipFor = null;
  function showTip(el) {
    const key = el.dataset.term || el.dataset.gl;
    const hit = lookupTerm(key);
    if (!hit) return;
    const tip = $("tip");
    tip.textContent = "";
    tip.append(h("div", { class: "tt-term" }, hit.k), h("div", { class: "tt-short" }, hit.v.short || ""),
      hit.v.long ? h("div", { class: "tt-long" }, hit.v.long) : null);
    tip.hidden = false;
    const r = el.getBoundingClientRect();
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    let left = clamp(r.left, 8, window.innerWidth - tw - 8);
    let top = r.bottom + 6;
    if (top + th > window.innerHeight - 8) top = Math.max(8, r.top - th - 6);
    tip.style.left = left + "px";
    tip.style.top = top + "px";
    el.setAttribute("aria-describedby", "tip");
    tipFor = el;
  }
  function hideTip() {
    $("tip").hidden = true;
    if (tipFor) tipFor.removeAttribute("aria-describedby");
    tipFor = null;
  }
  function initTooltips() {
    const sel = "[data-term], .gl";
    document.addEventListener("pointerover", (e) => { const el = e.target.closest && e.target.closest(sel); if (el && el !== tipFor) showTip(el); });
    document.addEventListener("pointerout", (e) => {
      if (!tipFor) return;
      const to = e.relatedTarget;
      if (!to || !tipFor.contains(to)) { const el = e.target.closest && e.target.closest(sel); if (el === tipFor) hideTip(); }
    });
    document.addEventListener("focusin", (e) => { const el = e.target.closest && e.target.closest(sel); if (el) showTip(el); });
    document.addEventListener("focusout", (e) => { if (tipFor && e.target === tipFor) hideTip(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") hideTip(); });
    window.addEventListener("scroll", hideTip, { passive: true });
  }
  function renderGlossaryTab() {
    const dl = $("gl-list");
    dl.textContent = "";
    const keys = Object.keys(GLOSS).sort();
    for (const k of keys) {
      const v = GLOSS[k];
      const wrap = h("div", { "data-k": k + " " + (v.short || "").toLowerCase() },
        h("dt", null, k), h("dd", { class: "s" }, v.short || ""), h("dd", null, v.long || ""));
      dl.append(wrap);
    }
    const inp = $("gl-search");
    inp.oninput = () => {
      const q = inp.value.trim().toLowerCase();
      for (const d of dl.children) d.hidden = q && !d.dataset.k.includes(q);
    };
  }

  // =========================================================================== connection
  let LAST = null, lastStateAt = 0, es = null, pollTimer = null, failCount = 0, sseErrorSince = 0, connMode = "connecting";
  function setConn(mode) {
    connMode = mode;
    const el = $("conn");
    const map = { live: ["live", "LIVE"], recon: ["recon", "RECONNECTING"], off: ["off", "ENGINE OFFLINE"], connecting: ["", "CONNECTING"] };
    const [cls, label] = map[mode] || map.connecting;
    setCls(el, "conn", cls);
    setText(el, label, mode === "off" ? "Start the engine: double-click start.command / start.bat or run `python -m nifty_engine app`" : "");
    show("offline-banner", mode === "off");
    document.querySelectorAll(".chart-wrap").forEach((w) => w.classList.toggle("stale", mode !== "live"));
    ["btn-buy-ce", "btn-buy-pe"].forEach((id) => { $(id).disabled = mode === "off"; });
  }
  function onState(s) {
    if (!s || typeof s !== "object") return;
    LAST = s;
    lastStateAt = Date.now();
    failCount = 0;
    if (connMode !== "live") setConn("live");
    scheduleRender();
  }
  function startSSE() {
    // ?poll=1 forces polling (proxies that buffer event streams; headless screenshots)
    if (!window.EventSource || new URLSearchParams(location.search).get("poll") === "1") { startPolling(); return; }
    try { es = new EventSource("/api/stream"); } catch (e) { startPolling(); return; }
    es.addEventListener("state", (ev) => {
      sseErrorSince = 0;
      try { onState(JSON.parse(ev.data)); } catch (e) { /* malformed frame: ignore */ }
      if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    });
    es.onerror = () => {
      if (!sseErrorSince) sseErrorSince = Date.now();
      // CLOSED = the server refused the stream (e.g. 204/404): the browser will not retry, so poll now.
      if (es.readyState === 2 || Date.now() - sseErrorSince > 8000) { es.close(); es = null; startPolling(); return; }
      if (connMode === "live") setConn("recon");
    };
  }
  async function pollOnce() {
    try {
      const r = await fetch("/api/state", { cache: "no-store" });
      if (!r.ok) throw new Error("HTTP " + r.status);
      onState(await r.json());
      return true;
    } catch (e) {
      failCount++;
      setConn(failCount >= 2 ? "off" : "recon");
      return false;
    }
  }
  function startPolling() {
    if (pollTimer) return;
    pollTimer = setInterval(async () => {
      const ok = await pollOnce();
      if (ok && !es && Math.random() < 0.2) startSSE();   // retry the stream now and then
    }, 5000);
    pollOnce();
  }
  setInterval(() => {   // watchdog: a silent stream is not "live"
    if (connMode === "live" && lastStateAt && Date.now() - lastStateAt > 15000) { setConn("recon"); if (!pollTimer) startPolling(); }
  }, 3000);

  // =========================================================================== render orchestration
  let renderPending = false, lastSig = "";
  function scheduleRender() {
    if (renderPending) return;
    renderPending = true;
    requestAnimationFrame(() => {
      renderPending = false;
      const s = LAST;
      if (!s) return;
      const parts = [renderHeader, renderPrediction, renderQuality, renderPositions, renderRules, renderBooks, renderAgent,
        renderChain, renderNews, renderSources, renderAlerts, renderHealth, renderCharts];
      for (const f of parts) { try { f(s); } catch (e) { console.error(f.name, e); } }
      // Refresh trades/equity when the set of positions or trade counts changes.
      const sig = JSON.stringify([(s.positions || []).map((p) => p && p.book + ":" + p.id), (s.books || []).map((b) => b && b.trades)]);
      if (sig !== lastSig) { lastSig = sig; fetchTrades(); fetchEquity(); }
    });
  }
  const positionsOf = (s) => (Array.isArray(s.positions) ? s.positions.filter(Boolean) : s.position ? [s.position] : []);
  const sessionDate = (s) => (s.market && s.market.session_date) || ymdIST(s.ts) || todayIST();

  // =========================================================================== header
  function renderHeader(s) {
    const q = s.quote || {};
    const noQuote = q.source === "none" || !isNum(q.spot) ? "no live quote (source: " + (q.source || "none") + ")" : "";
    setText("q-spot", num(q.spot, 2), noQuote);
    const ch = $("q-change");
    if (isNum(q.change)) {
      setText(ch, (q.change >= 0 ? "▲ " : "▼ ") + signed(q.change, 2) + " (" + signedPct(q.change_pct) + ")");
      setCls(ch, "delta", q.change > 0 ? "up" : q.change < 0 ? "down" : "");
    } else { setText(ch, "—", noQuote); setCls(ch, "delta"); }
    setText("q-open", num(q.open, 2)); setText("q-high", num(q.high, 2)); setText("q-low", num(q.low, 2)); setText("q-prev", num(q.prev_close, 2));
    setText("q-vix", num(q.vix, 2), isNum(q.vix) ? "" : "no VIX quote");
    const vc = $("q-vixchg");
    setText(vc, isNum(q.vix_change_pct) ? signedPct(q.vix_change_pct, 1) : "");
    setCls(vc, "delta small", isNum(q.vix_change_pct) ? (q.vix_change_pct > 0 ? "down" : "up") : "");   // rising VIX = risk-off
    const m = s.market || {};
    const st = { OPEN: ["good", "● Open"], PRE_OPEN: ["warn", "◐ Pre-open"], CLOSED: ["", "○ Closed"], HOLIDAY: ["", "○ Holiday"] }[m.status] || ["", m.status || "—"];
    setText("mkt-status", st[1]); setCls("mkt-status", "pill", st[0]);
    setText("mkt-note", m.note || "");
    const age = isNum(q.age_s) ? (q.age_s < 90 ? Math.round(q.age_s) + "s old" : Math.round(q.age_s / 60) + " min old") : "";
    setText("q-source", "source " + (q.source || "—") + (age ? " · " + age : ""),
      q.source === "yahoo" ? "Yahoo quotes can be delayed; add an Upstox token for live data" : q.source === "none" ? "no data source connected" : "");
    const hl = s.health || {};
    dot("h-engine", hl.engine === "running" ? "ok" : hl.engine ? "bad" : null, "Engine: " + (hl.engine || "unknown"));
    dot("h-llm", bool(hl.llm), hl.llm ? "Local LLM reachable" : "Local LLM not reachable (start Ollama)");
    dot("h-upstox", bool(hl.upstox), hl.upstox ? "Upstox connected" : "No Upstox token: option prices are modelled");
    dot("h-news", bool(hl.news_thread), hl.news_thread ? "News thread running" : "News thread not running");
  }
  const bool = (v) => (v === true ? "ok" : v === false ? "bad" : null);
  function dot(id, state, title) { setCls(id, "hdot", state || ""); $(id).title = title || ""; }

  // =========================================================================== prediction
  function dirWord(p, thr) {
    if (!isNum(p)) return ["", "flat"];
    if (p >= thr) return ["Bullish", "up"];
    if (p <= 1 - thr) return ["Bearish", "down"];
    if (Math.abs(p - 0.5) < 0.01) return ["Neutral", "flat"];
    return p > 0.5 ? ["Lean up", "up"] : ["Lean down", "down"];
  }
  function setGauge(prefix, p, thr, ext, reason, tickThr) {
    setText(prefix + "-val", prob(p), isNum(p) ? "" : reason);
    const [w, cls] = dirWord(p, thr);
    setText(prefix + "-word", w); setCls(prefix + "-word", "g-word", cls);
    const f = $(prefix + "-fill");
    if (isNum(p)) {
      f.style.left = Math.min(0.5, p) * 100 + "%";
      f.style.width = Math.abs(p - 0.5) * 100 + "%";
      setCls(f, "m-fill", p > 0.5 ? "up" : p < 0.5 ? "down" : "");
    } else { f.style.width = "0"; }
    const ticks = [[prefix + "-t1", tickThr], [prefix + "-t2", isNum(tickThr) ? 1 - tickThr : null], [prefix + "-t3", ext], [prefix + "-t4", isNum(ext) ? 1 - ext : null]];
    for (const [id, v] of ticks) {
      const t = $(id);
      if (!t) continue;
      if (isNum(v)) { t.style.display = "block"; t.style.left = "calc(" + v * 100 + "% - 1px)"; t.title = (id.endsWith("3") || id.endsWith("4") ? "hold-extension threshold " : "entry threshold ") + prob(v, 0); }
      else t.style.display = "none";
    }
  }
  function renderPrediction(s) {
    const p = s.prediction || {};
    const r = s.rules || {};
    const thr = isNum(r.entry_threshold) ? r.entry_threshold : 0.6;     // words only; ticks need real rules
    const ext = isNum(r.extend_prob) ? r.extend_prob : 0.8;
    const tThr = isNum(r.entry_threshold) ? r.entry_threshold : null, tExt = isNum(r.extend_prob) ? r.extend_prob : null;
    const why = p.model_version ? "model has not scored this bar yet" : "no intraday model trained yet (run training)";
    setGauge("g30", p.p_up_30m, thr, tExt, why, tThr);
    setGauge("gc", p.p_up_close, thr, null, why, tThr);
    setGauge("gd", p.p_daily, isNum(r.daily_threshold) ? r.daily_threshold : 0.56, null, "daily model not available", null);
    const eb = $("edge-badge");
    if (p.has_edge === true) { setText(eb, "✔ Edge proven (p < 0.05)"); setCls(eb, "badge", "good"); }
    else if (p.has_edge === false) { setText(eb, "⚠ Edge not proven"); setCls(eb, "badge", "bad"); }
    else { setText(eb, "Edge: —"); setCls(eb, "badge"); }
    // Soft majority vote: which models had a vote (trailing skill beat the base rate) and how many agree.
    const v30 = (p.voters || {})["30m"], ag30 = (p.agreement || {})["30m"];
    const voteTxt = !p.model_version ? "" : (!v30 || !v30.length) ? " · vote: no model beat the base rate (no trade)"
      : ` · vote: ${v30.join(" + ")}${ag30 == null ? "" : `, ${Math.round(ag30 * v30.length)}/${v30.length} agree`}`;
    setText("p-version", (p.model_version || "—") + voteTxt);
    setText("p-updated", p.updated ? timeIST(p.updated) + " IST" : "—");
    // Hold-extension rule: make it obvious whether it can fire.
    const pos = positionsOf(s)[0];
    const share = ((s.model_quality || {}).horizon_30m || {}).share_above_80;
    const shareTxt = isNum(share) ? " In testing the model reached ≥ " + prob(ext, 0) + " on " + prob(share, 2) + " of 30-min predictions." : "";
    const p30 = p.p_up_30m;
    const tgt = inr(isNum(r.target_inr) ? r.target_inr : 5000);
    let msg, armed = false;
    if (pos && isNum(p30)) {
      const pd = pos.direction > 0 ? p30 : 1 - p30;
      armed = pd >= ext;
      if (pos.target_hit) msg = armed ? "Hold-extension rule ARMED: P(move continues 30 min) = " + prob(pd) + " ≥ " + prob(ext, 0) + "."
        : "Target reached. Extension needs ≥ " + prob(ext, 0) + "; model says " + prob(pd) + ", so the trade exits at the target.";
      else msg = "Extension rule waits for the " + tgt + " target. P(trade direction continues 30 min) now " + prob(pd) + " (needs ≥ " + prob(ext, 0) + ").";
    } else if (isNum(p30)) {
      msg = "No open trade. After the " + tgt + " target the trade is held 30 more minutes only if P ≥ " + prob(ext, 0) + ". Now: " + prob(p30) + " up / " + prob(1 - p30) + " down.";
    } else msg = "Hold-extension rule needs a 30-minute prediction: none available yet.";
    setText("rule-armed", msg + shareTxt);
    setCls("rule-armed", "rule-armed", armed ? "armed" : "");
    renderDrivers(p.top_drivers);
  }
  function renderDrivers(list) {
    const ul = $("drivers");
    const arr = Array.isArray(list) ? list.filter((d) => d && d.feature).slice(0, 8) : [];
    show("drivers-empty", !arr.length);
    const maxAbs = Math.max(1e-9, ...arr.map((d) => (isNum(d.contribution) ? Math.abs(d.contribution) : 0)));
    const keep = new Set(arr.map((d) => d.feature));
    for (const li of [...ul.children]) if (!keep.has(li.dataset.k)) li.remove();
    arr.forEach((d, i) => {
      let li = ul.querySelector('li[data-k="' + CSS.escape(d.feature) + '"]');
      if (!li) {
        li = h("li", { "data-k": d.feature }, h("span", { class: "drv-l" }), h("span", { class: "drv-bar" }, h("i")), h("span", { class: "drv-v" }));
      }
      if (ul.children[i] !== li) ul.insertBefore(li, ul.children[i] || null);
      const [l, bar, v] = li.children;
      setText(l, d.label || d.feature, d.feature + (isNum(d.value) ? " = " + d.value.toPrecision(4) : ""));
      const c = isNum(d.contribution) ? d.contribution : 0;
      const w = (Math.abs(c) / maxAbs) * 50;
      const i2 = bar.firstChild;
      i2.className = c >= 0 ? "up" : "down";
      i2.style.left = (c >= 0 ? 50 : 50 - w) + "%";
      i2.style.width = w + "%";
      setText(v, isNum(d.contribution) ? signed(d.contribution, 3) : "—");
    });
  }

  // =========================================================================== quality
  function renderQuality(s) {
    const mq = s.model_quality || {};
    const rows = [["horizon_30m", "30 min"], ["horizon_close", "To 15:15"], ["daily", "Daily open→close"]];
    const body = $("q-body");
    if (!body.children.length) for (const [k, label] of rows) body.append(h("tr", { "data-k": k }, h("th", { scope: "row" }, label), ...Array.from({ length: 6 }, () => h("td", { class: "r" }))));
    for (const [k] of rows) {
      const q = mq[k] || {};
      const tds = body.querySelector('tr[data-k="' + k + '"]').querySelectorAll("td");
      const missing = !Object.keys(q).length ? "not trained yet" : "";
      setText(tds[0], prob(q.accuracy), missing);
      setText(tds[1], prob(q.base_rate));
      setText(tds[2], isNum(q.auc) ? q.auc.toFixed(3) : "—");
      setText(tds[3], isNum(q.logloss_skill) ? signed(q.logloss_skill, 4) : "—");
      setText(tds[4], pval(q.p_value));
      setText(tds[5], isNum(q.n) ? num(q.n, 0) : "—", q.period ? q.period.join(" → ") : "");
    }
    const q30 = mq.horizon_30m || {};
    const hits = Array.isArray(q30.hit_rate_at) ? q30.hit_rate_at : [];
    const hb = $("q-hits");
    const key = JSON.stringify(hits);
    if (hb.dataset.raw !== key) {
      hb.dataset.raw = key;
      hb.textContent = "";
      if (!hits.length) hb.append(h("tr", null, h("td", { colspan: "3", class: "muted" }, "—")));
      for (const x of hits) hb.append(h("tr", null, h("td", null, "≥ " + prob(x.threshold, 0)), h("td", { class: "r" }, prob(x.coverage, 2)), h("td", { class: "r" }, prob(x.hit_rate))));
    }
    const ext = isNum((s.rules || {}).extend_prob) ? s.rules.extend_prob : 0.8;
    let note = "";
    if (isNum(q30.share_above_80)) {
      note = q30.share_above_80 < 0.01
        ? "The 30-min model reached ≥ " + prob(ext, 0) + " (either direction) on only " + prob(q30.share_above_80, 2) + " of tested bars, so the hold-extension rule will rarely fire; most trades exit at the target."
        : "The 30-min model reached ≥ " + prob(ext, 0) + " on " + prob(q30.share_above_80, 2) + " of tested bars.";
    } else note = "Run training to see how often the model is confident enough for the hold-extension rule.";
    if (q30.period) note += " Tested " + q30.period.join(" → ") + ".";
    setRich("q-note", note);
  }

  // =========================================================================== positions & rules
  const posKey = (p) => (p.book || "?") + ":" + (p.id === undefined ? "?" : p.id);
  function buildPosCard(p) {
    const card = h("div", { class: "pos", "data-k": posKey(p) });
    card.innerHTML =
      '<div class="pos-h"><span class="side"></span><span class="pill small exp" data-term="expiry"></span><span class="pill small book"></span></div>' +
      '<div class="pnl-big"></div>' +
      '<div class="pnlbar" aria-hidden="true"><div class="tr"></div><div class="zero"></div><div class="fill"></div></div>' +
      '<div class="pnl-scale"><span class="sc-stop"></span><span>₹0</span><span class="sc-tgt"></span></div>' +
      '<div class="pos-grid">' +
      '<div><span data-term="premium">Entry</span><b class="v-entry"></b></div>' +
      '<div><span data-term="ltp">LTP</span><b class="v-ltp"></b></div>' +
      '<div><span data-term="lot">Qty</span><b class="v-qty"></b></div>' +
      '<div><span>Entered</span><b class="v-ets"></b></div></div>' +
      '<div class="rule-line"><span class="pill small v-state"></span>' +
      '<span><span data-term="against count">Against</span> <span class="dots"></span> <b class="v-ac"></b></span>' +
      '<span class="v-ext"></span><span class="muted small v-src" data-term="price source"></span></div>' +
      '<div class="rule-note small muted gl-auto"></div>' +
      '<div class="pos-actions"><button class="btn danger" type="button">Exit now (paper)</button></div>';
    card.querySelectorAll("[data-term]").forEach((el) => (el.tabIndex = 0));
    card.querySelector(".btn").addEventListener("click", () => paperExit(card.dataset.book, card.dataset.id));
    return card;
  }
  function renderPositions(s) {
    const list = positionsOf(s);
    const box = $("positions");
    show("pos-empty", !list.length);
    setText("pos-count", list.length ? list.length + " open" : "");
    const keep = new Set(list.map(posKey));
    for (const c of [...box.children]) if (!keep.has(c.dataset.k)) c.remove();
    list.forEach((p, i) => {
      let c = box.querySelector('[data-k="' + CSS.escape(posKey(p)) + '"]');
      if (!c) c = buildPosCard(p);
      if (box.children[i] !== c) box.insertBefore(c, box.children[i] || null);
      c.dataset.book = p.book || "";
      c.dataset.id = p.id === undefined ? "" : String(p.id);
      const dir = p.direction > 0 ? "up" : p.direction < 0 ? "down" : "";
      setText(c.querySelector(".side"), (p.side || "?") + " " + num(p.strike, 0));
      setCls(c.querySelector(".side"), "side", dir);
      setText(c.querySelector(".exp"), "exp " + dateIST(p.expiry));
      setText(c.querySelector(".book"), p.book || "—");
      const pnl = p.pnl_inr;
      const big = c.querySelector(".pnl-big");
      setText(big, isNum(pnl) ? inr(pnl, 0, true) : "—", isNum(pnl) ? "" : "no live option price");
      setCls(big, "pnl-big", isNum(pnl) ? (pnl > 0 ? "up" : pnl < 0 ? "down" : "") : "");
      const stop = isNum(p.stop_inr) ? Math.abs(p.stop_inr) : 2000, tgt = isNum(p.target_inr) ? p.target_inr : 5000;
      const span = stop + tgt, zero = stop / span;
      c.querySelector(".zero").style.left = zero * 100 + "%";
      const fill = c.querySelector(".fill");
      if (isNum(pnl)) {
        const at = clamp((pnl + stop) / span, 0, 1);
        fill.style.left = Math.min(zero, at) * 100 + "%";
        fill.style.width = Math.abs(at - zero) * 100 + "%";
        fill.className = "fill " + (pnl >= 0 ? "up" : "down");
      } else fill.style.width = "0";
      setText(c.querySelector(".sc-stop"), MINUS + "₹" + nf(0).format(stop) + " stop");
      setText(c.querySelector(".sc-tgt"), "+₹" + nf(0).format(tgt) + " target" + (isNum(pnl) && pnl > tgt ? " (beyond)" : ""));
      setText(c.querySelector(".v-entry"), num(p.entry_price, 2));
      setText(c.querySelector(".v-ltp"), num(p.ltp, 2));
      setText(c.querySelector(".v-qty"), (isNum(p.lots) ? p.lots + " lot" + (p.lots === 1 ? "" : "s") : "—") + (isNum(p.qty) ? " · " + p.qty : ""));
      setText(c.querySelector(".v-ets"), hmIST(p.entry_ts));
      const stMap = { WAITING_TARGET: ["", "Waiting for target"], EXTENDED: ["warn", "Extended (P ≥ threshold)"], EXITING: ["bad", "Exiting"] };
      const stv = stMap[p.rule_state] || ["", p.rule_state || "—"];
      setText(c.querySelector(".v-state"), (p.target_hit ? "Target hit · " : "") + stv[1]);
      setCls(c.querySelector(".v-state"), "pill small v-state", stv[0]);
      const lim = isNum(p.against_limit) ? p.against_limit : 2, ac = isNum(p.against_count) ? p.against_count : 0;
      const dots = c.querySelector(".dots");
      if (dots.children.length !== lim) { dots.textContent = ""; for (let k = 0; k < lim; k++) dots.append(h("i")); }
      [...dots.children].forEach((d, k) => d.classList.toggle("on", k < ac));
      setText(c.querySelector(".v-ac"), ac + "/" + lim);
      const ex = c.querySelector(".v-ext");
      if (p.extend_until) { ex.dataset.until = p.extend_until; updateCountdown(ex); } else { delete ex.dataset.until; setText(ex, ""); }
      setText(c.querySelector(".v-src"), p.price_source || "—");
      setRich(c.querySelector(".rule-note"), p.rule_note || "");
    });
  }
  function updateCountdown(el) {
    const d = toDate(el.dataset.until);
    if (!d) { setText(el, ""); return; }
    const sec = Math.round((d.getTime() - Date.now()) / 1000);
    setText(el, sec > 0 ? "extension ends in " + Math.floor(sec / 60) + ":" + String(sec % 60).padStart(2, "0") + " (" + hmIST(d) + ")" : "extension window over: re-checking");
  }
  function renderRules(s) {
    const r = s.rules;
    if (!r) { setText("rules-summary", "Rules: —"); return; }
    const scope = { in_profit: "while in profit", always: "at any time", after_target: "after the target" }[r.against_scope] || r.against_scope || "";
    const win = Array.isArray(r.entry_window) ? r.entry_window.join("–") : "—";
    const txt = "Rules: target " + inr(r.target_inr, 0, true) + " · stop " + (isNum(r.stop_inr) ? MINUS + "₹" + nf(0).format(Math.abs(r.stop_inr)) : "—") +
      " · after target hold " + (r.extend_minutes ?? "—") + " min more if P ≥ " + prob(r.extend_prob, 0) +
      " · exit after " + (r.against_bars ?? "—") + " consecutive 5-min closes against " + scope +
      " · enter when P(up by 15:15) ≥ " + prob(r.entry_threshold, 0) + " (CE) or ≤ " + (isNum(r.entry_threshold) ? prob(1 - r.entry_threshold, 0) : "—") + " (PE), " + win +
      " · exit by " + (r.exit_time || "—") + " · " + (r.lots ?? "—") + " lot.";
    setRich("rules-summary", txt);
  }

  // =========================================================================== books
  const BOOK_SLOTS = ["model_rules", "agent", "manual"];
  const bookColorVar = (name) => {
    let i = BOOK_SLOTS.indexOf(name);
    if (i < 0) { BOOK_SLOTS.push(name); i = BOOK_SLOTS.length - 1; }
    return "--series-" + (Math.min(i, 3) + 1);
  };
  function renderBooks(s) {
    const books = Array.isArray(s.books) ? s.books.filter((b) => b && b.name) : [];
    const body = $("books-body");
    show("books-empty", !books.length);
    const keep = new Set(books.map((b) => b.name));
    for (const tr of [...body.children]) if (!keep.has(tr.dataset.k)) tr.remove();
    books.forEach((b, i) => {
      let tr = body.querySelector('tr[data-k="' + CSS.escape(b.name) + '"]');
      if (!tr) {
        const cb = h("input", { type: "checkbox", "aria-label": "Auto-trade " + b.name });
        cb.addEventListener("change", () => setAuto(b.name, cb));
        tr = h("tr", { "data-k": b.name },
          h("th", { scope: "row" }, h("span", { class: "sw-dot" }), h("span", { class: "bk-l" })),
          h("td", null, h("label", { class: "switch" }, cb, h("span"))),
          ...Array.from({ length: 6 }, () => h("td", { class: "r" })));
      }
      if (body.children[i] !== tr) body.insertBefore(tr, body.children[i] || null);
      tr.querySelector(".sw-dot").style.background = "var(" + bookColorVar(b.name) + ")";
      setText(tr.querySelector(".bk-l"), b.label || b.name, b.name);
      const cb = tr.querySelector("input");
      if (!cb.dataset.pending) cb.checked = !!b.auto;
      if (b.name === "manual") { cb.disabled = true; cb.title = "The manual book trades only from the buttons"; }
      const tds = tr.querySelectorAll("td.r");
      setText(tds[0], inr(b.equity));
      setText(tds[1], inr(b.realized_pnl, 0, true)); tds[1].className = "r " + (b.realized_pnl > 0 ? "pos-v" : b.realized_pnl < 0 ? "neg-v" : "");
      setText(tds[2], inr(b.today_pnl, 0, true)); tds[2].className = "r " + (b.today_pnl > 0 ? "pos-v" : b.today_pnl < 0 ? "neg-v" : "");
      setText(tds[3], isNum(b.trades) ? num(b.trades, 0) : "—");
      setText(tds[4], prob(b.win_rate), isNum(b.win_rate) ? "" : "no closed trades yet");
      setText(tds[5], b.open ? "Yes" : "No");
    });
  }
  async function setAuto(book, cb) {
    cb.dataset.pending = "1";
    const want = cb.checked;
    const res = await postJSON("/api/auto", { book, enabled: want });
    delete cb.dataset.pending;
    if (!res.ok) { cb.checked = !want; toast("Could not change auto-trading for " + book + ": " + res.error, "err"); }
    else toast("Auto-trading " + (want ? "ON" : "OFF") + " for " + book + " (paper)", "ok");
  }

  // =========================================================================== agent
  function renderAgent(s) {
    const a = s.agent || {};
    dot("agent-avail", bool(a.available), a.available ? "Local LLM reachable" : "Local LLM not reachable: start Ollama and pull the model");
    setText("agent-model", a.model || "no model configured");
    const d = a.last_decision || null;
    const map = { ENTER_LONG: ["up", "Enter long (CE)"], ENTER_SHORT: ["down", "Enter short (PE)"], EXIT: ["exit", "Exit"], HOLD: ["", "Hold"], STAND_ASIDE: ["", "Stand aside"] };
    if (d) {
      const [cls, label] = map[d.action] || ["", d.action || "—"];
      setText("ag-action", label); setCls("ag-action", "action-pill", cls);
      setText("ag-conf", prob(d.confidence, 0)); setText("ag-p", prob(d.p_up_30m)); setText("ag-ts", timeIST(d.ts));
      show("ag-guard", !!d.guardrail);
      setText("ag-guard", d.guardrail || "");
      setRich("ag-reason", d.reasoning || "");
      setList("ag-factors", d.key_factors); setList("ag-risks", d.risks);
    } else {
      setText("ag-action", "—"); setCls("ag-action", "action-pill");
      setText("ag-conf", "—"); setText("ag-p", "—"); setText("ag-ts", "—"); show("ag-guard", false);
      setRich("ag-reason", a.available === false ? "The agent is idle: the local LLM is not reachable. Start Ollama (ollama serve) and pull the configured model." : "No agent decision yet today.");
      setList("ag-factors", []); setList("ag-risks", []);
    }
    setText("ag-count", isNum(a.decisions_today) ? num(a.decisions_today, 0) : "—");
    const b = a.brief || null;
    if (b) {
      const sc = { bullish: "good", bearish: "bad" }[b.stance] || "";
      setText("brief-stance", b.stance || "—"); setCls("brief-stance", "pill small", sc);
      setText("brief-p", isNum(b.p_up) ? "P(up) " + prob(b.p_up) : "");
      setRich("brief-narr", b.narrative || "");
      setList("brief-drivers", b.drivers); setList("brief-risks", b.risks);
    } else {
      setText("brief-stance", "none"); setCls("brief-stance", "pill small"); setText("brief-p", "");
      setRich("brief-narr", "No pre-open brief today (written at about 09:20 when the local LLM is running).");
      setList("brief-drivers", []); setList("brief-risks", []);
    }
  }

  // =========================================================================== option chain
  function renderChain(s) {
    const c = s.chain;
    show("chain-tiles", !!c); show("chain-empty", !c);
    if (!c) { setText("chain-meta", ""); return; }
    setText("c-pcr", isNum(c.pcr) ? c.pcr.toFixed(2) : "—");
    setText("c-maxpain", num(c.max_pain, 0)); setText("c-cwall", num(c.call_wall, 0)); setText("c-pwall", num(c.put_wall, 0));
    setText("c-iv", isNum(c.atm_iv) ? c.atm_iv.toFixed(1) + "%" : "—");
    setText("c-move", isNum(c.implied_move_pct) ? "±" + c.implied_move_pct.toFixed(2) + "%" : "—");
    setText("c-gex", isNum(c.gex_cr) ? inr(c.gex_cr, 0, true) + " Cr" : "—");
    setText("c-zg", num(c.zero_gamma, 0));
    setText("c-rr", isNum(c.rr25) ? signed(c.rr25, 1) : "—");
    setText("chain-meta", (c.expiry ? "expiry " + dateIST(c.expiry) : "") + (c.updated ? " · " + timeIST(c.updated) : ""));
  }

  // =========================================================================== news
  const NEWS = new Map();
  let newsCat = "__all", newsInit = false;
  const CAT_ORDER = [];
  function mergeNews(items) {
    if (!Array.isArray(items)) return;
    for (const it of items) if (it && it.id) {
      const old = NEWS.get(it.id);
      NEWS.set(it.id, Object.assign({}, old || {}, it, { _new: !old && newsInit }));
      if (it.category && !CAT_ORDER.includes(it.category)) CAT_ORDER.push(it.category);
    }
    if (NEWS.size > 300) {
      const sorted = [...NEWS.values()].sort((a, b) => (toDate(b.first_seen) || 0) - (toDate(a.first_seen) || 0));
      for (const it of sorted.slice(300)) NEWS.delete(it.id);
    }
  }
  function renderNewsFilters() {
    const box = $("news-filters");
    const have = new Set([...box.querySelectorAll(".chip")].map((b) => b.dataset.cat));
    const india = box.querySelector(".chk");
    for (const cat of CAT_ORDER) if (!have.has(cat)) {
      const b = h("button", { class: "chip", "data-cat": cat, type: "button" }, cat);
      box.insertBefore(b, india);
    }
    box.querySelectorAll(".chip").forEach((b) => b.classList.toggle("on", b.dataset.cat === newsCat));
  }
  function newsItemEl(it) {
    const li = h("li", { "data-k": it.id },
      h("span", { class: "nbar" }),
      h("div", null,
        h("div", { class: "n-t" }),
        h("div", { class: "n-meta" }),
        h("div", { class: "n-reason" }),
        h("div", { class: "n-impact" })));
    return li;
  }
  function fillNewsItem(li, it) {
    const llm = it.llm || {};
    const key = JSON.stringify([it.title, it.url, llm, it.impact, it.source, it.category]) + GLOSS_VERSION;
    const dirv = isNum(llm.direction) ? llm.direction : null;
    const bar = li.querySelector(".nbar");
    bar.className = "nbar " + (dirv === null || Math.abs(dirv) < 0.1 ? "" : dirv > 0 ? "up" : "down");
    bar.style.opacity = String(0.35 + 0.65 * clamp(isNum(llm.magnitude) ? llm.magnitude : 0.3, 0, 1));
    const ageEl = li.querySelector(".n-age");
    if (li.dataset.raw === key) { if (ageEl) setText(ageEl, ageText(it.first_seen) + " · " + hmIST(it.first_seen)); return; }
    li.dataset.raw = key;
    const t = li.querySelector(".n-t");
    t.textContent = "";
    const url = safeUrl(it.url);
    const title = url ? h("a", { class: "n-title", href: url, target: "_blank", rel: "noopener noreferrer" }, it.title || "(untitled)")
      : h("span", { class: "n-title" }, it.title || "(untitled)");
    t.append(title);
    glossify(title);
    const meta = li.querySelector(".n-meta");
    meta.textContent = "";
    const lab = (term, text) => h("span", { "data-term": term, tabindex: "0" }, text);
    meta.append(h("span", { class: "n-src" }, it.source || "?"));
    if (it.category) meta.append(h("span", null, it.category));
    meta.append(h("span", { class: "n-age" }, ageText(it.first_seen) + " · " + hmIST(it.first_seen)));
    if (dirv !== null) {
      const cls = Math.abs(dirv) < 0.1 ? "flat" : dirv > 0 ? "up" : "down";
      meta.append(h("span", { class: "n-dir " + cls, title: "LLM direction for NIFTY (−1 bearish … +1 bullish)" },
        (cls === "up" ? "▲ " : cls === "down" ? "▼ " : "● ") + signed(dirv, 2)));
    }
    if (isNum(llm.magnitude)) meta.append(h("span", null, lab("magnitude", "mag"), " " + llm.magnitude.toFixed(2)));
    if (isNum(llm.surprise)) meta.append(h("span", null, lab("surprise", "surprise"), " " + llm.surprise.toFixed(2)));
    if (isNum(llm.relevance)) meta.append(h("span", null, lab("relevance", "India rel."), " " + llm.relevance.toFixed(2)));
    if (isNum(llm.confidence)) meta.append(h("span", null, lab("confidence", "conf"), " " + llm.confidence.toFixed(2)));
    if (llm.channel || llm.horizon) meta.append(h("span", null, lab("channel", llm.channel || "—"), llm.horizon ? " · " + llm.horizon : ""));
    if (llm.scorer) meta.append(h("span", { title: "who scored this headline" }, "by " + llm.scorer));
    const rs = li.querySelector(".n-reason");
    rs.textContent = llm.reason || (it.llm ? "" : "Not scored yet.");
    glossify(rs);
    const im = li.querySelector(".n-impact");
    im.textContent = "";
    const imp = it.impact || {};
    if (isNum(imp.exp_bps_30m) || isNum(imp.p_up_30m)) {
      im.append(h("span", { "data-term": "news impact", tabindex: "0" }, "Learned impact"), ": ",
        h("b", null, isNum(imp.exp_bps_30m) ? signed(imp.exp_bps_30m, 1) + " bps" : "—"), " in 30 min",
        isNum(imp.p_up_30m) ? " · P(up 30m) " + prob(imp.p_up_30m, 0) : "", imp.basis ? " · " + imp.basis : "");
    } else im.append(h("span", { class: "muted" }, "Learned impact: — (" + (imp.basis || "not enough past headlines of this kind yet") + ")"));
  }
  function renderNews(s) {
    mergeNews(s.news);
    renderNewsFilters();
    const indiaOnly = $("news-india").checked;
    let items = [...NEWS.values()].sort((a, b) => (toDate(b.first_seen) || 0) - (toDate(a.first_seen) || 0));
    if (newsCat !== "__all") items = items.filter((it) => it.category === newsCat);
    if (indiaOnly) items = items.filter((it) => it.llm && isNum(it.llm.relevance) && it.llm.relevance >= 0.5);
    items = items.slice(0, 120);
    const ol = $("news-list");
    show("news-empty", !items.length);
    const keep = new Set(items.map((it) => it.id));
    for (const li of [...ol.children]) if (!keep.has(li.dataset.k)) li.remove();
    items.forEach((it, i) => {
      let li = ol.querySelector('li[data-k="' + CSS.escape(it.id) + '"]');
      if (!li) { li = newsItemEl(it); if (it._new) { li.classList.add("flash"); it._new = false; } }
      if (ol.children[i] !== li) ol.insertBefore(li, ol.children[i] || null);
      fillNewsItem(li, it);
    });
    newsInit = true;
  }
  function initNewsFilters() {
    $("news-filters").addEventListener("click", (e) => {
      const b = e.target.closest(".chip");
      if (!b) return;
      newsCat = b.dataset.cat;
      if (LAST) renderNews(LAST); else renderNewsFilters();
    });
    $("news-india").addEventListener("change", () => LAST && renderNews(LAST));
  }
  async function fetchNews() {
    try {
      const r = await fetch("/api/news?limit=100", { cache: "no-store" });
      if (!r.ok) return;
      const js = await r.json();
      mergeNews(Array.isArray(js) ? js : js.items || js.news || []);
      if (LAST) renderNews(LAST);
    } catch (e) { /* engine offline */ }
  }

  // =========================================================================== sources, alerts, health
  function renderSources(s) {
    const src = Array.isArray(s.sources) ? s.sources.filter((x) => x && x.name) : [];
    show("src-empty", !src.length);
    const body = $("src-body");
    const keep = new Set(src.map((x) => x.name));
    for (const tr of [...body.children]) if (!keep.has(tr.dataset.k)) tr.remove();
    src.forEach((x, i) => {
      let tr = body.querySelector('tr[data-k="' + CSS.escape(x.name) + '"]');
      if (!tr) tr = h("tr", { "data-k": x.name }, h("th", { scope: "row" }), h("td"), h("td"), h("td"), h("td", { class: "r" }), h("td", { class: "r" }), h("td", { class: "r" }), h("td", { class: "err" }));
      if (body.children[i] !== tr) body.insertBefore(tr, body.children[i] || null);
      const c = tr.children;
      setText(c[0], x.name); setText(c[1], x.category || "—");
      setText(c[2], x.ok ? "OK" : x.ok === false ? "Failing" : "—"); c[2].className = "st " + (x.ok ? "ok" : x.ok === false ? "bad" : "");
      setText(c[3], x.last_ok ? timeIST(x.last_ok) : "never");
      setText(c[4], isNum(x.latency_ms) ? num(x.latency_ms, 0) + " ms" : "—");
      setText(c[5], isNum(x.items_new_today) ? num(x.items_new_today, 0) : "—");
      setText(c[6], isNum(x.poll_s) ? x.poll_s + " s" : "—");
      setText(c[7], x.error || "");
    });
    const ok = src.filter((x) => x.ok).length;
    setText("src-summary", src.length ? ok + " of " + src.length + " sources OK" : "");
  }
  function renderAlerts(s) {
    const al = Array.isArray(s.alerts) ? s.alerts.filter(Boolean) : [];
    const key = JSON.stringify(al) + GLOSS_VERSION;
    const ul = $("alerts");
    show("alerts-empty", !al.length);
    const n = al.filter((a) => a.level === "warn" || a.level === "error").length;
    show("alert-count", n > 0); setText("alert-count", String(n));
    if (ul.dataset.raw === key) return;
    ul.dataset.raw = key;
    ul.textContent = "";
    for (const a of al) {
      const lvl = ["info", "warn", "error"].includes(a.level) ? a.level : "info";
      const txt = h("span", null, a.text || "");
      ul.append(h("li", null, h("span", { class: "lvl " + lvl }, lvl), h("span", { class: "muted" }, a.ts ? hmIST(a.ts) : ""), txt));
      glossify(txt);
    }
  }
  function renderHealth(s) {
    const hl = s.health || {};
    setText("hk-engine", hl.engine || "—");
    setText("hk-lag", isNum(hl.loop_lag_s) ? hl.loop_lag_s.toFixed(1) + " s" : "—");
    setText("hk-bar", hl.last_bar ? timeIST(hl.last_bar) : "—");
    setText("hk-up", isNum(hl.uptime_s) ? Math.floor(hl.uptime_s / 3600) + " h " + Math.floor((hl.uptime_s % 3600) / 60) + " min" : "—");
  }

  // =========================================================================== trades & equity
  let TRADES = [], tradeRange = "today", EQUITY = {};
  async function fetchTrades() {
    try {
      const r = await fetch("/api/trades?limit=200", { cache: "no-store" });
      if (!r.ok) return;
      const js = await r.json();
      TRADES = Array.isArray(js) ? js : js.trades || [];
      renderTrades();
      if (LAST) renderCharts(LAST);
    } catch (e) { /* offline */ }
  }
  async function fetchEquity() {
    try {
      const r = await fetch("/api/equity", { cache: "no-store" });
      if (!r.ok) return;
      EQUITY = (await r.json()) || {};
      updateEquityChart();
    } catch (e) { /* offline */ }
  }
  function tradeLeg(t) { const legs = parseMaybeJSON(t.legs); return Array.isArray(legs) && legs.length ? legs[0] : null; }
  function renderTrades() {
    const today = LAST ? sessionDate(LAST) : todayIST();
    const rows = TRADES.filter((t) => tradeRange === "all" || String(t.date || "").slice(0, 10) === today);
    const body = $("trades-body");
    show("trades-empty", !rows.length);
    const key = JSON.stringify(rows);
    if (body.dataset.raw === key) return;
    body.dataset.raw = key;
    body.textContent = "";
    for (const t of rows) {
      const leg = tradeLeg(t);
      const side = (leg && leg.typ) || (t.direction > 0 ? "CE" : t.direction < 0 ? "PE" : "—");
      const net = isNum(t.net_pnl) ? t.net_pnl : null;
      body.append(h("tr", null,
        h("td", null, t.id ?? "—"), h("td", null, t.book || "—"), h("td", null, t.date || "—"),
        h("td", null, hmIST(t.entry_ts)), h("td", null, t.exit_ts ? hmIST(t.exit_ts) : "—"),
        h("td", null, (t.direction > 0 ? "▲ " : t.direction < 0 ? "▼ " : "") + side),
        h("td", { class: "r" }, leg ? num(Number(leg.strike), 0) : "—"),
        h("td", { class: "r" }, num(t.entry_value, 2)), h("td", { class: "r" }, num(t.exit_value, 2)),
        h("td", { class: "r " + (net > 0 ? "pos-v" : net < 0 ? "neg-v" : "") }, net === null ? "—" : inr(net, 0, true)),
        h("td", { class: "r" }, inr(t.charges, 0)), h("td", null, t.exit_reason || "—"), h("td", null, t.status || "—"),
        h("td", null, t.price_source || "—")));
    }
  }
  function initTradeRange() {
    for (const [id, v] of [["tr-today", "today"], ["tr-all", "all"]]) {
      $(id).addEventListener("click", () => {
        tradeRange = v;
        $("tr-today").classList.toggle("on", v === "today"); $("tr-today").setAttribute("aria-pressed", String(v === "today"));
        $("tr-all").classList.toggle("on", v === "all"); $("tr-all").setAttribute("aria-pressed", String(v === "all"));
        $("trades-body").dataset.raw = "";
        renderTrades();
      });
    }
  }

  // =========================================================================== paper actions
  async function postJSON(url, body) {
    try {
      const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      let js = null;
      try { js = await r.json(); } catch (e) { /* non-JSON */ }
      if (!r.ok) return { ok: false, error: (js && (js.error || js.detail)) || "HTTP " + r.status };
      if (js && js.ok === false) return { ok: false, error: js.error || "rejected by engine" };
      return Object.assign({ ok: true }, js || {});
    } catch (e) { return { ok: false, error: "engine not reachable" }; }
  }
  async function paperOrder(action, label) {
    if (!window.confirm(label + "\n\nThis is a PAPER trade: no real order is sent.")) return;
    const btns = ["btn-buy-ce", "btn-buy-pe"].map($);
    btns.forEach((b) => (b.disabled = true));
    const res = await postJSON("/api/paper/order", { action, book: "manual" });
    btns.forEach((b) => (b.disabled = connMode === "off"));
    if (res.ok) { toast("Paper order done" + (res.trade_id !== undefined ? " (trade #" + res.trade_id + ")" : ""), "ok"); fetchTrades(); pollOnce(); }
    else toast("Paper order rejected: " + res.error, "err");
  }
  async function paperExit(book, id) {
    if (!window.confirm("Exit the " + (book || "?") + " position #" + (id || "?") + " now at the market (paper)?")) return;
    const res = await postJSON("/api/paper/order", { action: "exit", book: book || "manual" });
    if (res.ok) { toast("Paper exit done for " + book, "ok"); fetchTrades(); pollOnce(); }
    else toast("Exit rejected: " + res.error, "err");
  }

  // =========================================================================== charts
  const CHARTS = {};
  function niceStep(span, count) {
    const raw = span / Math.max(1, count);
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    const m = raw / p;
    return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p;
  }
  function niceTicks(lo, hi, count) {
    if (!(hi > lo)) return [lo];
    const st = niceStep(hi - lo, count);
    const out = [];
    for (let v = Math.ceil(lo / st) * st; v <= hi + 1e-9; v += st) out.push(+v.toFixed(10));
    return out;
  }
  function sessionBounds(dateStr) {
    const a = Date.parse(dateStr + "T09:15:00+05:30"), b = Date.parse(dateStr + "T15:30:00+05:30");
    return isNaN(a) ? null : [a, b];
  }
  class Chart {
    constructor(canvas, wrap) {
      this.c = canvas; this.wrap = wrap; this.ctx = canvas.getContext("2d");
      this.pts = []; this.hover = null; this.pending = false;
      this.c.tabIndex = 0;
      new ResizeObserver(() => this.resize()).observe(wrap);
      this.c.addEventListener("pointermove", (e) => this.onMove(e));
      this.c.addEventListener("pointerleave", () => { this.hover = null; hideChartTip(); this.requestDraw(); });
      this.c.addEventListener("keydown", (e) => this.onKey(e));
      this.c.addEventListener("blur", () => { this.hover = null; hideChartTip(); this.requestDraw(); });
      this.resize();
    }
    resize() {
      const r = this.wrap.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      this.w = Math.max(10, r.width); this.h = Math.max(10, r.height);
      this.c.width = Math.round(this.w * dpr); this.c.height = Math.round(this.h * dpr);
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.requestDraw();
    }
    requestDraw() {
      if (this.pending) return;
      this.pending = true;
      requestAnimationFrame(() => { this.pending = false; this.colors = this.readColors(); this.draw(); });
    }
    readColors() {
      const v = (n) => cssVar(n);
      return { surface: v("--surface-1"), grid: v("--grid"), axis: v("--axis"), muted: v("--text-muted"), text: v("--text-primary"),
        text2: v("--text-secondary"), s1: v("--series-1"), s2: v("--series-2"), s3: v("--series-3"), s4: v("--series-4"),
        good: v("--good"), bad: v("--critical"), warn: v("--warning"), surface3: v("--surface-3") };
    }
    nearest(px) {
      if (!this.pts.length) return null;
      let best = 0, bd = Infinity;
      for (let i = 0; i < this.pts.length; i++) { const d = Math.abs(this.pts[i].x - px); if (d < bd) { bd = d; best = i; } }
      return best;
    }
    onMove(e) {
      const r = this.c.getBoundingClientRect();
      const i = this.nearest(e.clientX - r.left);
      if (i === null) return;
      this.hover = i;
      this.requestDraw();
      showChartTip(this.tip(i), e.clientX, e.clientY);
    }
    onKey(e) {
      if (!this.pts.length || (e.key !== "ArrowLeft" && e.key !== "ArrowRight" && e.key !== "Home" && e.key !== "End")) return;
      e.preventDefault();
      let i = this.hover === null ? this.pts.length - 1 : this.hover;
      if (e.key === "ArrowLeft") i = Math.max(0, i - 1);
      else if (e.key === "ArrowRight") i = Math.min(this.pts.length - 1, i + 1);
      else if (e.key === "Home") i = 0; else i = this.pts.length - 1;
      this.hover = i;
      this.requestDraw();
      const r = this.c.getBoundingClientRect();
      showChartTip(this.tip(i), r.left + this.pts[i].x, r.top + (this.pts[i].y || this.h / 2));
    }
    line(x1, y1, x2, y2, color, w = 1) {
      const g = this.ctx; g.strokeStyle = color; g.lineWidth = w; g.beginPath();
      g.moveTo(Math.round(x1) + 0.5, Math.round(y1) + 0.5); g.lineTo(Math.round(x2) + 0.5, Math.round(y2) + 0.5); g.stroke();
    }
    text(t, x, y, color, align = "left", base = "middle", font = "11px system-ui, -apple-system, Segoe UI, sans-serif") {
      const g = this.ctx; g.fillStyle = color; g.font = font; g.textAlign = align; g.textBaseline = base; g.fillText(t, x, y);
    }
    dot(x, y, color, r = 4) {
      const g = this.ctx;
      g.beginPath(); g.arc(x, y, r + 2, 0, Math.PI * 2); g.fillStyle = this.colors.surface; g.fill();
      g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.fillStyle = color; g.fill();
    }
    poly(points, color) {
      const g = this.ctx; g.strokeStyle = color; g.lineWidth = 2; g.lineJoin = "round"; g.lineCap = "round";
      g.beginPath();
      let pen = false;
      for (const p of points) {
        if (!p) { pen = false; continue; }
        if (!pen) { g.moveTo(p[0], p[1]); pen = true; } else g.lineTo(p[0], p[1]);
      }
      g.stroke();
    }
    clear() { this.ctx.clearRect(0, 0, this.w, this.h); }
  }
  function showChartTip(rows, x, y) {
    const t = $("chart-tip");
    t.textContent = "";
    if (!rows) { t.hidden = true; return; }
    t.append(h("div", { class: "ct-h" }, rows.head));
    for (const r of rows.rows) {
      const key = h("span", { class: "ct-k" });
      if (r.color) key.append(h("i", { class: "ln", style: "background:" + r.color }));
      key.append(r.label);
      t.append(h("div", { class: "ct-r" }, h("b", null, r.value), key));
    }
    t.hidden = false;
    const tw = t.offsetWidth, th = t.offsetHeight;
    t.style.left = clamp(x + 14, 4, window.innerWidth - tw - 4) + "px";
    t.style.top = clamp(y - th - 10, 4, window.innerHeight - th - 4) + "px";
  }
  function hideChartTip() { $("chart-tip").hidden = true; }

  // --------------------------------------------------------------------------- price chart
  class PriceChart extends Chart {
    constructor(c, w) { super(c, w); this.mode = "line"; this.bars = []; this.markers = []; this.strike = null; this.prevClose = null; this.bounds = null; }
    set(data) { Object.assign(this, data); this.requestDraw(); }
    draw() {
      this.clear();
      const bars = this.bars;
      show("price-empty", !bars.length);
      this.pts = [];
      if (!bars.length) return;
      const C = this.colors, g = this.ctx;
      const L = 8, R = 64, T = 10, B = 22, W = this.w - L - R, H = this.h - T - B;
      let x0 = bars[0].t_ms, x1 = bars[bars.length - 1].t_ms;
      if (this.bounds) { x0 = Math.min(x0, this.bounds[0]); x1 = Math.max(x1, this.bounds[1]); }
      if (x1 <= x0) x1 = x0 + 60000;
      let lo = Infinity, hi = -Infinity;
      for (const b of bars) { lo = Math.min(lo, isNum(b.l) ? b.l : b.c); hi = Math.max(hi, isNum(b.h) ? b.h : b.c); }
      const mid = (lo + hi) / 2;
      const near = (v) => isNum(v) && Math.abs(v - mid) / mid < 0.012;
      for (const v of [this.strike && this.strike.price, this.prevClose, ...this.markers.map((m) => m.price)]) if (near(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      const pad = Math.max((hi - lo) * 0.06, 2);
      lo -= pad; hi += pad;
      const X = (t) => L + ((t - x0) / (x1 - x0)) * W;
      const Y = (v) => T + (1 - (v - lo) / (hi - lo)) * H;
      // grid + y ticks (right)
      for (const v of niceTicks(lo, hi, 5)) {
        const y = Y(v); this.line(L, y, L + W, y, C.grid);
        this.text(nf(0).format(v), L + W + 6, y, C.muted);
      }
      this.line(L, T + H, L + W, T + H, C.axis);
      // hour ticks
      const ds = ymdIST(bars[0].t_ms);
      for (let hh = 10; hh <= 15; hh++) {
        const t = Date.parse(ds + "T" + String(hh).padStart(2, "0") + ":00:00+05:30");
        if (t < x0 || t > x1) continue;
        const x = X(t); this.line(x, T + H, x, T + H + 4, C.axis);
        this.text(hh + ":00", x, T + H + 13, C.muted, "center");
      }
      // reference lines
      if (near(this.prevClose)) { const y = Y(this.prevClose); this.line(L, y, L + W, y, C.axis); this.text("prev close " + nf(0).format(this.prevClose), L + 4, y - 7, C.muted); }
      if (this.strike && near(this.strike.price)) {
        const y = Y(this.strike.price); this.line(L, y, L + W, y, C.warn);
        this.text(this.strike.label, L + W - 4, y - 7, C.text2, "right");
      }
      // series
      if (this.mode === "candle") {
        const cw = clamp((W / Math.max(60, (x1 - x0) / 60000)) * 0.7, 1, 7);
        for (const b of bars) {
          const x = X(b.t_ms), col = b.c > b.o ? C.good : b.c < b.o ? C.bad : C.muted;
          const hi2 = isNum(b.h) ? b.h : Math.max(b.o, b.c), lo2 = isNum(b.l) ? b.l : Math.min(b.o, b.c);
          this.line(x, Y(hi2), x, Y(lo2), col);
          const o = isNum(b.o) ? b.o : b.c;
          const y1 = Y(Math.max(o, b.c)), y2 = Y(Math.min(o, b.c));
          g.fillStyle = col; g.fillRect(Math.round(x - cw / 2), Math.round(y1), Math.max(1, Math.round(cw)), Math.max(1, Math.round(y2 - y1)));
        }
      } else {
        const ptsXY = bars.map((b) => [X(b.t_ms), Y(b.c)]);
        g.beginPath(); g.moveTo(ptsXY[0][0], T + H);
        for (const p of ptsXY) g.lineTo(p[0], p[1]);
        g.lineTo(ptsXY[ptsXY.length - 1][0], T + H); g.closePath();
        g.globalAlpha = 0.10; g.fillStyle = C.s1; g.fill(); g.globalAlpha = 1;
        this.poly(ptsXY, C.s1);
      }
      this.pts = bars.map((b) => ({ x: X(b.t_ms), y: Y(b.c), b }));
      // last price
      const last = bars[bars.length - 1];
      const ly = Y(last.c);
      this.dot(X(last.t_ms), ly, this.mode === "candle" ? C.text2 : C.s1, 4);
      const lbl = nf(2).format(last.c);
      g.font = "600 11px system-ui, -apple-system, Segoe UI, sans-serif";
      const tw = g.measureText(lbl).width + 10;
      g.fillStyle = C.surface3; g.beginPath();
      if (typeof g.roundRect === "function") g.roundRect(L + W + 2, ly - 9, tw, 18, 4); else g.rect(L + W + 2, ly - 9, tw, 18);
      g.fill();
      this.text(lbl, L + W + 7, ly, C.text, "left", "middle", "600 11px system-ui, -apple-system, Segoe UI, sans-serif");
      // markers
      for (const m of this.markers) {
        if (m.t_ms < x0 || m.t_ms > x1 || !isNum(m.price)) continue;
        const x = X(m.t_ms), y = Y(m.price);
        if (m.kind === "exit") { this.dot(x, y, C.text2, 4.5); continue; }
        const up = m.dir > 0, s = 7, yy = up ? y + 12 : y - 12;
        g.beginPath();
        if (up) { g.moveTo(x, yy - s); g.lineTo(x - s, yy + s * 0.6); g.lineTo(x + s, yy + s * 0.6); }
        else { g.moveTo(x, yy + s); g.lineTo(x - s, yy - s * 0.6); g.lineTo(x + s, yy - s * 0.6); }
        g.closePath(); g.lineWidth = 2; g.strokeStyle = C.surface; g.stroke(); g.fillStyle = up ? C.good : C.bad; g.fill();
      }
      // crosshair
      if (this.hover !== null && this.pts[this.hover]) {
        const p = this.pts[this.hover];
        this.line(p.x, T, p.x, T + H, C.axis);
        this.dot(p.x, p.y, C.s1, 4);
      }
    }
    tip(i) {
      const p = this.pts[i]; if (!p) return null;
      const b = p.b;
      const rows = [{ label: "close", value: nf(2).format(b.c), color: this.colors.s1 }];
      if (isNum(b.o)) rows.push({ label: "open", value: nf(2).format(b.o) }, { label: "high", value: nf(2).format(b.h) }, { label: "low", value: nf(2).format(b.l) });
      if (isNum(this.prevClose)) rows.push({ label: "vs prev close", value: signed(b.c - this.prevClose, 2) });
      for (const m of this.markers) if (Math.abs(m.t_ms - b.t_ms) < 90000) rows.push({ label: m.label, value: m.kind === "exit" ? "exit" : "entry" });
      return { head: timeIST(b.t_ms) + " IST", rows };
    }
  }

  // --------------------------------------------------------------------------- prediction chart
  class PredChart extends Chart {
    constructor(c, w) { super(c, w); this.hist = []; this.thr = 0.6; this.ext = 0.8; this.bounds = null; }
    set(d) { Object.assign(this, d); this.requestDraw(); }
    draw() {
      this.clear();
      const hs = this.hist;
      show("pred-empty", !hs.length);
      this.pts = [];
      if (!hs.length) return;
      const C = this.colors;
      const L = 8, R = 92, T = 8, B = 22, W = this.w - L - R, H = this.h - T - B;
      let x0 = hs[0].t_ms, x1 = hs[hs.length - 1].t_ms;
      if (this.bounds) { x0 = Math.min(x0, this.bounds[0]); x1 = Math.max(x1, this.bounds[1]); }
      if (x1 <= x0) x1 = x0 + 300000;
      let lo = Math.min(0.15, 1 - this.ext - 0.05), hi = Math.max(0.85, this.ext + 0.05);
      for (const p of hs) for (const v of [p.p30, p.pclose]) if (isNum(v)) { lo = Math.min(lo, v - 0.03); hi = Math.max(hi, v + 0.03); }
      lo = Math.max(0, lo); hi = Math.min(1, hi);
      const X = (t) => L + ((t - x0) / (x1 - x0)) * W;
      const Y = (v) => T + (1 - (v - lo) / (hi - lo)) * H;
      for (const v of niceTicks(lo, hi, 4)) { const y = Y(v); this.line(L, y, L + W, y, C.grid); }
      this.line(L, T + H, L + W, T + H, C.axis);
      const ref = (v, color, label) => { if (v < lo || v > hi) return; const y = Y(v); this.line(L, y, L + W, y, color); this.text(label, L + W + 6, y, C.muted); };
      ref(0.5, C.axis, "50%");
      ref(this.thr, C.axis, "entry " + Math.round(this.thr * 100) + "%");
      ref(1 - this.thr, C.axis, "entry " + Math.round((1 - this.thr) * 100) + "%");
      ref(this.ext, C.warn, "hold " + Math.round(this.ext * 100) + "%");
      ref(1 - this.ext, C.warn, "hold " + Math.round((1 - this.ext) * 100) + "%");
      const ds = ymdIST(hs[0].t_ms);
      for (let hh = 10; hh <= 15; hh++) {
        const t = Date.parse(ds + "T" + String(hh).padStart(2, "0") + ":00:00+05:30");
        if (t < x0 || t > x1) continue;
        const x = X(t); this.line(x, T + H, x, T + H + 4, C.axis); this.text(hh + ":00", x, T + H + 13, C.muted, "center");
      }
      const s1 = hs.map((p) => (isNum(p.p30) ? [X(p.t_ms), Y(p.p30)] : null));
      const s2 = hs.map((p) => (isNum(p.pclose) ? [X(p.t_ms), Y(p.pclose)] : null));
      this.poly(s2, C.s2); this.poly(s1, C.s1);
      const lastOf = (arr) => { for (let i = arr.length - 1; i >= 0; i--) if (arr[i]) return arr[i]; return null; };
      const l2 = lastOf(s2), l1 = lastOf(s1);
      if (l2) this.dot(l2[0], l2[1], C.s2, 4);
      if (l1) this.dot(l1[0], l1[1], C.s1, 4);
      this.pts = hs.map((p) => ({ x: X(p.t_ms), y: isNum(p.p30) ? Y(p.p30) : T + H / 2, p }));
      if (this.hover !== null && this.pts[this.hover]) {
        const pt = this.pts[this.hover];
        this.line(pt.x, T, pt.x, T + H, C.axis);
        if (isNum(pt.p.p30)) this.dot(pt.x, Y(pt.p.p30), C.s1, 4);
        if (isNum(pt.p.pclose)) this.dot(pt.x, Y(pt.p.pclose), C.s2, 4);
      }
    }
    tip(i) {
      const p = this.pts[i] && this.pts[i].p; if (!p) return null;
      return { head: timeIST(p.t_ms) + " IST", rows: [
        { label: "P(up 30 min)", value: prob(p.p30), color: this.colors.s1 },
        { label: "P(up by 15:15)", value: prob(p.pclose), color: this.colors.s2 }] };
    }
  }

  // --------------------------------------------------------------------------- equity chart
  class EquityChart extends Chart {
    constructor(c, w) { super(c, w); this.series = {}; }
    set(series) {
      const out = {};
      for (const k of Object.keys(series || {})) {
        if (!Array.isArray(series[k])) continue;
        out[k] = series[k].filter((p) => p && p.date).map((p) => ({ date: String(p.date).slice(0, 10), equity: p.equity }));
      }
      this.series = out;
      this.requestDraw();
    }
    draw() {
      this.clear();
      const names = Object.keys(this.series).filter((k) => Array.isArray(this.series[k]) && this.series[k].length);
      const dates = [...new Set(names.flatMap((k) => this.series[k].map((p) => String(p.date))))].sort();
      show("eq-empty", !dates.length);
      this.pts = [];
      if (!dates.length) return;
      const C = this.colors;
      const L = 8, R = 70, T = 10, B = 22, W = this.w - L - R, H = this.h - T - B;
      let lo = Infinity, hi = -Infinity;
      for (const k of names) for (const p of this.series[k]) if (isNum(p.equity)) { lo = Math.min(lo, p.equity); hi = Math.max(hi, p.equity); }
      if (!isFinite(lo)) return;
      const pad = Math.max((hi - lo) * 0.08, 500); lo -= pad; hi += pad;
      const X = (i) => L + (dates.length === 1 ? W / 2 : (i / (dates.length - 1)) * W);
      const Y = (v) => T + (1 - (v - lo) / (hi - lo)) * H;
      const lakh = (v) => (Math.abs(v) >= 1e5 ? "₹" + (v / 1e5).toFixed(2) + "L" : "₹" + nf(0).format(v));
      for (const v of niceTicks(lo, hi, 4)) { const y = Y(v); this.line(L, y, L + W, y, C.grid); this.text(lakh(v), L + W + 6, y, C.muted); }
      this.line(L, T + H, L + W, T + H, C.axis);
      const step = Math.max(1, Math.ceil(dates.length / 6));
      dates.forEach((d, i) => {
        if (i % step !== 0 && i !== dates.length - 1) return;
        const align = dates.length > 1 && i === 0 ? "left" : dates.length > 1 && i === dates.length - 1 ? "right" : "center";
        this.text(dateIST(d + "T12:00:00+05:30"), X(i), T + H + 13, C.muted, align);
      });
      const idx = new Map(dates.map((d, i) => [d, i]));
      for (const k of names) {
        const col = cssVar(bookColorVar(k));
        const pts = this.series[k].map((p) => (isNum(p.equity) ? [X(idx.get(String(p.date))), Y(p.equity)] : null));
        if (pts.filter(Boolean).length === 1) { const p = pts.find(Boolean); this.dot(p[0], p[1], col, 4); }
        else this.poly(pts, col);
        const last = pts.filter(Boolean).pop(); if (last) this.dot(last[0], last[1], col, 4);
      }
      this.pts = dates.map((d, i) => ({ x: X(i), y: T + H / 2, d }));
      this.Y = Y;
      if (this.hover !== null && this.pts[this.hover]) {
        const pt = this.pts[this.hover];
        this.line(pt.x, T, pt.x, T + H, C.axis);
        for (const k of names) { const p = this.series[k].find((q) => String(q.date) === pt.d); if (p && isNum(p.equity)) this.dot(pt.x, Y(p.equity), cssVar(bookColorVar(k)), 4); }
      }
    }
    tip(i) {
      const pt = this.pts[i]; if (!pt) return null;
      const rows = [];
      for (const k of Object.keys(this.series)) {
        const p = (this.series[k] || []).find((q) => String(q.date) === pt.d);
        rows.push({ label: k, value: p && isNum(p.equity) ? inr(p.equity) : "—", color: cssVar(bookColorVar(k)) });
      }
      return { head: pt.d, rows };
    }
  }
  function updateEquityChart() {
    const leg = $("eq-legend");
    leg.textContent = "";
    const names = Object.keys(EQUITY || {});
    if (names.length >= 2) for (const k of names) leg.append(h("span", { class: "lg" }, h("i", { class: "ln", style: "background:var(" + bookColorVar(k) + ")" }), k));
    if (CHARTS.eq) CHARTS.eq.set(EQUITY);
  }
  function renderCharts(s) {
    const bars = (Array.isArray(s.bars) ? s.bars : []).filter((b) => b && isNum(b.t_ms) && isNum(b.c)).sort((a, b) => a.t_ms - b.t_ms);
    const sd = sessionDate(s);
    const bounds = sessionBounds(sd);
    const pos = positionsOf(s)[0];
    const priceAt = (t) => {
      let best = null;
      for (const b of bars) { if (b.t_ms <= t) best = b; else break; }
      return best ? best.c : bars.length ? bars[0].c : null;
    };
    const markers = [];
    for (const t of TRADES) {
      if (String(t.date || "").slice(0, 10) !== sd) continue;
      const e = toDate(t.entry_ts), x = toDate(t.exit_ts);
      const leg = tradeLeg(t);
      const lbl = (t.book || "") + " #" + (t.id ?? "") + " " + ((leg && leg.typ) || "");
      if (e) markers.push({ t_ms: e.getTime(), price: priceAt(e.getTime()), kind: "entry", dir: t.direction, label: lbl });
      if (x) markers.push({ t_ms: x.getTime(), price: priceAt(x.getTime()), kind: "exit", dir: t.direction, label: lbl });
    }
    const q = s.quote || {};
    CHARTS.price.set({ bars, bounds, markers, prevClose: isNum(q.prev_close) ? q.prev_close : null,
      strike: pos && isNum(pos.strike) ? { price: pos.strike, label: "strike " + nf(0).format(pos.strike) + " " + (pos.side || "") } : null });
    const p = s.prediction || {}, r = s.rules || {};
    const hist = (Array.isArray(p.history) ? p.history : []).filter((x) => x && isNum(x.t_ms)).sort((a, b) => a.t_ms - b.t_ms);
    CHARTS.pred.set({ hist, bounds, thr: isNum(r.entry_threshold) ? r.entry_threshold : 0.6, ext: isNum(r.extend_prob) ? r.extend_prob : 0.8 });
  }
  function initCharts() {
    CHARTS.price = new PriceChart($("price-chart"), $("price-wrap"));
    CHARTS.pred = new PredChart($("pred-chart"), $("pred-wrap"));
    CHARTS.eq = new EquityChart($("eq-chart"), $("eq-wrap"));
    for (const [id, mode] of [["mode-line", "line"], ["mode-candle", "candle"]]) {
      $(id).addEventListener("click", () => {
        CHARTS.price.mode = mode;
        $("mode-line").classList.toggle("on", mode === "line"); $("mode-line").setAttribute("aria-pressed", String(mode === "line"));
        $("mode-candle").classList.toggle("on", mode === "candle"); $("mode-candle").setAttribute("aria-pressed", String(mode === "candle"));
        try { localStorage.setItem("nifty-chart-mode", mode); } catch (e) { /* ignore */ }
        CHARTS.price.requestDraw();
      });
    }
    try { if (localStorage.getItem("nifty-chart-mode") === "candle") $("mode-candle").click(); } catch (e) { /* ignore */ }
  }

  // =========================================================================== markdown reports
  function mdInline(s) {
    const esc = (t) => t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    const codes = [];
    s = s.replace(/`([^`]+)`/g, (_, c) => { codes.push(c); return "\u0000" + (codes.length - 1) + "\u0000"; });
    s = esc(s);
    s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (m, t, u) => {
      const raw = u.replace(/&amp;/g, "&");
      return /^(https?:\/\/|\/|#)/i.test(raw) ? '<a href="' + u + '"' + (/^https?:/i.test(raw) ? ' target="_blank" rel="noopener noreferrer"' : "") + ">" + t + "</a>" : t;
    });
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/(^|[^*\w])\*([^*\s][^*]*?)\*(?=[^*\w]|$)/g, "$1<em>$2</em>");
    s = s.replace(/(^|[^\w])_([^_\s][^_]*?)_(?=[^\w]|$)/g, "$1<em>$2</em>");
    s = s.replace(/\u0000(\d+)\u0000/g, (_, i) => "<code>" + esc(codes[+i]) + "</code>");
    return s;
  }
  function mdToHtml(src) {
    const lines = String(src || "").replace(/\r\n?/g, "\n").split("\n");
    const out = [];
    let i = 0, para = [];
    const flush = () => { if (para.length) { out.push("<p>" + mdInline(para.join(" ")) + "</p>"); para = []; } };
    const isRow = (l) => /^\s*\|.*\|\s*$/.test(l);
    const cells = (l) => l.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
    while (i < lines.length) {
      const l = lines[i];
      if (/^\s*```/.test(l)) {
        flush(); const buf = []; i++;
        while (i < lines.length && !/^\s*```/.test(lines[i])) buf.push(lines[i++]);
        i++;
        out.push("<pre><code>" + buf.join("\n").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;") + "</code></pre>");
        continue;
      }
      const hm = /^(#{1,6})\s+(.*)$/.exec(l);
      if (hm) { flush(); const n = hm[1].length; out.push("<h" + n + ">" + mdInline(hm[2]) + "</h" + n + ">"); i++; continue; }
      if (/^\s*(-{3,}|\*{3,})\s*$/.test(l)) { flush(); out.push("<hr>"); i++; continue; }
      if (isRow(l) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
        flush();
        const head = cells(l); i += 2;
        let html = "<table><thead><tr>" + head.map((c) => "<th>" + mdInline(c) + "</th>").join("") + "</tr></thead><tbody>";
        while (i < lines.length && isRow(lines[i])) { html += "<tr>" + cells(lines[i]).map((c) => "<td>" + mdInline(c) + "</td>").join("") + "</tr>"; i++; }
        out.push(html + "</tbody></table>");
        continue;
      }
      const lm = /^\s*([-*+]|\d+\.)\s+(.*)$/.exec(l);
      if (lm) {
        flush();
        const ordered = /\d/.test(lm[1]);
        const items = [];
        while (i < lines.length) {
          const m2 = /^\s*([-*+]|\d+\.)\s+(.*)$/.exec(lines[i]);
          if (m2) { items.push(m2[2]); i++; }
          else if (/^\s{2,}\S/.test(lines[i]) && items.length) { items[items.length - 1] += " " + lines[i].trim(); i++; }
          else break;
        }
        const tag = ordered ? "ol" : "ul";
        out.push("<" + tag + ">" + items.map((t) => "<li>" + mdInline(t) + "</li>").join("") + "</" + tag + ">");
        continue;
      }
      if (/^\s*>/.test(l)) {
        flush(); const buf = [];
        while (i < lines.length && /^\s*>/.test(lines[i])) buf.push(lines[i++].replace(/^\s*>\s?/, ""));
        out.push("<blockquote>" + mdInline(buf.join(" ")) + "</blockquote>");
        continue;
      }
      if (!l.trim()) { flush(); i++; continue; }
      para.push(l.trim()); i++;
    }
    flush();
    return out.join("\n");
  }
  function initReports() {
    $("report-btns").addEventListener("click", async (e) => {
      const b = e.target.closest("[data-report]");
      if (!b) return;
      $("report-btns").querySelectorAll(".seg-b").forEach((x) => x.classList.toggle("on", x === b));
      const body = $("report-body");
      body.style.opacity = "0.55";
      try {
        const r = await fetch("/api/report/" + encodeURIComponent(b.dataset.report), { cache: "no-store" });
        const txt = await r.text();
        if (!r.ok) {
          body.textContent = "";
          body.append(h("p", { class: "empty" }, "Report not available (" + r.status + "). " + (txt.length < 300 ? txt : "")));
        } else {
          body.innerHTML = mdToHtml(txt);   // mdToHtml escapes every piece of source text
          glossify(body);
        }
      } catch (err) {
        body.textContent = "";
        body.append(h("p", { class: "empty" }, "Engine not reachable: start it to load reports."));
      }
      body.style.opacity = "";
    });
  }

  // =========================================================================== init
  function tick() {
    setText("clock", TIME_FMT.format(new Date()));
    document.querySelectorAll("[data-until]").forEach(updateCountdown);
  }
  function init() {
    initTheme();
    initTabs();
    initTooltips();
    initNewsFilters();
    initTradeRange();
    initCharts();
    initReports();
    $("btn-buy-ce").addEventListener("click", () => paperOrder("buy_ce", "Buy 1 lot of the ATM call (CE) in the manual book?"));
    $("btn-buy-pe").addEventListener("click", () => paperOrder("buy_pe", "Buy 1 lot of the ATM put (PE) in the manual book?"));
    document.querySelectorAll("[data-term]").forEach((el) => { if (!el.hasAttribute("tabindex")) el.tabIndex = 0; });
    setConn("connecting");
    tick(); setInterval(tick, 1000);
    setInterval(() => { if (LAST) renderNews(LAST); }, 30000);   // refresh "x min ago"
    setInterval(fetchTrades, 20000);
    setInterval(fetchEquity, 60000);
    loadGlossary();
    pollOnce().then(() => startSSE());
    fetchTrades(); fetchEquity(); fetchNews();
  }
  // Exposed for offline tests only (no effect on the page).
  window.__niftyDash = { mdToHtml, mdInline, niceTicks, inr, prob, signed, signedPct, num, dirWord, ageText };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
