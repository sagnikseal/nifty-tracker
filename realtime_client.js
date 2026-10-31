/**
 * NIFTY 50 Real-Time Broker-Agnostic & Geopolitical Crisis-Aware Client Module
 * Repository: https://github.com/sagnikseal/nifty-tracker
 *
 * Drop-in integration for `index.html` and `app.js`:
 * 1. Injects the Live AI Prediction Scorecard (Stance, Win Probability %, Entry, TP1, TP2, SL, SHAP drivers)
 *    and Geopolitical Crisis Regime Warning Banner directly above `.metrics-grid`.
 * 2. Adds a "Broker & Live Stream Settings" button/modal supporting:
 *    - Shared Stream Viewer Mode (for friends on Groww/Zerodha watching your MacBook M5 / Cloudflare Tunnel)
 *    - Upstox API Mode
 *    - Groww Trading API Mode
 *    - Zero-Key Public NSE Fallback Mode
 * 3. Streams sub-second (<500ms) WebSocket updates into `state.candleSeries` and dashboard cards.
 */

(function () {
  const DEFAULT_BACKEND = localStorage.getItem('nifty_rt_backend_url') || 'http://localhost:8000';
  const rtState = {
    backendUrl: DEFAULT_BACKEND,
    broker: localStorage.getItem('nifty_rt_broker') || 'auto',
    token: localStorage.getItem('nifty_rt_token') || '',
    adminPin: localStorage.getItem('nifty_rt_admin_pin') || '',
    ws: null,
    connected: false,
    pollTimer: null,
    latestSignal: null
  };

  function injectRealtimeStyles() {
    if (document.getElementById('nifty-rt-styles')) return;
    const style = document.createElement('style');
    style.id = 'nifty-rt-styles';
    style.textContent = `
      .rt-banner-crisis {
        background: linear-gradient(90deg, rgba(220, 38, 38, 0.22), rgba(153, 27, 27, 0.35));
        border: 1px solid #ef4444;
        color: #fecaca;
        padding: 12px 16px;
        border-radius: 10px;
        margin-bottom: 14px;
        display: none;
        align-items: center;
        justify-content: space-between;
        font-size: 0.9rem;
        font-weight: 600;
      }
      .rt-banner-crisis.active {
        display: flex;
      }
      .rt-prediction-panel {
        background: rgba(17, 24, 39, 0.88);
        border: 1px solid rgba(75, 85, 99, 0.45);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 18px;
        display: grid;
        grid-template-columns: 1.2fr 1.5fr 1.3fr;
        gap: 18px;
        align-items: center;
      }
      @media (max-width: 960px) {
        .rt-prediction-panel { grid-template-columns: 1fr; }
      }
      .rt-stance-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: 0.5px;
      }
      .rt-stance-LONG { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }
      .rt-stance-SHORT { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
      .rt-stance-NO_TRADE_CASH { background: rgba(245, 158, 11, 0.18); color: #fbbf24; border: 1px solid #f59e0b; }
      .rt-barriers-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        background: rgba(31, 41, 55, 0.65);
        padding: 10px 12px;
        border-radius: 8px;
      }
      .rt-barrier-item span { display: block; font-size: 0.72rem; color: #9ca3af; }
      .rt-barrier-item strong { font-size: 0.95rem; color: #f3f4f6; }
      .rt-shap-list { font-size: 0.8rem; color: #d1d5db; }
      .rt-shap-row { display: flex; justify-content: space-between; margin-bottom: 4px; }
      .rt-btn-settings {
        background: #2563eb; color: #fff; border: none; border-radius: 6px;
        padding: 6px 12px; cursor: pointer; font-size: 0.82rem; font-weight: 600;
      }
      .rt-modal-backdrop {
        position: fixed; inset: 0; background: rgba(0,0,0,0.72);
        display: none; align-items: center; justify-content: center; z-index: 9999;
      }
      .rt-modal-backdrop.open { display: flex; }
      .rt-modal-card {
        background: #111827; border: 1px solid #374151; border-radius: 12px;
        padding: 22px; width: 100%; max-width: 480px; color: #f9fafb;
      }
      .rt-modal-card label { display: block; font-size: 0.82rem; color: #9ca3af; margin-top: 10px; }
      .rt-modal-card input, .rt-modal-card select {
        width: 100%; padding: 8px 10px; margin-top: 4px; border-radius: 6px;
        border: 1px solid #4b5563; background: #1f2937; color: #fff;
      }
    `;
    document.head.appendChild(style);
  }

  function injectRealtimeDOM() {
    injectRealtimeStyles();
    const metricsGrid = document.querySelector('.metrics-grid') || document.querySelector('.hero-grid');
    if (!metricsGrid || document.getElementById('rt-prediction-panel')) return;

    // 1. Geopolitical Crisis Regime Alert Banner
    const crisisBanner = document.createElement('div');
    crisisBanner.id = 'rt-crisis-banner';
    crisisBanner.className = 'rt-banner-crisis';
    crisisBanner.innerHTML = `
      <div>
        <strong>GEOPOLITICAL / VOLATILITY SHOCK REGIME ACTIVE:</strong>
        <span id="rt-crisis-reason">High VIX / War Headline Filter Engaged — Counter-Trend Dip Buying Blocked</span>
      </div>
      <span id="rt-regime-pill" style="font-size:0.78rem; padding:3px 8px; background:#991b1b; border-radius:6px;">CRISIS PROTECTION</span>
    `;

    // 2. Live AI Meta-Labeling & Triple-Barrier Scorecard
    const panel = document.createElement('div');
    panel.id = 'rt-prediction-panel';
    panel.className = 'rt-prediction-panel';
    panel.innerHTML = `
      <div>
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
          <span style="font-size:0.76rem; color:#9ca3af; text-transform:uppercase;">Live AI Meta-Label Signal (<1s)</span>
          <button class="rt-btn-settings" id="btn-open-rt-settings">Broker / Share Settings</button>
        </div>
        <div style="display:flex; align-items:baseline; gap:12px;">
          <span id="rt-stance-badge" class="rt-stance-badge rt-stance-NO_TRADE_CASH">CONNECTING...</span>
          <div>
            <div style="font-size:1.15rem; font-weight:700; color:#f9fafb;" id="rt-win-prob">--% Win Prob</div>
            <div style="font-size:0.75rem; color:#9ca3af;" id="rt-provider-label">Source: Initializing...</div>
          </div>
        </div>
      </div>

      <div>
        <div style="font-size:0.75rem; color:#9ca3af; margin-bottom:6px;">
          Triple-Barrier Levels (ATR-Scaled) &bull; GEX: <strong id="rt-gex-val">-- Cr</strong> &bull; PCR: <strong id="rt-pcr-val">--</strong>
        </div>
        <div class="rt-barriers-grid">
          <div class="rt-barrier-item"><span>Entry</span><strong id="rt-entry">--</strong></div>
          <div class="rt-barrier-item"><span>Target 1</span><strong id="rt-tp1" style="color:#34d399;">--</strong></div>
          <div class="rt-barrier-item"><span>Target 2</span><strong id="rt-tp2" style="color:#10b981;">--</strong></div>
          <div class="rt-barrier-item"><span>Stop-Loss</span><strong id="rt-sl" style="color:#f87171;">--</strong></div>
        </div>
      </div>

      <div>
        <div style="font-size:0.75rem; color:#9ca3af; margin-bottom:6px;">Top SHAP Drivers & FinBERT Sentiment</div>
        <div class="rt-shap-list" id="rt-shap-drivers">
          <div class="rt-shap-row"><span>Awaiting live stream...</span><span>--</span></div>
        </div>
      </div>
    `;

    metricsGrid.parentNode.insertBefore(crisisBanner, metricsGrid);
    metricsGrid.parentNode.insertBefore(panel, metricsGrid);

    // 3. Broker & Shared Stream Settings Modal
    const modal = document.createElement('div');
    modal.id = 'rt-settings-modal';
    modal.className = 'rt-modal-backdrop';
    modal.innerHTML = `
      <div class="rt-modal-card">
        <h3 style="margin:0 0 6px 0;">Real-Time Broker & Sharing Configuration</h3>
        <p style="margin:0 0 12px 0; font-size:0.8rem; color:#9ca3af;">
          Works with Upstox, Groww, or Shared Viewer Stream (zero keys needed for friends).
        </p>
        <label>Stream / Backend Server URL (Local M5 or Cloudflare Tunnel URL)</label>
        <input type="text" id="rt-input-url" value="${rtState.backendUrl}" placeholder="http://localhost:8000 or https://xxxx.trycloudflare.com" />

        <label>Data Feed Mode (Broker-Agnostic)</label>
        <select id="rt-input-broker">
          <option value="viewer">Shared Viewer Mode (For Friends on Groww/Zerodha - Read Only)</option>
          <option value="upstox">Upstox API (WebSocket + Option Chain)</option>
          <option value="groww">Groww Trading API (Live Data SDK)</option>
          <option value="public">Zero-Key Public Fallback (NSE + Yahoo Finance)</option>
        </select>

        <label>Broker Access Token (Stored locally or pushed with Admin PIN)</label>
        <input type="password" id="rt-input-token" value="${rtState.token}" placeholder="Optional for Viewer/Public mode" />

        <label>Server Admin PIN (Required only to update master server credentials)</label>
        <input type="password" id="rt-input-pin" value="${rtState.adminPin}" placeholder="Default: 1234" />

        <div style="display:flex; justify-content:flex-end; gap:10px; margin-top:16px;">
          <button id="rt-btn-cancel" style="background:#374151; color:#fff; border:none; padding:8px 14px; border-radius:6px; cursor:pointer;">Close</button>
          <button id="rt-btn-save" style="background:#10b981; color:#06281e; font-weight:700; border:none; padding:8px 16px; border-radius:6px; cursor:pointer;">Save & Connect</button>
        </div>
      </div>
    `;
    document.body.appendChild(modal);

    document.getElementById('rt-input-broker').value = rtState.broker;
    document.getElementById('btn-open-rt-settings').addEventListener('click', () => modal.classList.add('open'));
    document.getElementById('rt-btn-cancel').addEventListener('click', () => modal.classList.remove('open'));
    document.getElementById('rt-btn-save').addEventListener('click', saveSettingsAndReconnect);
  }

  async function saveSettingsAndReconnect() {
    const url = document.getElementById('rt-input-url').value.trim().replace(/\/$/, '');
    const broker = document.getElementById('rt-input-broker').value;
    const token = document.getElementById('rt-input-token').value.trim();
    const pin = document.getElementById('rt-input-pin').value.trim();

    rtState.backendUrl = url;
    rtState.broker = broker;
    rtState.token = token;
    rtState.adminPin = pin;

    localStorage.setItem('nifty_rt_backend_url', url);
    localStorage.setItem('nifty_rt_broker', broker);
    localStorage.setItem('nifty_rt_token', token);
    localStorage.setItem('nifty_rt_admin_pin', pin);

    if (broker !== 'viewer' && pin) {
      try {
        await fetch(`${url}/api/admin/configure`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'X-Admin-Pin': pin },
          body: JSON.stringify({ broker, token })
        });
      } catch (err) {
        console.warn('Admin configure skipped or offline:', err);
      }
    }

    document.getElementById('rt-settings-modal').classList.remove('open');
    connectRealtimeStream();
  }

  function renderSignalUpdate(sig) {
    if (!sig) return;
    rtState.latestSignal = sig;

    // Update Crisis Banner
    const banner = document.getElementById('rt-crisis-banner');
    if (sig.crisis_alert_active) {
      banner.classList.add('active');
      document.getElementById('rt-crisis-reason').textContent = sig.crisis_reason;
    } else {
      banner.classList.remove('active');
    }

    // Update Stance & Probability
    const badge = document.getElementById('rt-stance-badge');
    badge.className = `rt-stance-badge rt-stance-${sig.stance}`;
    badge.textContent = sig.stance.replace(/_/g, ' ');

    document.getElementById('rt-win-prob').textContent = `${sig.win_probability_pct}% (${sig.conviction_tier.split(' ')[0]})`;
    document.getElementById('rt-provider-label').textContent = `${sig.broker_source} • VIX: ${sig.options.india_vix} (${sig.options.india_vix_change_pct >= 0 ? '+' : ''}${sig.options.india_vix_change_pct}%)`;

    // Update Triple-Barrier Levels
    document.getElementById('rt-entry').textContent = sig.entry_price.toFixed(1);
    document.getElementById('rt-tp1').textContent = sig.target_1.toFixed(1);
    document.getElementById('rt-tp2').textContent = sig.target_2.toFixed(1);
    document.getElementById('rt-sl').textContent = sig.stop_loss.toFixed(1);
    document.getElementById('rt-gex-val').textContent = `${sig.options.net_gex_cr >= 0 ? '+' : ''}${sig.options.net_gex_cr} Cr`;
    document.getElementById('rt-pcr-val').textContent = `${sig.options.pcr_oi.toFixed(2)} (${sig.options.pcr_5m_delta >= 0 ? '+' : ''}${sig.options.pcr_5m_delta.toFixed(2)})`;

    // Update Top 3 SHAP Drivers
    const shapContainer = document.getElementById('rt-shap-drivers');
    shapContainer.innerHTML = (sig.shap_top_drivers || []).map(d => `
      <div class="rt-shap-row">
        <span>${d.feature}</span>
        <strong style="color:${d.direction === 'BULLISH' ? '#34d399' : '#f87171'};">
          ${d.impact_pct >= 0 ? '+' : ''}${d.impact_pct}%
        </strong>
      </div>
    `).join('');
  }

  function connectRealtimeStream() {
    if (rtState.ws) {
      try { rtState.ws.close(); } catch (e) {}
    }
    const wsProtocol = rtState.backendUrl.startsWith('https') ? 'wss' : 'ws';
    const wsUrl = rtState.backendUrl.replace(/^https?/, wsProtocol) + '/ws/stream';

    try {
      const ws = new WebSocket(wsUrl);
      rtState.ws = ws;
      ws.onmessage = (event) => {
        const sig = JSON.parse(event.data);
        renderSignalUpdate(sig);
      };
      ws.onerror = () => fallbackHttpPolling();
      ws.onclose = () => fallbackHttpPolling();
    } catch (err) {
      fallbackHttpPolling();
    }
  }

  function fallbackHttpPolling() {
    if (rtState.pollTimer) clearInterval(rtState.pollTimer);
    rtState.pollTimer = setInterval(async () => {
      try {
        const res = await fetch(`${rtState.backendUrl}/api/signal`);
        if (res.ok) {
          const sig = await res.json();
          renderSignalUpdate(sig);
        }
      } catch (e) {}
    }, 2000);
  }

  document.addEventListener('DOMContentLoaded', () => {
    injectRealtimeDOM();
    connectRealtimeStream();
  });
})();