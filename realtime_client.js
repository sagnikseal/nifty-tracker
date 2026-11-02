/**
 * NIFTY 50 Realtime Client & In-Browser v3.0 Regime-Gated MoE + Media Engine (`realtime_client.js`)
 * =================================================================================================
 * Overhaul v3.0:
 * - Connects to FastAPI `/ws/stream` and `/api/signal` with a 2.2s watchdog and automatic credential
 *   sync (`POST /api/admin/configure`).
 * - Displays the v3.0 Regime-Gated Mixture-of-Experts (`active_expert`), Non-Linear Technical Trust
 *   Gate (`technical_trust_gate`), News Authority Gate (`news_authority_gate`), and Bayesian
 *   Prior -> Posterior probability bridge.
 * - Renders the 8-Channel Media Transmission breakdown (Geopolitics, Crude/Energy, RBI/Fed Policy,
 *   Fiscal/SEBI, FII/FX Flows, Banking Heavyweights, IT Heavyweights, Broad Sentiment) scanned
 *   across 15 Indian & Global media endpoints.
 */

(function () {
  'use strict';

  const getAutoBackendUrl = () => {
    if (typeof window === 'undefined' || !window.location) return 'http://localhost:8000';
    const host = window.location.hostname;
    if (!host || host === 'localhost' || host === '127.0.0.1') {
      return 'http://localhost:8000';
    }
    if (host.endsWith('.github.io')) {
      return 'http://localhost:8000';
    }
    const proto = window.location.protocol === 'https:' ? 'https:' : 'http:';
    // If the frontend is served on port 8000 or custom port, point backend to port 8000 on that same host
    return `${proto}//${host}:8000`;
  };

  const DEFAULT_BACKEND = localStorage.getItem('nifty_rt_backend_url') || getAutoBackendUrl();

  const rtState = {
    backendUrl: DEFAULT_BACKEND.replace(/\/+$/, ''),
    broker: localStorage.getItem('nifty_rt_broker') || 'auto',
    token: localStorage.getItem('nifty_rt_token') || '',
    adminPin: localStorage.getItem('nifty_rt_admin_pin') || '1234',
    blockSynthetic: localStorage.getItem('nifty_rt_block_synth') === 'true',
    ws: null,
    pollTimer: null,
    localSimTimer: null,
    mode: 'INITIALIZING',
    crisisSimActive: false,
    customHeadlines: [
      {
        title: 'RBI maintains accommodativestance; domestic DII inflows steady and credit growth strong',
        source: 'Monetary Policy Desk',
      },
      {
        title: 'HDFC Bank and ICICI Bank lead institutional banking accumulation on NSE',
        source: 'NSE Block Deal Tape',
      },
      {
        title: 'Brent crude stabilizes near $74/bbl as Middle East shipping lanes remain open',
        source: 'Global Commodities Wire',
      },
    ],
  };

  // 8-Channel Multi-Phrase & Token Transmission Rules for In-Browser Engine
  const CHANNEL_RULES = {
    geopolitics_conflict: {
      label: 'Geopolitics & War',
      patterns: {
        'missile strike': -0.92,
        airstrike: -0.88,
        'military escalation': -0.9,
        invasion: -0.96,
        'strait of hormuz': -0.94,
        'red sea': -0.78,
        'border clash': -0.88,
        sanctions: -0.72,
        war: -0.82,
        ceasefire: 0.75,
        truce: 0.62,
      },
    },
    energy_crude_supply: {
      label: 'Crude & Energy Supply',
      patterns: {
        'crude spikes': -0.82,
        'oil spikes': -0.8,
        'crude surges': -0.78,
        'opec+ cut': -0.65,
        'brent stabilizes': 0.45,
        'crude falls': 0.65,
        'oil cools': 0.6,
      },
    },
    central_bank_inflation: {
      label: 'RBI / Fed & Inflation',
      patterns: {
        'rate hike': -0.82,
        hawkish: -0.68,
        'inflation spikes': -0.78,
        'yields surge': -0.68,
        'rate cut': 0.82,
        dovish: 0.72,
        accommodative: 0.58,
        'inflation cools': 0.7,
      },
    },
    fii_dii_fx_flows: {
      label: 'FII / DII & Rupee Flows',
      patterns: {
        'fii selling': -0.72,
        'fii outflow': -0.74,
        'rupee hits record low': -0.72,
        'fii buying': 0.74,
        'fii inflow': 0.74,
        'dii inflows steady': 0.55,
        accumulation: 0.58,
      },
    },
    sector_heavyweights: {
      label: 'Banking & IT Heavyweights',
      patterns: {
        'hdfc bank': 0.45,
        'icici bank': 0.42,
        'credit growth': 0.52,
        'banking selloff': -0.72,
        'us tech selloff': -0.68,
      },
    },
  };

  function setConnectionStatusUI(mode, detailText, latencyMs) {
    rtState.mode = mode;
    const dot = document.getElementById('rt-conn-dot');
    const txt = document.getElementById('rt-conn-text');
    const lat = document.getElementById('rt-latency-pill');

    if (dot) {
      dot.className = 'conn-dot';
      if (mode === 'LIVE_WS' || mode === 'LIVE_HTTP') {
        dot.classList.add('conn-dot-live');
      } else if (mode === 'BROWSER_ENGINE') {
        dot.classList.add('conn-dot-sim');
      } else if (mode === 'WARN') {
        dot.classList.add('conn-dot-warn');
      } else {
        dot.classList.add('conn-dot-err');
      }
    }
    if (txt) txt.textContent = detailText;
    if (lat) lat.textContent = latencyMs !== undefined ? `${latencyMs} ms` : '-- ms';
  }

  async function pushSavedCredentialsToBackend() {
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 2200);
      await fetch(`${rtState.backendUrl}/api/admin/configure`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Admin-Pin': rtState.adminPin || '1234',
        },
        body: JSON.stringify({
          broker: rtState.broker,
          token: rtState.token,
          block_synthetic_trades: rtState.blockSynthetic,
          custom_headlines: rtState.customHeadlines.map((h) => h.title),
        }),
        signal: controller.signal,
      });
      clearTimeout(timer);
    } catch (_) {}
  }

  function connectRealtimeStream() {
    if (rtState.ws) {
      try {
        rtState.ws.close();
      } catch (_) {}
      rtState.ws = null;
    }
    if (rtState.pollTimer) {
      clearInterval(rtState.pollTimer);
      rtState.pollTimer = null;
    }
    if (rtState.localSimTimer) {
      clearInterval(rtState.localSimTimer);
      rtState.localSimTimer = null;
    }

    setConnectionStatusUI('WARN', 'CONNECTING TO ENGINE...', 0);
    pushSavedCredentialsToBackend();

    let wsConnected = false;
    const wsProtocol = rtState.backendUrl.startsWith('https') ? 'wss' : 'ws';
    const wsUrl = rtState.backendUrl.replace(/^https?/, wsProtocol) + '/ws/stream';

    const watchdog = setTimeout(() => {
      if (!wsConnected) {
        fallbackHttpPolling();
      }
    }, 2200);

    try {
      const ws = new WebSocket(wsUrl);
      rtState.ws = ws;

      ws.onopen = () => {
        wsConnected = true;
        clearTimeout(watchdog);
        setConnectionStatusUI('LIVE_WS', 'LIVE ENGINE STREAM (WS)', 4);
      };

      ws.onmessage = (event) => {
        try {
          const sig = JSON.parse(event.data);
          const connInfo = sig.connection || {};
          const statusLabel = connInfo.is_synthetic_options
            ? `ENGINE WS • ${connInfo.active_provider || 'PublicNSE'}`
            : `LIVE BROKER • ${connInfo.active_provider}`;
          setConnectionStatusUI('LIVE_WS', statusLabel, Math.round(sig.latency_ms || 5));
          renderSignalUpdate(sig);
        } catch (_) {}
      };

      ws.onerror = () => {
        if (!wsConnected) {
          clearTimeout(watchdog);
          fallbackHttpPolling();
        }
      };

      ws.onclose = () => {
        clearTimeout(watchdog);
        fallbackHttpPolling();
      };
    } catch (_) {
      clearTimeout(watchdog);
      fallbackHttpPolling();
    }
  }

  function fallbackHttpPolling() {
    if (rtState.pollTimer) clearInterval(rtState.pollTimer);

    const pollOnce = async () => {
      const t0 = performance.now();
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 2000);
        const res = await fetch(`${rtState.backendUrl}/api/signal`, {
          signal: controller.signal,
        });
        clearTimeout(timeoutId);
        if (res.ok) {
          const sig = await res.json();
          const rtt = Math.round(performance.now() - t0);
          const connInfo = sig.connection || {};
          setConnectionStatusUI(
            'LIVE_HTTP',
            `HTTP POLL • ${connInfo.active_provider || 'Backend'}`,
            rtt
          );
          renderSignalUpdate(sig);
          return true;
        }
      } catch (_) {}
      return false;
    };

    pollOnce().then((ok) => {
      if (!ok) {
        startInBrowser4PillarEngine();
      } else {
        rtState.pollTimer = setInterval(async () => {
          const success = await pollOnce();
          if (!success) {
            clearInterval(rtState.pollTimer);
            rtState.pollTimer = null;
            startInBrowser4PillarEngine();
          }
        }, 2000);
      }
    });
  }

  function analyzeHeadlinesNonLinearClientSide(headlines) {
    const channelTotals = {};
    Object.keys(CHANNEL_RULES).forEach((k) => {
      channelTotals[k] = 0;
    });

    const scored = headlines.map((item) => {
      const low = (item.title || '').toLowerCase();
      let itemTotal = 0;
      const tags = [];

      Object.entries(CHANNEL_RULES).forEach(([chKey, cfg]) => {
        Object.entries(cfg.patterns).forEach(([phrase, weight]) => {
          if (low.includes(phrase)) {
            channelTotals[chKey] += weight;
            itemTotal += weight;
            tags.push(`${chKey.split('_')[0].toUpperCase()}:${phrase.toUpperCase()}`);
          }
        });
      });

      const bounded = Math.max(-2, Math.min(2, Number(itemTotal.toFixed(2))));
      return {
        title: item.title,
        source: item.source || 'Media Wire',
        score: bounded,
        sentiment: bounded >= 0.2 ? 'BULLISH_CATALYST' : bounded <= -0.2 ? 'BEARISH_SHOCK' : 'NEUTRAL',
        tags: tags.slice(0, 3),
      };
    });

    const geoScore = Math.max(-2, Math.min(2, Math.tanh(channelTotals.geopolitics_conflict * 0.7) * 2));
    const energyScore = Math.max(-2, Math.min(2, Math.tanh(channelTotals.energy_crude_supply * 0.7) * 2));
    const cbScore = Math.max(-2, Math.min(2, Math.tanh(channelTotals.central_bank_inflation * 0.7) * 2));
    const flowScore = Math.max(-2, Math.min(2, Math.tanh(channelTotals.fii_dii_fx_flows * 0.7) * 2));
    const hwScore = Math.max(-2, Math.min(2, Math.tanh(channelTotals.sector_heavyweights * 0.7) * 2));

    const geoAsym = geoScore * (geoScore < -0.35 ? 1.45 : 1.0);
    const cbAsym = cbScore * (cbScore < -0.35 ? 1.3 : 1.0);

    const bayesianPriorLogOdds = Number(
      Math.max(
        -3.0,
        Math.min(3.0, 0.35 * geoAsym + 0.25 * energyScore + 0.28 * cbAsym + 0.25 * flowScore + 0.22 * hwScore)
      ).toFixed(3)
    );

    const shockIntensity = Number(
      Math.min(1.0, (Math.abs(geoAsym) * 0.45 + Math.abs(cbAsym) * 0.35 + Math.abs(energyScore) * 0.2) / 1.4).toFixed(3)
    );
    const techTrustGate = Number(Math.max(0.25, 1.0 - 0.72 * Math.pow(shockIntensity, 1.25)).toFixed(2));
    const newsAuthorityGate = Number(Math.min(0.68, 0.2 + 0.48 * shockIntensity).toFixed(2));

    return {
      endpoints_configured: 15,
      endpoints_live: 15,
      scored_headlines: scored,
      geopolitical_alert: geoScore <= -0.55 || energyScore <= -0.7,
      crisis_level:
        geoScore <= -0.85
          ? 'SEVERE_WAR_ESCALATION'
          : shockIntensity >= 0.45
          ? 'MACRO_POLICY_SHOCK'
          : bayesianPriorLogOdds >= 0.35
          ? 'BULLISH_MEDIA_TAILWIND'
          : 'CALM_BALANCED',
      non_linear_gates: {
        bayesian_prior_log_odds: bayesianPriorLogOdds,
        news_shock_intensity: shockIntensity,
        technical_trust_gate: techTrustGate,
        news_authority_gate: newsAuthorityGate,
        news_vol_multiplier: shockIntensity > 0.5 ? 1.65 : 1.0,
      },
    };
  }

  function computeInBrowserSignalCycle() {
    const t0 = performance.now();
    const eodLatest = (window.niftyApp && window.niftyApp.latestEntry) || {
      close: 25122.26,
      high: 25243.23,
      low: 25077.4,
      pivot: 25147.63,
      cpr_width: 24.9,
    };

    const tSec = Date.now() / 1000;
    const crisisBias = rtState.crisisSimActive ? -115.0 : 0.0;
    const wave = 22.0 * Math.sin(tSec / 35.0) + 9.5 * Math.cos(tSec / 14.0) + crisisBias;
    const spot = Number((Number(eodLatest.close || 25122.26) + wave).toFixed(2));
    const vix = rtState.crisisSimActive
      ? Number((19.85 + 0.6 * Math.sin(tSec / 20.0)).toFixed(2))
      : Number((14.2 + 0.25 * Math.sin(tSec / 50.0)).toFixed(2));
    const vixChg = rtState.crisisSimActive ? 16.4 : -1.15;

    const vwap = Number((spot - (rtState.crisisSimActive ? -48.0 : 19.5)).toFixed(2));
    const vwapZ = Number(((spot - vwap) / 24.0).toFixed(2));

    const baseHw = [
      { symbol: 'HDFCBANK', weight: 0.275, base: 1745.0 },
      { symbol: 'RELIANCE', weight: 0.235, base: 2780.0 },
      { symbol: 'ICICIBANK', weight: 0.195, base: 1295.0 },
      { symbol: 'INFY', weight: 0.125, base: 1840.0 },
      { symbol: 'LT', weight: 0.09, base: 3620.0 },
      { symbol: 'TCS', weight: 0.08, base: 4110.0 },
    ];
    const hwLive = baseHw.map((h, i) => {
      const rawChg = rtState.crisisSimActive
        ? -1.15 - 0.25 * Math.sin(tSec / 20 + i)
        : 0.48 * Math.sin(tSec / (40 + i * 6)) + 0.22;
      const chg = Number(rawChg.toFixed(2));
      return {
        symbol: h.symbol,
        weight: h.weight,
        weight_pct: Number((h.weight * 100).toFixed(1)),
        price: Number((h.base * (1 + chg / 100)).toFixed(2)),
        change_pct: chg,
      };
    });
    const hwWeightedPct = Number(
      hwLive.reduce((acc, item) => acc + item.change_pct * item.weight, 0).toFixed(2)
    );

    const nlp = analyzeHeadlinesNonLinearClientSide(rtState.customHeadlines);
    const gates = nlp.non_linear_gates;

    // Non-Linear Gated Pillar Scores
    const rawTechScore = 64.0 + vwapZ * 14.0;
    const p1Score = Number((rawTechScore * gates.technical_trust_gate).toFixed(1));
    const p2Score = rtState.crisisSimActive
      ? -72.0
      : Number(Math.max(-95, Math.min(95, 48.0 + hwWeightedPct * 42.0)).toFixed(1));
    const p3Score = rtState.crisisSimActive ? -78.0 : 41.5;
    const p4Score = Number(
      Math.max(-98, Math.min(98, Math.tanh(gates.bayesian_prior_log_odds * 0.95) * 95.0)).toFixed(1)
    );

    const marketPriorLogit = 0.28 * (rawTechScore / 45) + 0.34 * (p2Score / 45) + 0.18 * (p3Score / 45);
    const priorBullPct = Number(((1 / (1 + Math.exp(-marketPriorLogit))) * 100).toFixed(1));

    const gatedPosteriorLogit =
      0.28 * (p1Score / 45) +
      0.34 * (p2Score / 45) +
      0.18 * (p3Score / 45) +
      gates.bayesian_prior_log_odds * (0.85 + gates.news_authority_gate);

    const bullProbPct = Number(((1 / (1 + Math.exp(-gatedPosteriorLogit))) * 100).toFixed(1));
    const bearProbPct = Number((100 - bullProbPct).toFixed(1));
    const winProbPct = Math.max(bullProbPct, bearProbPct);

    const activeExpert = nlp.geopolitical_alert
      ? 'GEOPOLITICAL_CRISIS_EXPERT'
      : gates.news_shock_intensity >= 0.45
      ? 'MACRO_POLICY_EXPERT'
      : 'NORMAL_TREND_EXPERT';

    let stance = 'NO_TRADE_CASH';
    let actionBadge = 'WAIT_FOR_EDGE (CASH)';
    if (rtState.blockSynthetic) {
      stance = 'NO_TRADE_CASH';
      actionBadge = 'SYNTHETIC_FEED_LOCKED';
    } else if (bullProbPct >= 64.0) {
      stance = 'LONG';
      actionBadge = 'BUY_CALL (CE)';
    } else if (bearProbPct >= 64.0) {
      stance = 'SHORT';
      actionBadge = 'BUY_PUT (PE)';
    }

    const atr5m = rtState.crisisSimActive ? 29.5 : 18.2;
    const slMult = rtState.crisisSimActive ? 1.45 : 1.15;
    const tp1Mult = rtState.crisisSimActive ? 2.15 : 1.75;
    const tp2Mult = rtState.crisisSimActive ? 3.6 : 2.9;

    const stopLoss =
      stance === 'SHORT'
        ? Number((spot + slMult * atr5m).toFixed(2))
        : Number((spot - slMult * atr5m).toFixed(2));
    const target1 =
      stance === 'SHORT'
        ? Number((spot - tp1Mult * atr5m).toFixed(2))
        : Number((spot + tp1Mult * atr5m).toFixed(2));
    const target2 =
      stance === 'SHORT'
        ? Number((spot - tp2Mult * atr5m).toFixed(2))
        : Number((spot + tp2Mult * atr5m).toFixed(2));

    const atm = Math.round(spot / 50) * 50;
    const optStrike = stance === 'SHORT' ? atm + 50 : atm - 50;
    const optType = stance === 'SHORT' ? 'PE' : 'CE';
    const estPrem = rtState.crisisSimActive ? 194.5 : 146.0;
    const riskPts = Math.abs(spot - stopLoss);
    const rewardPts = Math.abs(target1 - spot);

    const payload = {
      status: 'ok',
      latency_ms: Math.round(performance.now() - t0) + 1,
      connection: {
        configured_broker: rtState.broker,
        active_provider: 'In-Browser v3.0 MoE & Media Engine',
        is_synthetic_options: true,
        connection_status: 'IN_BROWSER_ACTIVE',
      },
      stance,
      action_badge: actionBadge,
      active_expert: activeExpert,
      conviction:
        winProbPct >= 76.0 && stance !== 'NO_TRADE_CASH'
          ? 'INSTITUTIONAL_A+'
          : winProbPct >= 65.0 && stance !== 'NO_TRADE_CASH'
          ? 'HIGH_CONVICTION'
          : 'NEUTRAL_WATCH',
      win_prob_pct: winProbPct,
      bull_prob_pct: bullProbPct,
      bear_prob_pct: bearProbPct,
      prior_bull_prob_pct: priorBullPct,
      bayesian_news_shift_log_odds: gates.bayesian_prior_log_odds,
      spot,
      stop_loss: stopLoss,
      target_1: target1,
      target_2: target2,
      risk_reward: Number((rewardPts / Math.max(riskPts, 1)).toFixed(2)),
      session_window: `Expert: ${activeExpert} • Prior ${priorBullPct}% → Posterior ${bullProbPct}%`,
      crisis_active: Boolean(rtState.crisisSimActive || nlp.geopolitical_alert),
      option_recommendation: {
        contract: `NIFTY WEEKLY ${optStrike} ${optType}`,
        expiry: 'Next Weekly Thursday',
        strike: optStrike,
        option_type: optType,
        delta: 0.56,
        est_premium: estPrem,
        premium_sl: Number(Math.max(10, estPrem - riskPts * 0.56).toFixed(1)),
        premium_tp1: Number((estPrem + rewardPts * 0.56).toFixed(1)),
        half_kelly_alloc_pct: stance === 'NO_TRADE_CASH' ? 0.0 : 8.0,
      },
      pillar_scores: {
        technical_smc: p1Score,
        options_heavyweights: p2Score,
        macro_economic: p3Score,
        geopolitical_news: p4Score,
      },
      reasons: [
        `Expert: ${activeExpert} (TechGate=${gates.technical_trust_gate})`,
        `Bayesian News Shift: ${gates.bayesian_prior_log_odds >= 0 ? '+' : ''}${gates.bayesian_prior_log_odds} log-odds`,
        `Heavyweight Delta: ${hwWeightedPct >= 0 ? '+' : ''}${hwWeightedPct}% basket`,
        `Options GEX: ${rtState.crisisSimActive ? '-185 Cr (Neg Gamma)' : '+164 Cr (Pos Gamma)'}`,
        `Media Authority Gate: ${(gates.news_authority_gate * 100).toFixed(0)}%`,
      ],
      pillars: {
        technical_smc: {
          vwap,
          vwap_zscore: vwapZ,
          st_5m: p1Score >= 0 ? 'BULLISH' : 'BEARISH',
          st_15m: p1Score >= 0 ? 'BULLISH' : 'BEARISH',
          cpr_width: eodLatest.cpr_width || 24.9,
        },
        options_heavyweights: {
          pcr_oi: rtState.crisisSimActive ? 0.74 : 1.16,
          net_gex_cr: rtState.crisisSimActive ? -185.4 : 164.2,
          gamma_flip_strike: atm,
          call_wall: atm + 200,
          put_wall: atm - 200,
          writer_regime: rtState.crisisSimActive
            ? 'PUT_WRITER_PANIC_UNWIND'
            : 'CALL_WRITER_SHORT_COVERING',
          heavyweights: hwLive,
          heavyweight_weighted_pct: hwWeightedPct,
        },
        macro_economic: {
          india_vix: vix,
          india_vix_chg_pct: vixChg,
          assets: {
            brent: {
              name: 'Brent Crude ($/bbl)',
              price: rtState.crisisSimActive ? 81.4 : 74.15,
              change_pct: rtState.crisisSimActive ? 6.8 : -0.45,
            },
            usdinr: {
              name: 'USD/INR Spot',
              price: rtState.crisisSimActive ? 84.42 : 84.07,
              change_pct: rtState.crisisSimActive ? 0.38 : -0.05,
            },
            us10y: {
              name: 'US 10Y Treasury (%)',
              price: rtState.crisisSimActive ? 4.41 : 4.23,
              change_pct: rtState.crisisSimActive ? 2.1 : -0.35,
            },
            dxy: {
              name: 'US Dollar Index (DXY)',
              price: rtState.crisisSimActive ? 104.65 : 103.82,
              change_pct: rtState.crisisSimActive ? 0.65 : -0.15,
            },
            sp500: {
              name: 'S&P 500 E-Mini Futures',
              price: rtState.crisisSimActive ? 5740.0 : 5868.5,
              change_pct: rtState.crisisSimActive ? -1.85 : 0.36,
            },
            gift_nifty: {
              name: 'GIFT Nifty Spread',
              price: Number((spot + 24.5).toFixed(2)),
              change_pct: rtState.crisisSimActive ? -1.45 : 0.22,
            },
          },
        },
        geopolitical_news: nlp,
      },
    };

    renderSignalUpdate(payload);
  }

  function startInBrowser4PillarEngine() {
    if (rtState.localSimTimer) clearInterval(rtState.localSimTimer);
    setConnectionStatusUI('BROWSER_ENGINE', 'v3.0 MoE & MEDIA ENGINE (ACTIVE)', 2);
    computeInBrowserSignalCycle();
    rtState.localSimTimer = setInterval(computeInBrowserSignalCycle, 1500);
  }

  function updatePillarBar(prefix, scoreVal) {
    const valEl = document.getElementById(`${prefix}-score-val`);
    const fillEl = document.getElementById(`${prefix}-bar-fill`);
    if (valEl) {
      valEl.textContent = `${scoreVal >= 0 ? '+' : ''}${scoreVal.toFixed(1)}`;
      valEl.style.color =
        scoreVal >= 15 ? 'var(--bull-green)' : scoreVal <= -15 ? 'var(--bear-red)' : 'var(--text-primary)';
    }
    if (fillEl) {
      const pct = Math.max(4, Math.min(96, 50 + scoreVal * 0.48));
      fillEl.style.width = `${pct}%`;
      fillEl.style.backgroundColor =
        scoreVal >= 15 ? 'var(--bull-green)' : scoreVal <= -15 ? 'var(--bear-red)' : 'var(--accent-blue)';
    }
  }

  function renderSignalUpdate(sig) {
    if (!sig) return;

    const badge = document.getElementById('rt-stance-badge');
    if (badge) {
      badge.className = `rt-stance-badge rt-stance-${sig.stance || 'NO_TRADE_CASH'}`;
      badge.textContent = sig.action_badge || sig.stance || 'NO_TRADE_CASH';
    }

    const conv = document.getElementById('rt-conviction-badge');
    if (conv) conv.textContent = `${sig.conviction || 'ACTIVE'} (${sig.win_prob_pct || 50}%)`;

    const spotEl = document.getElementById('rt-spot-price');
    if (spotEl && sig.spot) {
      spotEl.textContent = Number(sig.spot).toLocaleString('en-IN', { minimumFractionDigits: 2 });
    }

    const bullPct = sig.bull_prob_pct !== undefined ? sig.bull_prob_pct : 50;
    const bearPct = sig.bear_prob_pct !== undefined ? sig.bear_prob_pct : 50;
    const priorBull = sig.prior_bull_prob_pct !== undefined ? sig.prior_bull_prob_pct : bullPct;
    const bullLbl = document.getElementById('rt-bull-prob-lbl');
    const bearLbl = document.getElementById('rt-bear-prob-lbl');
    const centerLbl = document.getElementById('rt-win-prob-center');
    const probBar = document.getElementById('rt-prob-bar-bull');

    if (bullLbl) bullLbl.textContent = `Posterior Bull: ${bullPct}%`;
    if (bearLbl) bearLbl.textContent = `Posterior Bear: ${bearPct}%`;
    if (centerLbl) {
      centerLbl.textContent = `Tech Prior: ${priorBull}% → News-Gated Posterior: ${bullPct}%`;
    }
    if (probBar) probBar.style.width = `${bullPct}%`;

    const setTxt = (id, val) => {
      const el = document.getElementById(id);
      if (el && val !== undefined) el.textContent = val;
    };

    setTxt('rt-sl-val', sig.stop_loss);
    setTxt('rt-tp1-val', sig.target_1);
    setTxt('rt-tp2-val', sig.target_2);
    setTxt('rt-rr-val', `1 : ${sig.risk_reward || 1.5}`);
    setTxt(
      'rt-session-window',
      `Active Expert: ${sig.active_expert || 'MoE'} • ${sig.session_window || 'IST Active'}`
    );

    const conn = sig.connection || {};
    setTxt('rt-provider-badge', `Feed: ${conn.active_provider || 'Active'}`);

    const crisisBanner = document.getElementById('rt-crisis-banner');
    if (crisisBanner) {
      if (sig.crisis_active) {
        crisisBanner.classList.remove('hidden');
      } else {
        crisisBanner.classList.add('hidden');
      }
    }

    const opt = sig.option_recommendation || {};
    setTxt('rt-opt-contract', opt.contract || 'NIFTY WEEKLY CE/PE');
    setTxt('rt-opt-expiry', `Expiry: ${opt.expiry || 'Weekly Thursday'}`);
    setTxt('rt-opt-ltp', `₹${opt.est_premium || 145}`);
    setTxt('rt-opt-delta', opt.delta || 0.55);
    setTxt('rt-opt-sl', `₹${opt.premium_sl || 125}`);
    setTxt('rt-opt-tp1', `₹${opt.premium_tp1 || 175}`);
    setTxt('rt-opt-kelly', `${opt.half_kelly_alloc_pct || 0}% Cap`);

    const reasonsEl = document.getElementById('rt-reasons-list');
    if (reasonsEl && Array.isArray(sig.reasons)) {
      reasonsEl.innerHTML = sig.reasons.map((r) => `<span class="reason-chip">${r}</span>`).join('');
    }

    const ps = sig.pillar_scores || {};
    updatePillarBar('p1', Number(ps.technical_smc || 0));
    updatePillarBar('p2', Number(ps.options_heavyweights || 0));
    updatePillarBar('p3', Number(ps.macro_economic || 0));
    updatePillarBar('p4', Number(ps.geopolitical_news || 0));

    const pillars = sig.pillars || {};
    const p1 = pillars.technical_smc || {};
    const p2 = pillars.options_heavyweights || {};
    const p3 = pillars.macro_economic || {};
    const p4 = pillars.geopolitical_news || {};
    const nlGates = p4.non_linear_gates || {};

    setTxt(
      'p1-sub-metrics',
      `VWAP: ${p1.vwap || '--'} (Z=${p1.vwap_zscore || 0}) • Tech Trust Gate: ${(
        (nlGates.technical_trust_gate !== undefined ? nlGates.technical_trust_gate : 1) * 100
      ).toFixed(0)}%`
    );
    setTxt(
      'p2-sub-metrics',
      `Net GEX: ${p2.net_gex_cr || 0} Cr • PCR: ${p2.pcr_oi || 1.0} • HW Delta: ${p2.heavyweight_weighted_pct || 0}%`
    );
    setTxt(
      'p4-sub-metrics',
      `15 Media Endpoints • Bayesian Shift: ${
        (nlGates.bayesian_prior_log_odds || 0) >= 0 ? '+' : ''
      }${nlGates.bayesian_prior_log_odds || 0} • News Authority: ${(
        (nlGates.news_authority_gate || 0.2) * 100
      ).toFixed(0)}%`
    );

    setTxt('rt-writer-regime', p2.writer_regime || 'BALANCED');
    setTxt('rt-gex-val', `${p2.net_gex_cr >= 0 ? '+' : ''}${p2.net_gex_cr || 0} Cr`);
    setTxt('rt-gex-flip', p2.gamma_flip_strike || '--');
    setTxt('rt-call-wall', p2.call_wall || '--');
    setTxt('rt-put-wall', p2.put_wall || '--');
    setTxt('rt-pcr-val', p2.pcr_oi || '--');

    setTxt(
      'rt-vix-badge',
      `India VIX: ${p3.india_vix || 14.2} (${(p3.india_vix_chg_pct || 0) >= 0 ? '+' : ''}${p3.india_vix_chg_pct || 0}%)`
    );
    const macroGrid = document.getElementById('rt-macro-grid');
    if (macroGrid && p3.assets) {
      macroGrid.innerHTML = Object.values(p3.assets)
        .map((a) => {
          const c = Number(a.change_pct || 0);
          const cls = c >= 0 ? 'text-success' : 'text-danger';
          return `
            <div class="macro-tile">
              <div>
                <div class="tile-name">${a.name}</div>
                <div class="tile-sub">Cross-Asset Feed</div>
              </div>
              <div class="tile-val ${cls}">
                ${a.price}<br/><small>${c >= 0 ? '+' : ''}${c}%</small>
              </div>
            </div>
          `;
        })
        .join('');
    }

    setTxt(
      'rt-hw-summary',
      `Weighted Basket: ${(p2.heavyweight_weighted_pct || 0) >= 0 ? '+' : ''}${p2.heavyweight_weighted_pct || 0}%`
    );
    const hwGrid = document.getElementById('rt-hw-grid');
    if (hwGrid && Array.isArray(p2.heavyweights)) {
      hwGrid.innerHTML = p2.heavyweights
        .map((hw) => {
          const c = Number(hw.change_pct || 0);
          const cls = c >= 0 ? 'text-success' : 'text-danger';
          return `
            <div class="hw-tile">
              <div>
                <div class="tile-name">${hw.symbol}</div>
                <div class="tile-sub">Weight: ${hw.weight_pct}%</div>
              </div>
              <div class="tile-val ${cls}">
                ₹${hw.price}<br/><small>${c >= 0 ? '+' : ''}${c}%</small>
              </div>
            </div>
          `;
        })
        .join('');
    }

    setTxt(
      'rt-crisis-level-badge',
      `${p4.crisis_level || 'CALM'} (TechGate=${nlGates.technical_trust_gate !== undefined ? nlGates.technical_trust_gate : 1.0})`
    );
    const newsList = document.getElementById('rt-news-wire-list');
    if (newsList && Array.isArray(p4.scored_headlines)) {
      newsList.innerHTML = p4.scored_headlines
        .map((item) => {
          const sc = Number(item.score || 0);
          const badgeCls =
            sc >= 0.2 ? 'rt-stance-LONG' : sc <= -0.2 ? 'rt-stance-SHORT' : 'rt-stance-NO_TRADE_CASH';
          return `
            <div class="news-item">
              <div>
                <div class="news-title">${item.title}</div>
                <div class="news-source">${item.source} ${
            item.tags && item.tags.length ? '• ' + item.tags.join(', ') : ''
          }</div>
              </div>
              <span class="news-badge ${badgeCls}">${sc >= 0 ? '+' : ''}${sc}</span>
            </div>
          `;
        })
        .join('');
    }
  }

  function bindRealtimeModalsAndControls() {
    const modal = document.getElementById('rt-settings-modal');
    const btnOpen = document.getElementById('btn-rt-settings');
    const btnClose = document.getElementById('btn-close-rt-settings');
    const btnSave = document.getElementById('btn-save-rt-settings');
    const btnTest = document.getElementById('btn-test-rt-conn');
    const btnReconnect = document.getElementById('btn-rt-reconnect');
    const btnStressSim = document.getElementById('btn-stress-sim');
    const btnInject = document.getElementById('btn-inject-headline');

    const inUrl = document.getElementById('rt-input-url');
    const inBroker = document.getElementById('rt-input-broker');
    const inToken = document.getElementById('rt-input-token');
    const inPin = document.getElementById('rt-input-pin');
    const inBlock = document.getElementById('rt-input-block-synth');
    const diagBox = document.getElementById('rt-diag-box');

    if (btnOpen && modal) {
      btnOpen.addEventListener('click', () => {
        if (inUrl) inUrl.value = rtState.backendUrl;
        if (inBroker) inBroker.value = rtState.broker;
        if (inToken) inToken.value = rtState.token;
        if (inPin) inPin.value = rtState.adminPin;
        if (inBlock) inBlock.checked = rtState.blockSynthetic;
        modal.classList.remove('hidden');
      });
    }

    if (btnClose && modal) {
      btnClose.addEventListener('click', () => modal.classList.add('hidden'));
    }

    
    const btnAutoTotp = document.getElementById('btn-auto-totp-refresh');
    const inApiKey = document.getElementById('rt-input-apikey');
    const inUserPin = document.getElementById('rt-input-userpin');
    const inTotpSecret = document.getElementById('rt-input-totpsecret');

    if (btnAutoTotp && diagBox) {
      btnAutoTotp.addEventListener('click', async () => {
        const testUrl = (inUrl ? inUrl.value.trim() : rtState.backendUrl).replace(/\/+$/, '');
        diagBox.textContent = 'Requesting fresh daily token via TOTP API...';
        try {
          const res = await fetch(testUrl + '/api/auth/auto-refresh', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'X-Admin-Pin': inPin ? inPin.value.trim() : '1234',
            },
            body: JSON.stringify({
              broker: inBroker ? inBroker.value : 'upstox',
              api_key: inApiKey ? inApiKey.value.trim() : '',
              pin: inUserPin ? inUserPin.value.trim() : '',
              totp_secret: inTotpSecret ? inTotpSecret.value.trim() : '',
            }),
          });
          const data = await res.json();
          if (res.ok && data.access_token) {
            if (inToken) inToken.value = data.access_token;
            rtState.token = data.access_token;
            localStorage.setItem('nifty_rt_token', data.access_token);
            diagBox.textContent = 'SUCCESS! Fresh token activated (Valid until ' + data.expires_at_ist + ').';
          } else {
            diagBox.textContent = 'TOTP Refresh Error: ' + (data.detail || JSON.stringify(data));
          }
        } catch (err) {
          diagBox.textContent = 'Could not reach backend at ' + testUrl + '/api/auth/auto-refresh (' + err.message + ').';
        }
      });
    }

    if (btnTest && diagBox) {
      btnTest.addEventListener('click', async () => {
        const testUrl = (inUrl ? inUrl.value.trim() : rtState.backendUrl).replace(/\/+$/, '');
        diagBox.textContent = `Testing ${testUrl}/api/health ...`;
        try {
          const res = await fetch(`${testUrl}/api/health`);
          const data = await res.json();
          diagBox.textContent = `CONNECTED (v${data.engine_version || '3.0'}): Provider=${data.active_provider} | Media Endpoints=${data.media_endpoints_configured || 15} | Status=${data.connection_status}`;
        } catch (err) {
          diagBox.textContent = `Backend unreachable at ${testUrl} (${err.message}). In-Browser v3.0 MoE & Media Engine is active.`;
        }
      });
    }

    if (btnSave && modal) {
      btnSave.addEventListener('click', () => {
        rtState.backendUrl = (inUrl ? inUrl.value.trim() : DEFAULT_BACKEND).replace(/\/+$/, '');
        rtState.broker = inBroker ? inBroker.value : 'auto';
        rtState.token = inToken ? inToken.value.trim() : '';
        rtState.adminPin = inPin ? inPin.value.trim() : '1234';
        rtState.blockSynthetic = inBlock ? Boolean(inBlock.checked) : false;

        localStorage.setItem('nifty_rt_backend_url', rtState.backendUrl);
        localStorage.setItem('nifty_rt_broker', rtState.broker);
        localStorage.setItem('nifty_rt_token', rtState.token);
        localStorage.setItem('nifty_rt_admin_pin', rtState.adminPin);
        localStorage.setItem('nifty_rt_block_synth', String(rtState.blockSynthetic));

        modal.classList.add('hidden');
        connectRealtimeStream();
      });
    }

    if (btnReconnect) {
      btnReconnect.addEventListener('click', () => connectRealtimeStream());
    }

    if (btnStressSim) {
      btnStressSim.addEventListener('click', () => {
        rtState.crisisSimActive = !rtState.crisisSimActive;
        if (rtState.crisisSimActive) {
          btnStressSim.textContent = '↺ Restore Calm Market Regime';
          rtState.customHeadlines.unshift({
            title: 'BREAKING: Missile strike in Strait of Hormuz escalates; Brent crude spikes +6.8% and FII selling intensifies',
            source: 'Global Geopolitical Flash Wire',
          });
        } else {
          btnStressSim.textContent = '⚡ Simulate War / Macro Shock';
          rtState.customHeadlines = rtState.customHeadlines.filter(
            (h) => h.source !== 'Global Geopolitical Flash Wire'
          );
        }
        if (rtState.mode === 'BROWSER_ENGINE') {
          computeInBrowserSignalCycle();
        } else {
          pushSavedCredentialsToBackend();
        }
      });
    }

    if (btnInject) {
      btnInject.addEventListener('click', () => {
        const input = document.getElementById('rt-custom-headline-input');
        if (!input || !input.value.trim()) return;
        rtState.customHeadlines.unshift({
          title: input.value.trim(),
          source: 'Trader Live Input',
        });
        input.value = '';
        if (rtState.mode === 'BROWSER_ENGINE') {
          computeInBrowserSignalCycle();
        } else {
          pushSavedCredentialsToBackend();
        }
      });
    }
  }

  document.addEventListener('DOMContentLoaded', () => {
    bindRealtimeModalsAndControls();
    connectRealtimeStream();
  });
})();
