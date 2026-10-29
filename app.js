/**
 * NIFTY 50 Daily High-Low Tracker & Trading Journal with SMC Engine
 * Features:
 * - Smart Money Concepts (SMC) Liquidity & Sweep Engine (PDH, PDL, PWH, PWL, BOS, Sweeps)
 * - Next-Day Algorithmic Directional Bias & Fuel Targets
 * - Beginner Hover-Activated Definitions for All Technical Terms
 * - SMC Terminology Glossary & Searchable Cheat Sheet
 * - Local Storage + Two-Way GitHub Repository Cloud Sync
 * - Interactive TradingView Lightweight Charts & Filtering
 */

// Application State
const state = {
  data: null,
  history: [],          // Chronological list
  displayHistory: [],   // Reverse chronological for table
  filteredHistory: [],
  marks: {},            // Saved in localStorage & GitHub repo
  activeFilter: 'all',  // 'all', 'marked', 'smc', 'inside'
  searchQuery: '',
  currentPage: 1,
  pageSize: 20,
  chart: null,
  candleSeries: null,
  activeTheme: localStorage.getItem('nifty_theme') || 'dark',
  github: {
    repo: localStorage.getItem('nifty_gh_repo') || 'sagnikseal/nifty-tracker',
    token: localStorage.getItem('nifty_gh_token') || '',
    autoSync: localStorage.getItem('nifty_gh_autosync') !== 'false',
    isSyncing: false,
    fileSha: null,
  }
};

// SMC & Trading Terminology Dictionary (For Tooltips and Glossary Modal)
const GLOSSARY_DATA = {
  SMC: {
    term: 'SMC',
    full: 'Smart Money Concepts',
    cat: 'Theory',
    def: 'A trading methodology focused on tracking the footprints of institutional market participants (FIIs, DIIs, banks, and hedge funds) rather than retail indicators.',
    edge: 'Institutions move markets with massive orders that require counterparty liquidity. SMC shows where those orders enter and exit.'
  },
  PDH: {
    term: 'PDH',
    full: 'Previous Day High (External Buy-Side Liquidity)',
    cat: 'Liquidity',
    def: 'The highest price reached during yesterday\'s trading session. Acts as a major cluster for retail stop-losses and breakout buy orders.',
    edge: 'Algorithms treat PDH as "Fuel". Price often surges above PDH to trigger buy stops, and either expands higher (BOS) or reverses downward (Liquidity Sweep).'
  },
  PDL: {
    term: 'PDL',
    full: 'Previous Day Low (External Sell-Side Liquidity)',
    cat: 'Liquidity',
    def: 'The lowest price reached during yesterday\'s session. Clusters retail panic sell stops and breakdown orders.',
    edge: 'Algorithms hunt below PDL to absorb sell orders at discount prices before driving a bullish reversal (Turtle Soup).'
  },
  PWH: {
    term: 'PWH',
    full: 'Previous Week High',
    cat: 'Liquidity',
    def: 'The highest price traded in the preceding calendar week.',
    edge: 'Higher timeframe liquidity pool. When swept, can cause multi-day trend shifts on NIFTY.'
  },
  PWL: {
    term: 'PWL',
    full: 'Previous Week Low',
    cat: 'Liquidity',
    def: 'The lowest price traded in the preceding calendar week.',
    edge: 'Higher timeframe institutional discount level. Major demand target for swing traders.'
  },
  ERL: {
    term: 'ERL',
    full: 'External Range Liquidity',
    cat: 'Liquidity',
    def: 'Liquidity pools sitting outside the recent trading range: PDH, PDL, Weekly Highs/Lows, and major swing extremes.',
    edge: 'The market continuously alternates: it runs from Internal Liquidity (gaps/CPR) outward to sweep External Liquidity (PDH/PDL).'
  },
  IRL: {
    term: 'IRL',
    full: 'Internal Range Liquidity',
    cat: 'Liquidity',
    def: 'Liquidity pools inside the recent range: Fair Value Gaps (FVG), Central Pivot Range (CPR), opening gaps, and 50% Equilibrium.',
    edge: 'After sweeping an external high or low, price frequently retraces back inside to rebalance internal liquidity.'
  },
  BSL: {
    term: 'BSL',
    full: 'Buy-Side Liquidity',
    cat: 'Liquidity',
    def: 'A concentration of buy orders (short-seller stop losses and breakout buyers) above key highs.',
    edge: 'Smart money sells into Buy-Side Liquidity to exit long trades without negative slippage.'
  },
  SSL: {
    term: 'SSL',
    full: 'Sell-Side Liquidity',
    cat: 'Liquidity',
    def: 'A concentration of sell orders (buyer stop losses and breakdown sellers) below key lows.',
    edge: 'Smart money buys into Sell-Side Liquidity at discounted prices to accumulate massive long positions.'
  },
  BOS: {
    term: 'BOS',
    full: 'Break of Structure',
    cat: 'Structure',
    def: 'When a daily candle closes beyond a key swing high or low, confirming trend continuation.',
    edge: 'Indicates strong institutional commitment to carry the market in the direction of the break.'
  },
  Sweep: {
    term: 'Sweep',
    full: 'Liquidity Sweep / Turtle Soup',
    cat: 'Pattern',
    def: 'Price spikes past PDH or PDL to trigger stop-losses, but immediately fails to sustain and closes back inside the range.',
    edge: 'Classic institutional trap. Generates rapid, high-probability counter-trend reversals.'
  },
  CPR: {
    term: 'CPR',
    full: 'Central Pivot Range',
    cat: 'Pivots',
    def: 'A 3-level price band (Top Central TC, Pivot, Bottom Central BC) calculated from yesterday\'s High, Low, and Close.',
    edge: 'Narrow CPR (<40 pts on NIFTY) predicts a strong trending day. Wide CPR (>90 pts) predicts sideways, rangebound chop.'
  },
  ADR: {
    term: 'ADR',
    full: 'Average Daily Range (20-Day)',
    cat: 'Volatility',
    def: 'The average distance between the daily high and daily low over the last 20 trading sessions.',
    edge: 'Used to calculate realistic daily profit targets. When daily range exceeds ADR, exhaustion is likely.'
  },
  Discount: {
    term: 'Discount',
    full: 'Discount Zone (<50% Equilibrium)',
    cat: 'Valuation',
    def: 'When price trades below the 50% midpoint of the 20-day high-low range.',
    edge: 'Institutions prefer buying in Discount zones where prices are mathematically cheap.'
  },
  Premium: {
    term: 'Premium',
    full: 'Premium Zone (>50% Equilibrium)',
    cat: 'Valuation',
    def: 'When price trades above the 50% midpoint of the 20-day high-low range.',
    edge: 'Institutions avoid chasing longs in Premium; they prefer distributing or hedging.'
  },
  Fuel: {
    term: 'Fuel',
    full: 'Liquidity Fuel',
    cat: 'Theory',
    def: 'Retail stop-loss orders that algorithms seek to trigger to absorb liquidity.',
    edge: 'As noted in market theory: trends need fuel to continue. Without liquidity, price stalls.'
  },
  Event: {
    term: 'Event',
    full: 'Macro / Catalyst Event',
    cat: 'Fundamentals',
    def: 'High-impact news (RBI policy, US Fed, CPI, elections, corporate earnings, breaking news on Twitter/X).',
    edge: 'Injects sudden volume that can invalidate standard technical setups and cause aggressive two-sided liquidity sweeps.'
  },
  Pivots: {
    term: 'Pivots',
    full: 'Classic Pivot Points (S1, S2, R1, R2, P)',
    cat: 'Pivots',
    def: 'Support and resistance benchmark levels derived mathematically from the prior day\'s auction.',
    edge: 'Widely observed by institutional execution algos for intraday target setting.'
  }
};

// DOM Elements
const el = {
  valDayHigh: document.getElementById('val-day-high'),
  valDayLow: document.getElementById('val-day-low'),
  valDayRange: document.getElementById('val-day-range'),
  valDayClose: document.getElementById('val-day-close'),
  diffPrevHigh: document.getElementById('diff-prev-high'),
  diffPrevLow: document.getElementById('diff-prev-low'),
  valAdr20: document.getElementById('val-adr-20'),
  valNetChange: document.getElementById('val-net-change'),
  pillRangeType: document.getElementById('pill-range-type'),
  pillCloseChange: document.getElementById('pill-close-change'),
  val52wLow: document.getElementById('val-52w-low'),
  val52wHigh: document.getElementById('val-52w-high'),
  track52wProgress: document.getElementById('track-52w-progress'),
  pin52w: document.getElementById('pin-52w'),
  badgeCprType: document.getElementById('badge-cpr-type'),
  pS1: document.getElementById('p-s1'),
  pBc: document.getElementById('p-bc'),
  pPivot: document.getElementById('p-pivot'),
  pTc: document.getElementById('p-tc'),
  pR1: document.getElementById('p-r1'),

  // SMC Prediction Radar Elements
  smcBiasBadge: document.getElementById('smc-bias-badge'),
  predConfidence: document.getElementById('pred-confidence'),
  predBiasText: document.getElementById('pred-bias-text'),
  predRationale: document.getElementById('pred-rationale'),
  predEventRule: document.getElementById('pred-event-rule'),
  predPrimaryTarget: document.getElementById('pred-primary-target'),
  predSecondaryTarget: document.getElementById('pred-secondary-target'),
  predAdrExp: document.getElementById('pred-adr-exp'),
  predCprExp: document.getElementById('pred-cpr-exp'),
  smcPdh: document.getElementById('smc-pdh'),
  smcPdl: document.getElementById('smc-pdl'),
  smcPwh: document.getElementById('smc-pwh'),
  smcPwl: document.getElementById('smc-pwl'),
  smcZone: document.getElementById('smc-zone'),
  smcStructureEvent: document.getElementById('smc-structure-event'),

  // Table & Controls
  marketStatusBadge: document.getElementById('market-status-badge'),
  marketStatusText: document.getElementById('market-status-text'),
  lastSyncText: document.getElementById('last-sync-text'),
  tableBody: document.getElementById('table-body'),
  tableCountBadge: document.getElementById('table-count-badge'),
  markedCount: document.getElementById('marked-count'),
  paginationInfo: document.getElementById('pagination-info'),
  btnPrevPage: document.getElementById('btn-prev-page'),
  btnNextPage: document.getElementById('btn-next-page'),
  pageNum: document.getElementById('page-number'),
  chartLegend: document.getElementById('chart-legend'),
  tvChart: document.getElementById('tv-chart'),
  btnQuickMark: document.getElementById('btn-quick-mark'),
  btnExportJournal: document.getElementById('btn-export-journal'),
  btnImportJournal: document.getElementById('btn-import-journal'),
  importFileInput: document.getElementById('import-file-input'),
  btnThemeToggle: document.getElementById('btn-theme-toggle'),
  btnDownloadCsv: document.getElementById('btn-download-csv'),
  filterSearch: document.getElementById('filter-search'),
  filterAll: document.getElementById('filter-all'),
  filterMarked: document.getElementById('filter-marked'),
  filterSmc: document.getElementById('filter-smc'),
  filterInside: document.getElementById('filter-inside'),

  // Mark Modal
  markModal: document.getElementById('mark-modal'),
  markForm: document.getElementById('mark-form'),
  btnCloseModal: document.getElementById('btn-close-modal'),
  btnCancelModal: document.getElementById('btn-cancel-modal'),
  btnDeleteMark: document.getElementById('btn-delete-mark'),
  markDate: document.getElementById('mark-date'),
  modalDateSubtitle: document.getElementById('modal-date-subtitle'),
  snapHigh: document.getElementById('snap-high'),
  snapLow: document.getElementById('snap-low'),
  snapRange: document.getElementById('snap-range'),
  snapClose: document.getElementById('snap-close'),
  markPattern: document.getElementById('mark-pattern'),
  markTradeTaken: document.getElementById('mark-trade-taken'),
  markOutcome: document.getElementById('mark-outcome'),
  markLevels: document.getElementById('mark-levels'),
  markPredBias: document.getElementById('mark-pred-bias'),
  markPredTarget: document.getElementById('mark-pred-target'),
  markEventCheck: document.getElementById('mark-event-check'),
  markNotes: document.getElementById('mark-notes'),
  markTags: document.getElementById('mark-tags'),
  toast: document.getElementById('toast'),

  // GitHub Sync Elements
  btnGithubSync: document.getElementById('btn-github-sync'),
  syncBtnLabel: document.getElementById('sync-btn-label'),
  syncModal: document.getElementById('sync-modal'),
  btnCloseSyncModal: document.getElementById('btn-close-sync-modal'),
  ghRepoInput: document.getElementById('gh-repo-input'),
  ghTokenInput: document.getElementById('gh-token-input'),
  ghAutoSyncToggle: document.getElementById('gh-auto-sync-toggle'),
  btnSaveGhToken: document.getElementById('btn-save-gh-token'),
  btnPushToGithub: document.getElementById('btn-push-to-github'),
  btnPullFromGithub: document.getElementById('btn-pull-from-github'),
  btnDisconnectGithub: document.getElementById('btn-disconnect-github'),
  syncStatusMsg: document.getElementById('sync-status-msg'),

  // Glossary & Tooltips
  btnOpenGlossary: document.getElementById('btn-open-glossary'),
  glossaryModal: document.getElementById('glossary-modal'),
  btnCloseGlossaryModal: document.getElementById('btn-close-glossary-modal'),
  glossarySearch: document.getElementById('glossary-search'),
  glossaryGrid: document.getElementById('glossary-grid'),
  termTooltip: document.getElementById('term-tooltip'),
  ttTerm: document.getElementById('tt-term'),
  ttFull: document.getElementById('tt-full'),
  ttBody: document.getElementById('tt-body'),
  ttEdge: document.getElementById('tt-edge'),
};

// Number Format Helper
function formatNumber(num) {
  if (num === null || num === undefined || isNaN(num)) return '--';
  return Number(num).toLocaleString('en-IN', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  });
}

function showToast(message, duration = 3000) {
  el.toast.textContent = message;
  el.toast.classList.add('show');
  setTimeout(() => el.toast.classList.remove('show'), duration);
}

// ==========================================================================
// Market Clock (IST)
// ==========================================================================
function updateMarketStatus() {
  const now = new Date();
  const utc = now.getTime() + (now.getTimezoneOffset() * 60000);
  const ist = new Date(utc + (3600000 * 5.5));
  
  const day = ist.getDay();
  const hours = ist.getHours();
  const minutes = ist.getMinutes();
  const totalMinutes = hours * 60 + minutes;
  
  const isWeekend = (day === 0 || day === 6);
  const isMarketHours = (!isWeekend && totalMinutes >= (9 * 60 + 15) && totalMinutes <= (15 * 60 + 30));
  const isPreMarket = (!isWeekend && totalMinutes >= (9 * 60) && totalMinutes < (9 * 60 + 15));
  const isPostClose = (!isWeekend && totalMinutes > (15 * 60 + 30));

  el.marketStatusBadge.className = 'market-status';

  if (isMarketHours) {
    el.marketStatusBadge.classList.add('status-open');
    el.marketStatusText.textContent = 'NSE Live (Closes 3:30 PM IST)';
  } else if (isPreMarket) {
    el.marketStatusBadge.classList.add('status-open');
    el.marketStatusText.textContent = 'Pre-Market Session (Opens 9:15 AM)';
  } else if (isWeekend) {
    el.marketStatusBadge.classList.add('status-closed');
    el.marketStatusText.textContent = 'Market Closed (Weekend)';
  } else if (isPostClose) {
    el.marketStatusBadge.classList.add('status-closed');
    el.marketStatusText.textContent = 'Market Closed (Settled for Today)';
  } else {
    el.marketStatusBadge.classList.add('status-closed');
    el.marketStatusText.textContent = 'Market Closed (Opens 9:15 AM IST)';
  }
}

// ==========================================================================
// Hover-Activated Term Tooltip Engine
// ==========================================================================
function initTermTooltips() {
  let activeTarget = null;

  document.addEventListener('mouseover', e => {
    const tipEl = e.target.closest('.term-tip');
    if (!tipEl) return;

    activeTarget = tipEl;
    const termKey = tipEl.dataset.term;
    const item = GLOSSARY_DATA[termKey];
    if (!item) return;

    el.ttTerm.textContent = item.term;
    el.ttFull.textContent = item.full;
    el.ttBody.textContent = item.def;
    el.ttEdge.textContent = item.edge;

    el.termTooltip.style.display = 'block';
    positionTooltip(e);
    requestAnimationFrame(() => el.termTooltip.classList.add('visible'));
  });

  document.addEventListener('mousemove', e => {
    if (activeTarget && el.termTooltip.classList.contains('visible')) {
      positionTooltip(e);
    }
  });

  document.addEventListener('mouseout', e => {
    const tipEl = e.target.closest('.term-tip');
    if (tipEl && tipEl === activeTarget) {
      activeTarget = null;
      el.termTooltip.classList.remove('visible');
      setTimeout(() => {
        if (!activeTarget) el.termTooltip.style.display = 'none';
      }, 150);
    }
  });
}

function positionTooltip(e) {
  const tt = el.termTooltip;
  const padding = 14;
  let x = e.clientX + 16;
  let y = e.clientY + 16;

  // Window bounds check
  const rect = tt.getBoundingClientRect();
  if (x + rect.width + padding > window.innerWidth) {
    x = e.clientX - rect.width - 12;
  }
  if (y + rect.height + padding > window.innerHeight) {
    y = e.clientY - rect.height - 12;
  }

  tt.style.left = `${Math.max(padding, x)}px`;
  tt.style.top = `${Math.max(padding, y)}px`;
}

// ==========================================================================
// SMC Glossary Modal & Search
// ==========================================================================
function renderGlossary(filterText = '') {
  if (!el.glossaryGrid) return;

  const query = filterText.toLowerCase().trim();
  const entries = Object.values(GLOSSARY_DATA).filter(item => {
    if (!query) return true;
    return (
      item.term.toLowerCase().includes(query) ||
      item.full.toLowerCase().includes(query) ||
      item.def.toLowerCase().includes(query) ||
      item.cat.toLowerCase().includes(query)
    );
  });

  if (entries.length === 0) {
    el.glossaryGrid.innerHTML = `
      <div style="grid-column: 1 / -1; padding: 2rem; text-align: center; color: var(--text-muted);">
        No matching SMC concepts found for "${filterText}".
      </div>
    `;
    return;
  }

  el.glossaryGrid.innerHTML = entries.map(item => `
    <div class="glossary-card">
      <div class="glossary-card-top">
        <span class="glossary-term">${item.term}</span>
        <span class="glossary-cat">${item.cat}</span>
      </div>
      <div class="glossary-full-name">${item.full}</div>
      <div class="glossary-def">${item.def}</div>
      <div class="glossary-edge">
        <strong style="color:var(--blue);">Smart Money Edge:</strong> ${item.edge}
      </div>
    </div>
  `).join('');
}

// ==========================================================================
// Trading Marks Management: Local Storage + GitHub Repository
// ==========================================================================
async function loadMarks() {
  try {
    const raw = localStorage.getItem('nifty_user_marks');
    state.marks = raw ? JSON.parse(raw) : {};
  } catch (e) {
    console.error('Failed to load marks from localStorage:', e);
    state.marks = {};
  }

  try {
    const resp = await fetch('data/trading_marks.json?v=' + Date.now());
    if (resp.ok) {
      const repoData = await resp.json();
      const repoMarks = repoData.marks || {};
      let mergedCount = 0;
      for (const [dt, m] of Object.entries(repoMarks)) {
        if (!state.marks[dt]) {
          state.marks[dt] = m;
          mergedCount++;
        }
      }
      if (mergedCount > 0) {
        saveMarksToStorage(false);
      }
    }
  } catch (e) {
    console.log('No repository trading_marks.json loaded or file offline');
  }

  updateMarkedCount();
  updateSyncButtonUI();
}

function saveMarksToStorage(triggerGitHubSync = true) {
  try {
    localStorage.setItem('nifty_user_marks', JSON.stringify(state.marks));
    updateMarkedCount();
  } catch (e) {
    console.error('Failed to save marks to localStorage:', e);
  }

  if (triggerGitHubSync && state.github.token && state.github.autoSync) {
    pushMarksToGitHub(true);
  }
}

function updateMarkedCount() {
  const count = Object.keys(state.marks).length;
  el.markedCount.textContent = count;
}

// ==========================================================================
// GitHub Repository Sync API Integration
// ==========================================================================
function updateSyncButtonUI() {
  if (state.github.isSyncing) {
    el.syncBtnLabel.textContent = 'Syncing...';
    el.btnGithubSync.style.borderColor = 'var(--yellow)';
  } else if (state.github.token) {
    el.syncBtnLabel.textContent = '🟢 GitHub Synced';
    el.btnGithubSync.style.borderColor = 'var(--green)';
  } else {
    el.syncBtnLabel.textContent = '☁️ GitHub Sync';
    el.btnGithubSync.style.borderColor = 'var(--border-subtle)';
  }
}

async function getFileShaFromGitHub(repo, token) {
  const url = `https://api.github.com/repos/${repo}/contents/data/trading_marks.json`;
  const resp = await fetch(url, {
    headers: {
      'Authorization': `Bearer ${token}`,
      'Accept': 'application/vnd.github+json',
      'X-GitHub-Api-Version': '2022-11-28'
    }
  });

  if (resp.status === 404) return null;
  if (!resp.ok) {
    const errData = await resp.json();
    throw new Error(errData.message || `GitHub error ${resp.status}`);
  }
  const data = await resp.json();
  return data.sha;
}

async function pushMarksToGitHub(silent = false) {
  const { repo, token } = state.github;
  if (!token || !repo) {
    if (!silent) alert('Please configure your GitHub Repository and Token first.');
    return;
  }

  state.github.isSyncing = true;
  updateSyncButtonUI();
  if (el.syncStatusMsg) el.syncStatusMsg.innerHTML = '<span style="color:var(--yellow)">⏳ Committing to GitHub repository...</span>';

  try {
    const sha = await getFileShaFromGitHub(repo, token);

    const payload = {
      version: '1.0',
      last_updated: new Date().toISOString(),
      marks: state.marks
    };

    const contentStr = JSON.stringify(payload, null, 2);
    const contentB64 = btoa(unescape(encodeURIComponent(contentStr)));

    const putUrl = `https://api.github.com/repos/${repo}/contents/data/trading_marks.json`;
    const body = {
      message: `Update trading marks (${Object.keys(state.marks).length} entries) [skip ci]`,
      content: contentB64,
    };
    if (sha) body.sha = sha;

    const resp = await fetch(putUrl, {
      method: 'PUT',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Accept': 'application/vnd.github+json',
        'Content-Type': 'application/json',
        'X-GitHub-Api-Version': '2022-11-28'
      },
      body: JSON.stringify(body)
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.message || `HTTP ${resp.status}`);
    }

    state.github.isSyncing = false;
    updateSyncButtonUI();
    if (el.syncStatusMsg) {
      el.syncStatusMsg.innerHTML = `<span style="color:var(--green)">✅ Successfully committed to ${repo}!</span>`;
    }
    showToast(`☁️ Marks committed to GitHub (${repo})!`);
  } catch (err) {
    console.error('GitHub Push failed:', err);
    state.github.isSyncing = false;
    updateSyncButtonUI();
    if (el.syncStatusMsg) {
      el.syncStatusMsg.innerHTML = `<span style="color:var(--red)">❌ Sync failed: ${err.message}</span>`;
    }
    if (!silent) alert(`GitHub Sync error: ${err.message}`);
  }
}

async function pullMarksFromGitHub() {
  const { repo, token } = state.github;
  if (!repo) return;

  if (el.syncStatusMsg) el.syncStatusMsg.innerHTML = '<span style="color:var(--yellow)">⏳ Fetching from GitHub...</span>';

  try {
    const url = `https://api.github.com/repos/${repo}/contents/data/trading_marks.json`;
    const headers = { 'Accept': 'application/vnd.github+json' };
    if (token) headers['Authorization'] = `Bearer ${token}`;

    const resp = await fetch(url, { headers });
    if (!resp.ok) {
      throw new Error(`Could not fetch data/trading_marks.json (${resp.status})`);
    }

    const data = await resp.json();
    const jsonStr = decodeURIComponent(escape(atob(data.content)));
    const parsed = JSON.parse(jsonStr);
    const incomingMarks = parsed.marks || {};

    let added = 0;
    for (const [dt, m] of Object.entries(incomingMarks)) {
      state.marks[dt] = m;
      added++;
    }

    localStorage.setItem('nifty_user_marks', JSON.stringify(state.marks));
    updateMarkedCount();
    updateChartMarkers();
    applyFilter(state.activeFilter);

    if (el.syncStatusMsg) {
      el.syncStatusMsg.innerHTML = `<span style="color:var(--green)">✅ Pulled ${added} marks from GitHub!</span>`;
    }
    showToast(`☁️ Pulled latest marks from GitHub!`);
  } catch (err) {
    console.error('GitHub Pull failed:', err);
    if (el.syncStatusMsg) {
      el.syncStatusMsg.innerHTML = `<span style="color:var(--red)">❌ Pull failed: ${err.message}</span>`;
    }
  }
}

// ==========================================================================
// Data Fetching: NIFTY 50 Daily Data
// ==========================================================================
async function loadNiftyData() {
  try {
    const response = await fetch('data/nifty_data.json?v=' + Date.now());
    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }
    const data = await response.json();
    state.data = data;
    state.history = data.history || [];
    state.displayHistory = [...state.history].reverse();
    state.filteredHistory = [...state.displayHistory];
    
    renderHeroMetrics();
    renderSmcPrediction();
    renderChart();
    applyFilter(state.activeFilter);
    el.lastSyncText.textContent = `Sync: ${data.last_updated || 'Updated'}`;
  } catch (err) {
    console.error('Error loading NIFTY data:', err);
    el.tableBody.innerHTML = `
      <tr>
        <td colspan="11" class="text-center" style="padding: 2rem; color: var(--red);">
          ⚠️ Could not load <code>data/nifty_data.json</code>.
        </td>
      </tr>
    `;
    el.lastSyncText.textContent = 'Sync: Failed';
  }
}

// ==========================================================================
// Render Hero & Levels
// ==========================================================================
function renderHeroMetrics() {
  const latest = state.data?.latest;
  const prev = state.data?.prev_day;
  const meta = state.data?.meta;

  if (!latest) return;

  // Daily High
  el.valDayHigh.textContent = formatNumber(latest.high);
  if (prev) {
    const diffHigh = round(latest.high - prev.high, 2);
    const sign = diffHigh >= 0 ? '+' : '';
    el.diffPrevHigh.textContent = `${sign}${formatNumber(diffHigh)} pts`;
    el.diffPrevHigh.style.color = diffHigh >= 0 ? 'var(--green)' : 'var(--red)';
  }

  // Daily Low
  el.valDayLow.textContent = formatNumber(latest.low);
  if (prev) {
    const diffLow = round(latest.low - prev.low, 2);
    const sign = diffLow >= 0 ? '+' : '';
    el.diffPrevLow.textContent = `${sign}${formatNumber(diffLow)} pts`;
    el.diffPrevLow.style.color = diffLow >= 0 ? 'var(--green)' : 'var(--red)';
  }

  // Daily Range
  el.valDayRange.textContent = `${formatNumber(latest.range)} pts`;
  el.pillRangeType.textContent = `${latest.range_pct}%`;
  el.valAdr20.textContent = `${formatNumber(meta?.adr_20 || latest.range)} pts`;

  // Close & Net Change
  el.valDayClose.textContent = formatNumber(latest.close);
  const changeSign = latest.change >= 0 ? '+' : '';
  el.valNetChange.textContent = `${changeSign}${formatNumber(latest.change)} pts`;
  el.valNetChange.style.color = latest.change >= 0 ? 'var(--green)' : 'var(--red)';
  el.pillCloseChange.textContent = `${changeSign}${latest.change_pct}%`;
  el.pillCloseChange.className = `card-pill ${latest.change >= 0 ? 'pill-high' : 'pill-low'}`;

  // 52-Week Range Positioning
  const low52 = meta?.fifty_two_week_low || latest.low;
  const high52 = meta?.fifty_two_week_high || latest.high;
  el.val52wLow.textContent = formatNumber(low52);
  el.val52wHigh.textContent = formatNumber(high52);

  const rangeSpan = high52 - low52;
  const posPct = rangeSpan > 0 ? Math.min(Math.max(((latest.close - low52) / rangeSpan) * 100, 2), 98) : 50;
  el.pin52w.style.left = `${posPct}%`;
  el.pin52w.title = `Current Close: ${formatNumber(latest.close)} (${posPct.toFixed(1)}% of 52W Range)`;

  // CPR & Pivots
  el.badgeCprType.textContent = `CPR: ${latest.cpr_type || 'Normal'} (${latest.cpr_width || '--'} pts)`;
  el.pS1.textContent = formatNumber(latest.s1);
  el.pBc.textContent = formatNumber(latest.bc);
  el.pPivot.textContent = formatNumber(latest.pivot);
  el.pTc.textContent = formatNumber(latest.tc);
  el.pR1.textContent = formatNumber(latest.r1);
}

// ==========================================================================
// Render SMC Prediction & Liquidity Radar
// ==========================================================================
function renderSmcPrediction() {
  const pred = state.data?.next_day_prediction;
  const latest = state.data?.latest;
  const meta = state.data?.meta;

  if (!pred || !latest) return;

  // Bias Headline & Badge
  el.smcBiasBadge.textContent = pred.bias;
  el.predBiasText.textContent = pred.bias;
  el.predConfidence.textContent = `${pred.confidence} Confidence`;

  const isBull = pred.bias.toLowerCase().includes('bull');
  const isBear = pred.bias.toLowerCase().includes('bear');
  el.predBiasText.style.color = isBull ? 'var(--green)' : (isBear ? 'var(--red)' : 'var(--yellow)');

  // Rationale & Event Warning
  el.predRationale.textContent = pred.rationale;
  if (el.predEventRule) el.predEventRule.textContent = pred.event_rule;

  // Targets
  el.predPrimaryTarget.textContent = pred.primary_liquidity_target;
  el.predSecondaryTarget.textContent = pred.secondary_target;
  el.predAdrExp.textContent = `${meta?.adr_20 || '--'} pts`;
  el.predCprExp.textContent = latest.cpr_type || 'Normal';

  // Matrix: PDH, PDL, PWH, PWL, Zone, Structure
  el.smcPdh.textContent = formatNumber(latest.high);
  el.smcPdl.textContent = formatNumber(latest.low);
  el.smcPwh.textContent = latest.pwh ? formatNumber(latest.pwh) : '--';
  el.smcPwl.textContent = latest.pwl ? formatNumber(latest.pwl) : '--';

  const zoneText = `${latest.zone || 'Discount'} Zone`;
  el.smcZone.innerHTML = `<span class="term-tip" data-term="${latest.zone || 'Discount'}">${zoneText}</span>`;

  let structBadge = latest.smc_structure || 'Normal';
  el.smcStructureEvent.innerHTML = `<span class="term-tip" data-term="BOS">${structBadge}</span>`;
}

function openMarkModalWithPrediction() {
  const latest = state.data?.latest;
  const pred = state.data?.next_day_prediction;
  openMarkModal(latest?.date);

  if (pred) {
    if (el.markPredBias) {
      if (pred.bias.toLowerCase().includes('bull')) el.markPredBias.value = 'Bullish Continuation';
      else if (pred.bias.toLowerCase().includes('bear')) el.markPredBias.value = 'Bearish Continuation';
      else if (pred.bias.toLowerCase().includes('revers')) el.markPredBias.value = 'BSL Sweep & Reverse';
      else el.markPredBias.value = 'Rangebound / Inside Value';
    }
    if (el.markPredTarget) {
      el.markPredTarget.value = pred.primary_liquidity_target;
    }
  }
}

function round(val, dec = 2) {
  return Number(Math.round(val + 'e' + dec) + 'e-' + dec);
}

// ==========================================================================
// Lightweight Chart
// ==========================================================================
function renderChart() {
  if (!window.LightweightCharts || !el.tvChart) return;

  el.tvChart.innerHTML = '';

  const isDark = state.activeTheme === 'dark';
  const bgColor = isDark ? '#141721' : '#ffffff';
  const textColor = isDark ? '#94a3b8' : '#475569';
  const gridColor = isDark ? '#1c2130' : '#f1f5f9';

  state.chart = LightweightCharts.createChart(el.tvChart, {
    width: el.tvChart.clientWidth,
    height: 420,
    layout: {
      background: { color: bgColor },
      textColor: textColor,
      fontFamily: 'Inter, sans-serif',
      fontSize: 12,
    },
    grid: {
      vertLines: { color: gridColor },
      horzLines: { color: gridColor },
    },
    rightPriceScale: {
      borderColor: isDark ? '#23293a' : '#e2e8f0',
      scaleMargins: { top: 0.1, bottom: 0.1 },
    },
    timeScale: {
      borderColor: isDark ? '#23293a' : '#e2e8f0',
      timeVisible: true,
      secondsVisible: false,
    },
    crosshair: {
      mode: LightweightCharts.CrosshairMode.Normal,
    },
  });

  state.candleSeries = state.chart.addCandlestickSeries({
    upColor: '#10b981',
    downColor: '#ef4444',
    borderDownColor: '#ef4444',
    borderUpColor: '#10b981',
    wickDownColor: '#ef4444',
    wickUpColor: '#10b981',
  });

  const chartData = state.history.map(item => ({
    time: item.date,
    open: item.open,
    high: item.high,
    low: item.low,
    close: item.close,
  }));

  state.candleSeries.setData(chartData);
  updateChartMarkers();
  setTimeframeDays(90);

  state.chart.subscribeCrosshairMove(param => {
    if (!param || !param.time || !param.seriesData) {
      const latest = state.data?.latest;
      if (latest) {
        el.chartLegend.innerHTML = `<strong>${latest.date}</strong> | O: <strong>${formatNumber(latest.open)}</strong> H: <strong style="color:var(--green)">${formatNumber(latest.high)}</strong> L: <strong style="color:var(--red)">${formatNumber(latest.low)}</strong> C: <strong>${formatNumber(latest.close)}</strong> | Range: <strong>${formatNumber(latest.range)}</strong>`;
      }
      return;
    }
    const candle = param.seriesData.get(state.candleSeries);
    if (candle) {
      const dRange = round(candle.high - candle.low, 2);
      el.chartLegend.innerHTML = `<strong>${param.time}</strong> | O: <strong>${formatNumber(candle.open)}</strong> H: <strong style="color:var(--green)">${formatNumber(candle.high)}</strong> L: <strong style="color:var(--red)">${formatNumber(candle.low)}</strong> C: <strong>${formatNumber(candle.close)}</strong> | Range: <strong>${formatNumber(dRange)}</strong>`;
    }
  });

  window.addEventListener('resize', () => {
    if (state.chart && el.tvChart) {
      state.chart.applyOptions({ width: el.tvChart.clientWidth });
    }
  });
}

function updateChartMarkers() {
  if (!state.candleSeries) return;

  const markers = [];
  for (const [date, mark] of Object.entries(state.marks)) {
    const isBull = mark.sentiment === 'Bullish';
    const isBear = mark.sentiment === 'Bearish';
    markers.push({
      time: date,
      position: isBear ? 'aboveBar' : 'belowBar',
      color: isBull ? '#10b981' : (isBear ? '#ef4444' : '#3b82f6'),
      shape: isBear ? 'arrowDown' : 'arrowUp',
      text: mark.sentiment ? mark.sentiment.charAt(0) : 'M',
    });
  }

  markers.sort((a, b) => (a.time > b.time ? 1 : -1));
  state.candleSeries.setMarkers(markers);
}

function setTimeframeDays(days) {
  if (!state.chart || state.history.length === 0) return;

  if (days === 'all') {
    state.chart.timeScale().fitContent();
    return;
  }

  const count = parseInt(days, 10);
  const total = state.history.length;
  const startIndex = Math.max(0, total - count);
  const fromTime = state.history[startIndex].date;
  const toTime = state.history[total - 1].date;

  state.chart.timeScale().setVisibleRange({
    from: fromTime,
    to: toTime,
  });
}

// ==========================================================================
// Table Rendering & Filtering
// ==========================================================================
function applyFilter(filterName) {
  state.activeFilter = filterName;

  el.filterAll.classList.toggle('active', filterName === 'all');
  el.filterMarked.classList.toggle('active', filterName === 'marked');
  if (el.filterSmc) el.filterSmc.classList.toggle('active', filterName === 'smc');
  el.filterInside.classList.toggle('active', filterName === 'inside');

  let list = [...state.displayHistory];

  if (filterName === 'marked') {
    list = list.filter(item => Boolean(state.marks[item.date]));
  } else if (filterName === 'inside') {
    list = list.filter(item => item.is_inside_day);
  } else if (filterName === 'smc') {
    list = list.filter(item => item.pdh_sweep || item.pdl_sweep || item.bullish_bos || item.bearish_bos);
  }

  if (state.searchQuery.trim() !== '') {
    const q = state.searchQuery.toLowerCase().trim();
    list = list.filter(item => {
      const mark = state.marks[item.date];
      const matchDate = item.date.includes(q);
      const matchNotes = mark?.notes?.toLowerCase().includes(q);
      const matchPattern = (mark?.pattern || item.candle_type || item.smc_structure)?.toLowerCase().includes(q);
      const matchTags = mark?.tags?.toLowerCase().includes(q);
      const matchLevels = mark?.levels?.toLowerCase().includes(q);
      return matchDate || matchNotes || matchPattern || matchTags || matchLevels;
    });
  }

  state.filteredHistory = list;
  state.currentPage = 1;
  renderTable();
}

function renderTable() {
  const totalItems = state.filteredHistory.length;
  el.tableCountBadge.textContent = `${totalItems} trading days`;

  if (totalItems === 0) {
    el.tableBody.innerHTML = `
      <tr>
        <td colspan="11" class="text-center" style="padding: 2.5rem; color: var(--text-muted);">
          No records found matching the active filter or search query.
        </td>
      </tr>
    `;
    el.paginationInfo.textContent = 'Showing 0 records';
    el.btnPrevPage.disabled = true;
    el.btnNextPage.disabled = true;
    return;
  }

  const totalPages = Math.ceil(totalItems / state.pageSize);
  const startIdx = (state.currentPage - 1) * state.pageSize;
  const endIdx = Math.min(startIdx + state.pageSize, totalItems);
  const pageItems = state.filteredHistory.slice(startIdx, endIdx);

  el.tableBody.innerHTML = pageItems.map(item => {
    const mark = state.marks[item.date];
    const isUp = item.change >= 0;
    const changeSign = isUp ? '+' : '';

    // Standard Badges
    const insideBadge = item.is_inside_day ? `<span class="pill-badge badge-inside term-tip" data-term="Inside Bar">Inside Bar</span>` : '';
    const outsideBadge = item.is_outside_day ? `<span class="pill-badge badge-outside term-tip" data-term="Outside Bar">Outside Bar</span>` : '';
    const gapBadge = item.gap_type === 'Gap Up' ? `<span class="pill-badge badge-gap-up">Gap Up (${item.gap})</span>`
                   : (item.gap_type === 'Gap Down' ? `<span class="pill-badge badge-gap-down">Gap Down (${item.gap})</span>` : '');

    // SMC Structure Badge
    let smcBadge = '';
    if (item.pdh_sweep) {
      smcBadge = `<span class="pill-badge badge-sweep-bear term-tip" data-term="Sweep">⚡ BSL Swept (Reversal)</span>`;
    } else if (item.pdl_sweep) {
      smcBadge = `<span class="pill-badge badge-sweep-bull term-tip" data-term="Sweep">⚡ SSL Swept (Turtle Soup)</span>`;
    } else if (item.bullish_bos) {
      smcBadge = `<span class="pill-badge badge-bos-bull term-tip" data-term="BOS">Bullish BOS</span>`;
    } else if (item.bearish_bos) {
      smcBadge = `<span class="pill-badge badge-bos-bear term-tip" data-term="BOS">Bearish BOS</span>`;
    } else if (item.is_expansion_bar) {
      smcBadge = `<span class="pill-badge badge-inside term-tip" data-term="Expansion Bar">Expansion (${item.body_pct}% body)</span>`;
    } else {
      smcBadge = `<span style="font-size:0.7rem; color:var(--text-dim);">${item.zone || 'Normal'}</span>`;
    }

    // User Mark Cell
    let markCellHtml = '';
    if (mark) {
      const sentimentClass = `sentiment-${mark.sentiment || 'Neutral'}`;
      const tagList = mark.tags ? mark.tags.split(',').map(t => `<span class="mark-tag-pill">${t.trim()}</span>`).join(' ') : '';
      const predHtml = mark.predBias ? `<span style="font-size:0.68rem; color:var(--blue); font-weight:700;">Pred: ${mark.predBias}</span>` : '';
      const eventPill = mark.isEvent ? `<span style="font-size:0.68rem; color:var(--yellow); font-weight:700;">⚠️ Event</span>` : '';

      markCellHtml = `
        <div class="mark-cell-content" onclick="openMarkModal('${item.date}')" style="cursor: pointer;" title="Click to view/edit mark">
          <span class="badge-sentiment ${sentimentClass}">${mark.sentiment || 'Marked'}</span>
          ${predHtml}
          ${eventPill}
          <span class="mark-notes-snippet">${mark.notes ? escapeHtml(mark.notes) : (mark.pattern || 'Marked day')}</span>
          ${tagList}
        </div>
      `;
    } else {
      markCellHtml = `
        <button class="btn btn-sm btn-outline" onclick="openMarkModal('${item.date}')">
          + Mark Info
        </button>
      `;
    }

    return `
      <tr id="row-${item.date}">
        <td>
          <div style="font-weight: 700;">${item.date}</div>
          <div style="font-size: 0.72rem; color: var(--text-dim);">${item.day}</div>
        </td>
        <td class="text-right num-mono td-high">${formatNumber(item.high)}</td>
        <td class="text-right num-mono td-low">${formatNumber(item.low)}</td>
        <td class="text-right num-mono">
          <div>${formatNumber(item.range)}</div>
          <div style="font-size: 0.7rem; color: var(--text-dim);">${item.range_pct}%</div>
        </td>
        <td class="text-right num-mono">${formatNumber(item.open)}</td>
        <td class="text-right num-mono" style="font-weight: 700;">${formatNumber(item.close)}</td>
        <td class="text-right num-mono ${isUp ? 'td-up' : 'td-down'}">
          <div>${changeSign}${formatNumber(item.change)}</div>
          <div style="font-size: 0.7rem;">${changeSign}${item.change_pct}%</div>
        </td>
        <td>
          ${insideBadge}
          ${outsideBadge}
          ${gapBadge}
          <div style="font-size: 0.7rem; color: var(--text-muted); margin-top: 2px;">
            P: ${formatNumber(item.pivot)} | CPR: ${item.cpr_width} pts
          </div>
        </td>
        <td>
          ${smcBadge}
        </td>
        <td>${markCellHtml}</td>
        <td class="text-center">
          <button class="btn btn-sm btn-outline" onclick="openMarkModal('${item.date}')" title="Edit Trading Mark">
            ✍️
          </button>
        </td>
      </tr>
    `;
  }).join('');

  el.paginationInfo.textContent = `Showing ${startIdx + 1}–${endIdx} of ${totalItems} records`;
  el.pageNum.textContent = `${state.currentPage} / ${totalPages}`;
  el.btnPrevPage.disabled = state.currentPage <= 1;
  el.btnNextPage.disabled = state.currentPage >= totalPages;
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ==========================================================================
// Mark Trading Info Modal
// ==========================================================================
function openMarkModal(dateStr) {
  const targetDate = dateStr || state.data?.latest?.date || new Date().toISOString().split('T')[0];
  el.markDate.value = targetDate;
  el.modalDateSubtitle.textContent = `Date: ${targetDate}`;

  const record = state.history.find(d => d.date === targetDate);
  if (record) {
    el.snapHigh.textContent = formatNumber(record.high);
    el.snapLow.textContent = formatNumber(record.low);
    el.snapRange.textContent = `${formatNumber(record.range)} pts`;
    el.snapClose.textContent = formatNumber(record.close);
  } else {
    el.snapHigh.textContent = '--';
    el.snapLow.textContent = '--';
    el.snapRange.textContent = '--';
    el.snapClose.textContent = '--';
  }

  const existing = state.marks[targetDate];
  if (existing) {
    const radio = document.querySelector(`input[name="sentiment"][value="${existing.sentiment || 'Bullish'}"]`);
    if (radio) radio.checked = true;

    el.markPattern.value = existing.pattern || 'Trend Day (Unidirectional)';
    el.markTradeTaken.value = existing.tradeTaken || 'No';
    el.markOutcome.value = existing.outcome || 'N/A';
    el.markLevels.value = existing.levels || '';
    if (el.markPredBias) el.markPredBias.value = existing.predBias || 'Bullish Continuation';
    if (el.markPredTarget) el.markPredTarget.value = existing.predTarget || '';
    if (el.markEventCheck) el.markEventCheck.checked = Boolean(existing.isEvent);
    el.markNotes.value = existing.notes || '';
    el.markTags.value = existing.tags || '';
    el.btnDeleteMark.style.display = 'inline-block';
  } else {
    const defaultRadio = document.querySelector('input[name="sentiment"][value="Bullish"]');
    if (defaultRadio) defaultRadio.checked = true;
    el.markPattern.value = record?.is_inside_day ? 'Inside Bar Setup (Harami)' : 'Trend Day (Unidirectional)';
    el.markTradeTaken.value = 'No';
    el.markOutcome.value = 'N/A';
    el.markLevels.value = record ? `PDH: ${record.high}, PDL: ${record.low}` : '';
    if (el.markPredBias) {
      el.markPredBias.value = record?.close > record?.open ? 'Bullish Continuation' : 'Bearish Continuation';
    }
    if (el.markPredTarget) {
      el.markPredTarget.value = record ? `Sweep PDH: ${record.high}` : '';
    }
    if (el.markEventCheck) el.markEventCheck.checked = false;
    el.markNotes.value = '';
    el.markTags.value = record?.pdh_sweep ? '#BSL-Sweep' : (record?.pdl_sweep ? '#SSL-Sweep' : '');
    el.btnDeleteMark.style.display = 'none';
  }

  el.markModal.style.display = 'flex';
}

function closeMarkModal() {
  el.markModal.style.display = 'none';
}

function appendLevel(text) {
  const current = el.markLevels.value.trim();
  el.markLevels.value = current ? `${current}, ${text}` : text;
}

function appendTarget(text) {
  const current = el.markPredTarget.value.trim();
  el.markPredTarget.value = current ? `${current}, ${text}` : text;
}

function appendTag(tag) {
  const current = el.markTags.value.trim();
  if (!current.includes(tag)) {
    el.markTags.value = current ? `${current}, ${tag}` : tag;
  }
}

// ==========================================================================
// Event Listeners & Interactions
// ==========================================================================
function setupEventListeners() {
  // Theme Toggle
  el.btnThemeToggle.addEventListener('click', () => {
    const isDark = document.body.classList.contains('theme-dark');
    const newTheme = isDark ? 'light' : 'dark';
    document.body.className = `theme-${newTheme}`;
    state.activeTheme = newTheme;
    localStorage.setItem('nifty_theme', newTheme);
    renderChart();
  });

  // Timeframe selector buttons
  const tfContainer = document.getElementById('timeframe-buttons');
  if (tfContainer) {
    tfContainer.addEventListener('click', e => {
      const btn = e.target.closest('.btn-tf');
      if (!btn) return;
      tfContainer.querySelectorAll('.btn-tf').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      setTimeframeDays(btn.dataset.days);
    });
  }

  // Filter Buttons
  el.filterAll.addEventListener('click', () => applyFilter('all'));
  el.filterMarked.addEventListener('click', () => applyFilter('marked'));
  if (el.filterSmc) el.filterSmc.addEventListener('click', () => applyFilter('smc'));
  el.filterInside.addEventListener('click', () => applyFilter('inside'));

  // Search input with debounce
  let debounceTimer;
  el.filterSearch.addEventListener('input', e => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      state.searchQuery = e.target.value;
      applyFilter(state.activeFilter);
    }, 200);
  });

  // Pagination
  el.btnPrevPage.addEventListener('click', () => {
    if (state.currentPage > 1) {
      state.currentPage--;
      renderTable();
    }
  });

  el.btnNextPage.addEventListener('click', () => {
    const totalPages = Math.ceil(state.filteredHistory.length / state.pageSize);
    if (state.currentPage < totalPages) {
      state.currentPage++;
      renderTable();
    }
  });

  // Quick Mark Button
  el.btnQuickMark.addEventListener('click', () => openMarkModal());

  // Modal Close
  el.btnCloseModal.addEventListener('click', closeMarkModal);
  el.btnCancelModal.addEventListener('click', closeMarkModal);
  el.markModal.addEventListener('click', e => {
    if (e.target === el.markModal) closeMarkModal();
  });

  el.markDate.addEventListener('change', e => {
    openMarkModal(e.target.value);
  });

  // Form Submit: Save Mark
  el.markForm.addEventListener('submit', e => {
    e.preventDefault();
    const date = el.markDate.value;
    if (!date) return;

    const sentiment = document.querySelector('input[name="sentiment"]:checked')?.value || 'Bullish';
    const markData = {
      date: date,
      sentiment: sentiment,
      pattern: el.markPattern.value,
      tradeTaken: el.markTradeTaken.value,
      outcome: el.markOutcome.value,
      levels: el.markLevels.value.trim(),
      predBias: el.markPredBias ? el.markPredBias.value : '',
      predTarget: el.markPredTarget ? el.markPredTarget.value.trim() : '',
      isEvent: el.markEventCheck ? el.markEventCheck.checked : false,
      notes: el.markNotes.value.trim(),
      tags: el.markTags.value.trim(),
      savedAt: new Date().toISOString(),
    };

    state.marks[date] = markData;
    saveMarksToStorage(true);
    updateChartMarkers();
    applyFilter(state.activeFilter);
    closeMarkModal();
    showToast(`✅ Trading mark saved for ${date}!`);
  });

  // Delete Mark
  el.btnDeleteMark.addEventListener('click', () => {
    const date = el.markDate.value;
    if (!date || !state.marks[date]) return;
    if (confirm(`Are you sure you want to delete your trading mark for ${date}?`)) {
      delete state.marks[date];
      saveMarksToStorage(true);
      updateChartMarkers();
      applyFilter(state.activeFilter);
      closeMarkModal();
      showToast(`🗑️ Trading mark removed for ${date}`);
    }
  });

  // ========================================================================
  // Glossary Modal & Search Handlers
  // ========================================================================
  if (el.btnOpenGlossary) {
    el.btnOpenGlossary.addEventListener('click', () => {
      renderGlossary('');
      if (el.glossarySearch) el.glossarySearch.value = '';
      el.glossaryModal.style.display = 'flex';
    });
  }

  if (el.btnCloseGlossaryModal) {
    el.btnCloseGlossaryModal.addEventListener('click', () => {
      el.glossaryModal.style.display = 'none';
    });
  }

  if (el.glossaryModal) {
    el.glossaryModal.addEventListener('click', e => {
      if (e.target === el.glossaryModal) el.glossaryModal.style.display = 'none';
    });
  }

  if (el.glossarySearch) {
    el.glossarySearch.addEventListener('input', e => {
      renderGlossary(e.target.value);
    });
  }

  // ========================================================================
  // GitHub Sync Modal Handlers
  // ========================================================================
  el.btnGithubSync.addEventListener('click', () => {
    el.ghRepoInput.value = state.github.repo;
    el.ghTokenInput.value = state.github.token;
    el.ghAutoSyncToggle.checked = state.github.autoSync;
    el.btnDisconnectGithub.style.display = state.github.token ? 'inline-block' : 'none';
    el.syncStatusMsg.innerHTML = state.github.token ? `<span style="color:var(--green)">Connected to repository ${state.github.repo}</span>` : '<span>Not connected to GitHub</span>';
    el.syncModal.style.display = 'flex';
  });

  el.btnCloseSyncModal.addEventListener('click', () => {
    el.syncModal.style.display = 'none';
  });

  el.syncModal.addEventListener('click', e => {
    if (e.target === el.syncModal) el.syncModal.style.display = 'none';
  });

  el.btnSaveGhToken.addEventListener('click', async () => {
    const repo = el.ghRepoInput.value.trim();
    const token = el.ghTokenInput.value.trim();
    const autoSync = el.ghAutoSyncToggle.checked;

    if (!repo) {
      alert('Please enter your GitHub repository (e.g. sagnikseal/nifty-tracker)');
      return;
    }

    state.github.repo = repo;
    state.github.token = token;
    state.github.autoSync = autoSync;

    localStorage.setItem('nifty_gh_repo', repo);
    localStorage.setItem('nifty_gh_token', token);
    localStorage.setItem('nifty_gh_autosync', String(autoSync));

    updateSyncButtonUI();
    el.btnDisconnectGithub.style.display = token ? 'inline-block' : 'none';

    if (token) {
      el.syncStatusMsg.innerHTML = '<span style="color:var(--yellow)">Testing GitHub connection...</span>';
      try {
        await getFileShaFromGitHub(repo, token);
        el.syncStatusMsg.innerHTML = `<span style="color:var(--green)">✅ Connected successfully to ${repo}!</span>`;
        showToast('Connected to GitHub!');
      } catch (err) {
        el.syncStatusMsg.innerHTML = `<span style="color:var(--yellow)">⚠️ Connected to repo: ${err.message}</span>`;
      }
    } else {
      el.syncStatusMsg.innerHTML = '<span>Saved. Operating in local-only mode.</span>';
    }
  });

  el.btnPushToGithub.addEventListener('click', () => pushMarksToGitHub(false));
  if (el.btnPullFromGithub) {
    el.btnPullFromGithub.addEventListener('click', () => pullMarksFromGitHub());
  }

  el.btnDisconnectGithub.addEventListener('click', () => {
    if (confirm('Disconnect GitHub token? Your marks will remain stored safely locally.')) {
      state.github.token = '';
      localStorage.removeItem('nifty_gh_token');
      el.ghTokenInput.value = '';
      el.btnDisconnectGithub.style.display = 'none';
      updateSyncButtonUI();
      el.syncStatusMsg.innerHTML = '<span>Disconnected. Operating locally.</span>';
      showToast('GitHub token disconnected.');
    }
  });

  // Export Journal JSON
  el.btnExportJournal.addEventListener('click', () => {
    const count = Object.keys(state.marks).length;
    if (count === 0) {
      alert('You have no saved trading marks yet. Add some marks first!');
      return;
    }
    const exportData = {
      version: '1.0',
      exportedAt: new Date().toISOString(),
      totalMarks: count,
      marks: state.marks,
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `nifty_trading_journal_${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
    showToast(`Exported ${count} trading marks to JSON.`);
  });

  // Import Journal JSON
  el.btnImportJournal.addEventListener('click', () => {
    el.importFileInput.click();
  });

  el.importFileInput.addEventListener('change', e => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = event => {
      try {
        const parsed = JSON.parse(event.target.result);
        const importedMarks = parsed.marks || parsed;
        let addedCount = 0;
        for (const [dt, mark] of Object.entries(importedMarks)) {
          if (dt && typeof mark === 'object') {
            state.marks[dt] = mark;
            addedCount++;
          }
        }
        saveMarksToStorage(true);
        updateChartMarkers();
        applyFilter(state.activeFilter);
        showToast(`✅ Successfully imported ${addedCount} trading marks!`);
      } catch (err) {
        alert('Invalid JSON file format. Please upload a valid journal JSON backup.');
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  });

  // Download CSV
  el.btnDownloadCsv.addEventListener('click', () => {
    if (!state.history.length) return;
    const headers = ['Date', 'Day', 'Open', 'High', 'Low', 'Close', 'Range', 'Change', 'Change_Pct', 'Pivot', 'CPR_Width', 'Structure_Event', 'Zone'];
    const rows = state.history.map(d => [
      d.date,
      d.day,
      d.open,
      d.high,
      d.low,
      d.close,
      d.range,
      d.change,
      d.change_pct,
      d.pivot,
      d.cpr_width,
      d.smc_structure || '',
      d.zone || ''
    ]);
    const csvContent = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `nifty50_daily_history_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  });
}

// ==========================================================================
// Initialization
// ==========================================================================
document.addEventListener('DOMContentLoaded', () => {
  document.body.className = `theme-${state.activeTheme}`;

  updateMarketStatus();
  setInterval(updateMarketStatus, 60000);

  initTermTooltips();
  loadMarks();
  setupEventListeners();
  loadNiftyData();
});
