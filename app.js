/**
 * NIFTY 50 Daily High-Low Tracker & Trading Journal
 * Handles data loading, Lightweight Charts rendering, local markings, table filters, and theme toggling.
 */

// Application State
const state = {
  data: null,
  history: [],          // Chronological list
  displayHistory: [],   // Reverse chronological for table
  filteredHistory: [],
  marks: {},            // Saved in localStorage: { [date]: { ... } }
  activeFilter: 'all',  // 'all', 'marked', 'inside'
  searchQuery: '',
  currentPage: 1,
  pageSize: 20,
  chart: null,
  candleSeries: null,
  activeTheme: localStorage.getItem('nifty_theme') || 'dark',
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
  filterInside: document.getElementById('filter-inside'),
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
  markNotes: document.getElementById('mark-notes'),
  markTags: document.getElementById('mark-tags'),
  toast: document.getElementById('toast'),
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
  // Current UTC time + 5.5 hours for IST
  const now = new Date();
  const utc = now.getTime() + (now.getTimezoneOffset() * 60000);
  const ist = new Date(utc + (3600000 * 5.5));
  
  const day = ist.getDay(); // 0 is Sunday, 6 is Saturday
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
// Local Storage: Trading Marks Management
// ==========================================================================
function loadMarksFromStorage() {
  try {
    const raw = localStorage.getItem('nifty_user_marks');
    if (raw) {
      state.marks = JSON.parse(raw);
    } else {
      state.marks = {};
    }
  } catch (e) {
    console.error('Failed to load marks from localStorage:', e);
    state.marks = {};
  }
  updateMarkedCount();
}

function saveMarksToStorage() {
  try {
    localStorage.setItem('nifty_user_marks', JSON.stringify(state.marks));
    updateMarkedCount();
  } catch (e) {
    console.error('Failed to save marks to localStorage:', e);
  }
}

function updateMarkedCount() {
  const count = Object.keys(state.marks).length;
  el.markedCount.textContent = count;
}

// ==========================================================================
// Data Fetching
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
    state.displayHistory = [...state.history].reverse(); // Latest day first
    state.filteredHistory = [...state.displayHistory];
    
    renderHeroMetrics();
    renderChart();
    applyFilter(state.activeFilter);
    el.lastSyncText.textContent = `Sync: ${data.last_updated || 'Updated'}`;
  } catch (err) {
    console.error('Error loading NIFTY data:', err);
    el.tableBody.innerHTML = `
      <tr>
        <td colspan="10" class="text-center" style="padding: 2rem; color: var(--red);">
          ⚠️ Could not load <code>data/nifty_data.json</code>. Please verify that the data file exists or trigger the GitHub Action.
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

function round(val, dec = 2) {
  return Number(Math.round(val + 'e' + dec) + 'e-' + dec);
}

// ==========================================================================
// Lightweight Chart
// ==========================================================================
function renderChart() {
  if (!window.LightweightCharts || !el.tvChart) return;

  // Clear previous instance
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
      scaleMargins: {
        top: 0.1,
        bottom: 0.1,
      },
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

  // Convert history data to chart format
  const chartData = state.history.map(item => ({
    time: item.date,
    open: item.open,
    high: item.high,
    low: item.low,
    close: item.close,
  }));

  state.candleSeries.setData(chartData);
  updateChartMarkers();

  // Set default view to last 90 days
  setTimeframeDays(90);

  // Crosshair move handler
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

  // Responsive resize
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

  // Markers must be sorted chronologically
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

  // Update filter buttons UI
  el.filterAll.classList.toggle('active', filterName === 'all');
  el.filterMarked.classList.toggle('active', filterName === 'marked');
  el.filterInside.classList.toggle('active', filterName === 'inside');

  let list = [...state.displayHistory];

  if (filterName === 'marked') {
    list = list.filter(item => Boolean(state.marks[item.date]));
  } else if (filterName === 'inside') {
    list = list.filter(item => item.is_inside_day);
  }

  // Apply search query if present
  if (state.searchQuery.trim() !== '') {
    const q = state.searchQuery.toLowerCase().trim();
    list = list.filter(item => {
      const mark = state.marks[item.date];
      const matchDate = item.date.includes(q);
      const matchNotes = mark?.notes?.toLowerCase().includes(q);
      const matchPattern = (mark?.pattern || item.candle_type)?.toLowerCase().includes(q);
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
        <td colspan="10" class="text-center" style="padding: 2.5rem; color: var(--text-muted);">
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

    // Badges
    const insideBadge = item.is_inside_day ? `<span class="pill-badge badge-inside" title="High <= Prev High, Low >= Prev Low">Inside Bar</span>` : '';
    const outsideBadge = item.is_outside_day ? `<span class="pill-badge badge-outside" title="High >= Prev High, Low <= Prev Low">Outside Bar</span>` : '';
    const gapBadge = item.gap_type === 'Gap Up' ? `<span class="pill-badge badge-gap-up">Gap Up (${item.gap})</span>`
                   : (item.gap_type === 'Gap Down' ? `<span class="pill-badge badge-gap-down">Gap Down (${item.gap})</span>` : '');

    // User Mark Cell
    let markCellHtml = '';
    if (mark) {
      const sentimentClass = `sentiment-${mark.sentiment || 'Neutral'}`;
      const tagList = mark.tags ? mark.tags.split(',').map(t => `<span class="mark-tag-pill">${t.trim()}</span>`).join(' ') : '';
      markCellHtml = `
        <div class="mark-cell-content" onclick="openMarkModal('${item.date}')" style="cursor: pointer;" title="Click to view/edit mark">
          <span class="badge-sentiment ${sentimentClass}">${mark.sentiment || 'Marked'}</span>
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
        <td>${markCellHtml}</td>
        <td class="text-center">
          <button class="btn btn-sm btn-outline" onclick="openMarkModal('${item.date}')" title="Edit Trading Mark">
            ✍️
          </button>
        </td>
      </tr>
    `;
  }).join('');

  // Update Pagination Controls
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
  // If no date passed, default to latest trading day
  const targetDate = dateStr || state.data?.latest?.date || new Date().toISOString().split('T')[0];
  el.markDate.value = targetDate;
  el.modalDateSubtitle.textContent = `Date: ${targetDate}`;

  // Find record in history
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

  // Populate existing mark if present
  const existing = state.marks[targetDate];
  if (existing) {
    // Radio sentiment
    const radio = document.querySelector(`input[name="sentiment"][value="${existing.sentiment || 'Bullish'}"]`);
    if (radio) radio.checked = true;

    el.markPattern.value = existing.pattern || 'Trend Day (Unidirectional)';
    el.markTradeTaken.value = existing.tradeTaken || 'No';
    el.markOutcome.value = existing.outcome || 'N/A';
    el.markLevels.value = existing.levels || '';
    el.markNotes.value = existing.notes || '';
    el.markTags.value = existing.tags || '';
    el.btnDeleteMark.style.display = 'inline-block';
  } else {
    // Reset to defaults
    const defaultRadio = document.querySelector('input[name="sentiment"][value="Bullish"]');
    if (defaultRadio) defaultRadio.checked = true;
    el.markPattern.value = record?.is_inside_day ? 'Inside Bar Setup (Harami)' : 'Trend Day (Unidirectional)';
    el.markTradeTaken.value = 'No';
    el.markOutcome.value = 'N/A';
    el.markLevels.value = record ? `PDH: ${record.high}, PDL: ${record.low}` : '';
    el.markNotes.value = '';
    el.markTags.value = '';
    el.btnDeleteMark.style.display = 'none';
  }

  el.markModal.style.display = 'flex';
}

function closeMarkModal() {
  el.markModal.style.display = 'none';
}

function appendLevel(text) {
  const current = el.markLevels.value.trim();
  if (current.length > 0 && !current.endsWith(',')) {
    el.markLevels.value = `${current}, ${text}`;
  } else {
    el.markLevels.value = `${current} ${text}`.trim();
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

  // Date change inside modal updates snapshot
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
      notes: el.markNotes.value.trim(),
      tags: el.markTags.value.trim(),
      savedAt: new Date().toISOString(),
    };

    state.marks[date] = markData;
    saveMarksToStorage();
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
      saveMarksToStorage();
      updateChartMarkers();
      applyFilter(state.activeFilter);
      closeMarkModal();
      showToast(`🗑️ Trading mark removed for ${date}`);
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
        saveMarksToStorage();
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
    const headers = ['Date', 'Day', 'Open', 'High', 'Low', 'Close', 'Range', 'Change', 'Change_Pct', 'Pivot', 'CPR_Width', 'Inside_Bar'];
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
      d.is_inside_day ? 'YES' : 'NO'
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
  // Set theme from storage
  document.body.className = `theme-${state.activeTheme}`;

  updateMarketStatus();
  setInterval(updateMarketStatus, 60000); // Check market status every 1 min

  loadMarksFromStorage();
  setupEventListeners();
  loadNiftyData();
});

