/**
 * NIFTY 50 Historical EOD SMC & CPR Dashboard Controller (`app.js`)
 * =================================================================
 * Loads `data/nifty_data.json`, populates the hero EOD cards, renders the interactive
 * HTML5 Candlestick + Central Pivot Range (CPR) + PDH/PDL Sweep canvas chart,
 * manages the filterable historical table, and persists trade journal annotations.
 */

(function () {
  'use strict';

  const appState = {
    dataset: null,
    entries: [],
    latestEntry: null,
    chartRange: 60,
    journalMarks: JSON.parse(localStorage.getItem('nifty_trading_marks') || '{}'),
  };

  window.niftyApp = appState;

  function fmtNum(val, digits = 2) {
    if (val === undefined || val === null || isNaN(Number(val))) return '--';
    return Number(val).toLocaleString('en-IN', {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
  }

  async function loadNiftyDataset() {
    try {
      const res = await fetch('data/nifty_data.json', { cache: 'no-store' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      appState.dataset = data;
      appState.entries = Array.isArray(data.data) ? data.data : [];
      appState.latestEntry = data.latest || appState.entries[0] || null;
      renderHeroCards();
      renderCandlestickCanvas();
      renderHistoricalTable();
    } catch (err) {
      console.warn('Could not fetch data/nifty_data.json:', err);
    }
  }

  function renderHeroCards() {
    const latest = appState.latestEntry;
    if (!latest) return;
    const meta = (appState.dataset && appState.dataset.meta) || {};

    const banner = document.getElementById('last-updated-banner');
    if (banner && appState.dataset) {
      banner.textContent = `NSE NIFTY 50 • Last EOD Sync: ${appState.dataset.last_updated || latest.date} • 52W Range: ${fmtNum(meta.fifty_two_week_high, 0)} - ${fmtNum(meta.fifty_two_week_low, 0)}`;
    }

    const setEl = (id, text) => {
      const el = document.getElementById(id);
      if (el) el.textContent = text;
    };

    setEl('val-day-high', fmtNum(latest.high));
    setEl('sub-day-high', `Prev High (PDH): ${fmtNum(latest.prev_high)} | Upper Wick: ${fmtNum(latest.upper_wick, 1)} pts`);
    setEl(
      'diff-prev-high',
      latest.pdh_sweep ? 'PDH SWEEP TRAP' : latest.bullish_bos ? 'BULLISH BOS' : 'Inside PDH'
    );

    setEl('val-day-low', fmtNum(latest.low));
    setEl('sub-day-low', `Prev Low (PDL): ${fmtNum(latest.prev_low)} | Lower Wick: ${fmtNum(latest.lower_wick, 1)} pts`);
    setEl(
      'diff-prev-low',
      latest.pdl_sweep ? 'PDL SPRING SWEEP' : latest.bearish_bos ? 'BEARISH BOS' : 'Above PDL'
    );

    setEl('val-day-range', `${fmtNum(latest.range, 1)} pts`);
    setEl('sub-day-range', `20D ADR: ${fmtNum(meta.adr_20, 1)} pts | 5D ADR: ${fmtNum(meta.adr_5, 1)} pts`);
    setEl('pill-range-type', latest.is_expansion_bar ? 'EXPANSION BAR' : latest.is_pinbar ? 'PINBAR REJECTION' : `${latest.range_pct}% Range`);

    const chgPrefix = latest.change >= 0 ? '+' : '';
    setEl('val-day-close', fmtNum(latest.close));
    setEl('sub-day-close', `${chgPrefix}${fmtNum(latest.change)} (${chgPrefix}${latest.change_pct}%) • ${latest.smc_structure}`);
    setEl('pill-smc-zone', `${latest.zone} Zone (Eq: ${fmtNum(latest.equilibrium, 0)})`);

    setEl('val-cpr-pivot', fmtNum(latest.pivot));
    setEl('sub-cpr-band', `TC: ${fmtNum(latest.tc)} | BC: ${fmtNum(latest.bc)} | R1: ${fmtNum(latest.r1)} | S1: ${fmtNum(latest.s1)}`);
    setEl('pill-cpr-type', `Width: ${fmtNum(latest.cpr_width, 1)} pts`);
  }

  function renderCandlestickCanvas() {
    const canvas = document.getElementById('nifty-ohlc-canvas');
    if (!canvas || !appState.entries.length) return;
    const ctx = canvas.getContext('2d');
    const slice = appState.entries.slice(0, appState.chartRange).slice().reverse();

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const padLeft = 18;
    const padRight = 72;
    const padTop = 20;
    const padBot = 28;
    const plotW = w - padLeft - padRight;
    const plotH = h - padTop - padBot;

    const maxHigh = Math.max(...slice.map((d) => d.high));
    const minLow = Math.min(...slice.map((d) => d.low));
    const span = Math.max(maxHigh - minLow, 10);

    const yOf = (price) => padTop + plotH - ((price - minLow) / span) * plotH;

    // Horizontal grid lines & price labels
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.14)';
    ctx.fillStyle = '#94a3b8';
    ctx.font = '11px monospace';
    ctx.lineWidth = 1;

    for (let i = 0; i <= 5; i++) {
      const p = minLow + (span * i) / 5;
      const y = yOf(p);
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(w - padRight, y);
      ctx.stroke();
      ctx.fillText(p.toFixed(0), w - padRight + 8, y + 4);
    }

    const stepX = plotW / Math.max(slice.length, 1);
    const bodyW = Math.max(3, Math.min(14, stepX * 0.64));

    // Draw CPR Pivot Ribbon connecting sessions
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.55)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    slice.forEach((d, idx) => {
      const x = padLeft + idx * stepX + stepX * 0.5;
      const y = yOf(d.pivot);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw Candlesticks & Sweep Markers
    slice.forEach((d, idx) => {
      const x = padLeft + idx * stepX + stepX * 0.5;
      const yH = yOf(d.high);
      const yL = yOf(d.low);
      const yO = yOf(d.open);
      const yC = yOf(d.close);
      const isBull = d.close >= d.open;

      ctx.strokeStyle = isBull ? '#10b981' : '#f43f5e';
      ctx.fillStyle = isBull ? '#10b981' : '#f43f5e';
      ctx.lineWidth = 1.3;

      ctx.beginPath();
      ctx.moveTo(x, yH);
      ctx.lineTo(x, yL);
      ctx.stroke();

      const topY = Math.min(yO, yC);
      const boxH = Math.max(Math.abs(yC - yO), 2);
      ctx.fillRect(x - bodyW / 2, topY, bodyW, boxH);

      // Highlight PDH/PDL Liquidity Sweeps with a diamond dot
      if (d.pdh_sweep || d.pdl_sweep) {
        ctx.fillStyle = '#f59e0b';
        ctx.beginPath();
        ctx.arc(x, d.pdh_sweep ? yH - 5 : yL + 5, 3.2, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    canvas.onmousemove = (ev) => {
      const rect = canvas.getBoundingClientRect();
      const relX = ((ev.clientX - rect.left) / rect.width) * w;
      const idx = Math.floor((relX - padLeft) / stepX);
      if (idx >= 0 && idx < slice.length) {
        const d = slice[idx];
        const readout = document.getElementById('chart-hover-readout');
        if (readout) {
          readout.textContent = `${d.date} (${d.day.slice(0, 3)}) • O:${d.open} H:${d.high} L:${d.low} C:${d.close} (${d.change >= 0 ? '+' : ''}${d.change}) • CPR:${d.pivot} • ${d.smc_structure}`;
        }
      }
    };
  }

  function renderHistoricalTable() {
    const tbody = document.getElementById('historical-table-body');
    if (!tbody) return;

    const searchInput = document.getElementById('table-search-input');
    const structFilter = document.getElementById('table-structure-filter');
    const query = (searchInput ? searchInput.value : '').trim().toLowerCase();
    const sType = structFilter ? structFilter.value : 'ALL';

    const filtered = appState.entries.filter((row) => {
      if (sType === 'Sweep' && !(row.pdh_sweep || row.pdl_sweep)) return false;
      if (sType === 'BOS' && !(row.bullish_bos || row.bearish_bos)) return false;
      if (sType === 'Narrow' && !String(row.cpr_type).includes('Narrow')) return false;
      if (sType === 'Pinbar' && !(row.is_pinbar || row.is_expansion_bar)) return false;

      if (!query) return true;
      const hay = `${row.date} ${row.day} ${row.smc_structure} ${row.cpr_type} ${row.gap_type} ${row.zone}`.toLowerCase();
      return hay.includes(query);
    });

    tbody.innerHTML = filtered
      .slice(0, 120)
      .map((r) => {
        const isPos = r.change >= 0;
        const chgCls = isPos ? 'text-success' : 'text-danger';
        const noteObj = appState.journalMarks[r.date];
        const noteTxt = noteObj ? `${noteObj.setup}: ${noteObj.note}` : '';
        return `
          <tr>
            <td style="font-family:var(--font-mono);font-weight:700;">${r.date}</td>
            <td>${r.day.slice(0, 3)}</td>
            <td style="font-family:var(--font-mono);">${fmtNum(r.open)}</td>
            <td style="font-family:var(--font-mono);">${fmtNum(r.high)}</td>
            <td style="font-family:var(--font-mono);">${fmtNum(r.low)}</td>
            <td style="font-family:var(--font-mono);font-weight:700;">${fmtNum(r.close)}</td>
            <td class="${chgCls}" style="font-family:var(--font-mono);font-weight:700;">
              ${isPos ? '+' : ''}${fmtNum(r.change)} (${isPos ? '+' : ''}${r.change_pct}%)
            </td>
            <td style="font-family:var(--font-mono);">${fmtNum(r.range, 1)} (${r.clq})</td>
            <td><span class="badge-pill">${r.smc_structure}</span></td>
            <td style="font-family:var(--font-mono);">${fmtNum(r.pivot)} (${fmtNum(r.cpr_width, 1)}w)</td>
            <td style="font-family:var(--font-mono);">${fmtNum(r.r1, 0)} / ${fmtNum(r.s1, 0)}</td>
            <td>${r.zone}</td>
            <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;">${noteTxt}</td>
          </tr>
        `;
      })
      .join('');
  }

  function bindDashboardEvents() {
    // Chart range buttons
    document.querySelectorAll('.chart-range-btns button').forEach((btn) => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.chart-range-btns button').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        appState.chartRange = Number(btn.getAttribute('data-range') || 60);
        renderCandlestickCanvas();
      });
    });

    const searchInput = document.getElementById('table-search-input');
    const structFilter = document.getElementById('table-structure-filter');
    if (searchInput) searchInput.addEventListener('input', renderHistoricalTable);
    if (structFilter) structFilter.addEventListener('change', renderHistoricalTable);

    // Theme toggle
    const btnTheme = document.getElementById('btn-theme-toggle');
    if (btnTheme) {
      btnTheme.addEventListener('click', () => {
        const cur = document.documentElement.getAttribute('data-theme') || 'dark';
        const next = cur === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', next);
      });
    }

    // Journal Modal
    const jModal = document.getElementById('journal-modal');
    const btnQuickMark = document.getElementById('btn-quick-mark');
    const btnCloseJournal = document.getElementById('btn-close-journal');
    const btnSaveJournal = document.getElementById('btn-save-journal');

    if (btnQuickMark && jModal) {
      btnQuickMark.addEventListener('click', () => {
        const dateEl = document.getElementById('journal-date');
        if (dateEl && appState.latestEntry) dateEl.value = appState.latestEntry.date;
        jModal.classList.remove('hidden');
      });
    }
    if (btnCloseJournal && jModal) {
      btnCloseJournal.addEventListener('click', () => jModal.classList.add('hidden'));
    }
    if (btnSaveJournal && jModal) {
      btnSaveJournal.addEventListener('click', () => {
        const d = document.getElementById('journal-date').value;
        const s = document.getElementById('journal-setup').value;
        const n = document.getElementById('journal-note').value;
        if (d) {
          appState.journalMarks[d] = { setup: s, note: n, updated: new Date().toISOString() };
          localStorage.setItem('nifty_trading_marks', JSON.stringify(appState.journalMarks));
          renderHistoricalTable();
        }
        jModal.classList.add('hidden');
      });
    }

    // Glossary Modal
    const gModal = document.getElementById('glossary-modal');
    const btnOpenGlossary = document.getElementById('btn-open-glossary');
    const btnCloseGlossary = document.getElementById('btn-close-glossary');
    if (btnOpenGlossary && gModal) {
      btnOpenGlossary.addEventListener('click', () => gModal.classList.remove('hidden'));
    }
    if (btnCloseGlossary && gModal) {
      btnCloseGlossary.addEventListener('click', () => gModal.classList.add('hidden'));
    }

    // Export / Import Journal JSON
    const btnExport = document.getElementById('btn-export-journal');
    const btnImport = document.getElementById('btn-import-journal');
    const fileInput = document.getElementById('import-file-input');

    if (btnExport) {
      btnExport.addEventListener('click', () => {
        const blob = new Blob([JSON.stringify(appState.journalMarks, null, 2)], {
          type: 'application/json',
        });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'nifty_trading_journal.json';
        a.click();
      });
    }
    if (btnImport && fileInput) {
      btnImport.addEventListener('click', () => fileInput.click());
      fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (ev) => {
          try {
            const parsed = JSON.parse(ev.target.result);
            Object.assign(appState.journalMarks, parsed);
            localStorage.setItem('nifty_trading_marks', JSON.stringify(appState.journalMarks));
            renderHistoricalTable();
          } catch (_) {}
        };
        reader.readAsText(file);
      });
    }
  }

  document.addEventListener('DOMContentLoaded', () => {
    bindDashboardEvents();
    loadNiftyDataset();
  });
})();
