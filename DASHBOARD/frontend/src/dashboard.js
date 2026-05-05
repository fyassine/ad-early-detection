import { Chart } from 'chart.js';
import { $, diagColor, BAR_COLORS, showLoading, hideLoading, tooltipStyle } from './utils.js';
import { state } from './state.js';
import { renderTable, renderRows } from './table.js';
import { getRecent, saveRecent } from './config.js';

// ── Analyze ───────────────────────────────────────────────────────────────────

export async function analyze() {
    const csvPath = $('csvSelect').value;
    state.activeFilter = null;
    showLoading('Analyzing…');
    state._cohortStatsCache.clear();
    Object.keys(state._cohortRefMatrices).forEach(k => delete state._cohortRefMatrices[k]);
    try {
        state.lastScan = null; state.globalMeta = null; state.filteredMeta = null;
        if (state.selectedScanFolders.length) {
            $('loadingText').textContent = 'Scanning folders…';
            const r = await fetch(`/api/scan?folders=${encodeURIComponent(state.selectedScanFolders.join(','))}`);
            state.lastScan = await r.json();
            state.scanSubjects = new Set(Object.keys(state.lastScan.subject_scan_counts || {}));
        }
        if (csvPath) {
            $('loadingText').textContent = 'Parsing metadata…';
            const p = new URLSearchParams({ csv_path: csvPath });
            if (state.selectedScanFolders.length) p.set('scan_folders', state.selectedScanFolders.join(','));
            const r = await fetch(`/api/metadata?${p}`);
            state.globalMeta = await r.json();
            state.filteredMeta = state.globalMeta;
        }
        if (csvPath || state.selectedScanFolders.length) {
            const existing = getRecent().some(r =>
                r.csv === csvPath && JSON.stringify(r.folders) === JSON.stringify(state.selectedScanFolders));
            if (!existing) saveRecent(csvPath, state.selectedScanFolders);
        }
        render(state.lastScan, state.globalMeta, state.filteredMeta);

        if (csvPath && state.selectedScanFolders.length) {
            fetch(`/api/cohort/warmup?csv_path=${encodeURIComponent(csvPath)}` +
                  `&scan_folders=${encodeURIComponent(state.selectedScanFolders.join(','))}`).catch(() => {});
        }
    } catch (e) { console.error(e); alert('Analysis failed'); }
    hideLoading();
}

// ── Render all dashboard sections ─────────────────────────────────────────────

export function render(scan, global, filtered) {
    Object.values(state.activeCharts).forEach(c => c.destroy());
    state.activeCharts = {};
    $('emptyState').style.display = 'none';
    $('dashboardResults').classList.add('active');

    if (global?.diagnosis_distribution) {
        const labels = Object.keys(global.diagnosis_distribution);
        $('globalCohortSelect').innerHTML = '<option value="">— All Patients —</option>' +
            labels.map(l => `<option value="${l}">${l} (${global.diagnosis_distribution[l]})</option>`).join('');
        if (state.activeFilter?.field === 'diagnosis') $('globalCohortSelect').value = state.activeFilter.value;
        $('dashboardControls').style.display = 'flex';
    } else {
        $('dashboardControls').style.display = 'none';
    }

    renderSummary(scan, global);
    renderDemo(filtered);
    renderDiag(global, scan);
    renderScans(scan, filtered);
    renderClinical(filtered);
    renderTable(filtered, scan);
}

// ── Summary cards ─────────────────────────────────────────────────────────────

export function renderSummary(scan, meta) {
    const cards = [];
    if (meta) {
        const rows = meta.total_rows ? `${meta.total_rows} rows` : null;
        cards.push({ icon: '👥', value: meta.unique_subjects || '—', label: 'Subjects (CSV)', sub: rows });
    }
    if (scan) {
        cards.push({ icon: '🧲', value: scan.total_scans, label: 'Files on Disk' });
        cards.push({ icon: '📁', value: scan.total_subjects, label: 'Subjects w/ Scans' });
        const visitCount = scan.visit_distribution
            ? Object.values(scan.visit_distribution).reduce((a, b) => a + b, 0) : null;
        if (visitCount !== null) cards.push({ icon: '🗓️', value: visitCount, label: 'Visits on Disk' });
        if (scan.longitudinal_subjects > 0) cards.push({ icon: '🔄', value: scan.longitudinal_subjects, label: 'Longitudinal' });
        if (scan.format_info?.description) {
            const formatLabel = scan.format_info.description.split('·')[0].trim();
            const formatSub = scan.format_info.parcellation || scan.format_info.type || null;
            cards.push({ icon: '📦', value: formatLabel, label: 'Format', sub: formatSub, kind: 'text' });
        }
    }
    if (meta?.scan_coverage) {
        const c = meta.scan_coverage;
        const pct = c.metadata_subjects > 0 ? Math.round(c.matched / c.metadata_subjects * 100) : 0;
        cards.push({ icon: '🎯', value: `${pct}%`, label: 'Scan Coverage' });
    }
    if (meta?.age_stats) cards.push({ icon: '📅', value: `${meta.age_stats.mean}±${meta.age_stats.std}`, label: 'Age (mean±SD)' });
    $('summaryCards').innerHTML = cards.map(c => {
        const valueClass = c.kind === 'text' ? 'card-value is-text' : 'card-value';
        const sub = c.sub ? `<div class="card-sub">${c.sub}</div>` : '';
        return `<div class="summary-card">
            <div class="card-top">
                <div class="card-icon">${c.icon}</div>
                <div class="card-label">${c.label}</div>
            </div>
            <div class="${valueClass}">${c.value}</div>
            ${sub}
        </div>`;
    }).join('');
}

// ── Demographics ──────────────────────────────────────────────────────────────

export function renderDemo(meta) {
    const sec = $('sectionDemo'), cont = $('demoCharts');
    cont.innerHTML = '';
    if (!meta?.sex_distribution && !meta?.age_histogram) { sec.style.display = 'none'; return; }
    sec.style.display = '';
    if (meta.sex_distribution) addHBar('demoCharts', 'sex', 'Sex', meta.sex_distribution, BAR_COLORS);
    if (meta.age_histogram) addVBar('demoCharts', 'age', 'Age Distribution', meta.age_histogram);
}

// ── Diagnosis ─────────────────────────────────────────────────────────────────

export function renderDiag(meta, scan) {
    const sec = $('sectionDiag'), cont = $('diagCharts');
    cont.innerHTML = '';
    if (!meta?.diagnosis_distribution) { sec.style.display = 'none'; return; }
    sec.style.display = '';

    const labels = Object.keys(meta.diagnosis_distribution);
    const patVals = labels.map(k => meta.diagnosis_distribution[k]);
    const visitVals = meta.diagnosis_visits ? labels.map(k => meta.diagnosis_visits[k] || 0) : null;
    const scanVals = meta.diagnosis_scans ? labels.map(k => meta.diagnosis_scans[k] || 0) : null;
    const bgColors = labels.map(l => diagColor(l));
    const bgColorsDim = labels.map(l => diagColor(l) + '55');
    const bgColorsMid = labels.map(l => diagColor(l) + '99');

    const card = document.createElement('div');
    card.className = 'chart-card';
    card.innerHTML = `<h3>Diagnosis — Patients${visitVals ? ' &amp; Visits' : ''}${scanVals ? ' &amp; Scans' : ''} <span style="font-size:.65rem;color:var(--text-3);font-weight:400">(click to filter)</span></h3><div class="chart-container"><canvas id="chart-diag"></canvas></div>`;
    cont.appendChild(card);

    const datasets = [{
        label: 'Patients', data: patVals, backgroundColor: bgColorsDim,
        borderWidth: 0, borderRadius: 4, barThickness: 24, grouped: false,
    }];
    if (visitVals) datasets.push({ label: 'Visits', data: visitVals, backgroundColor: bgColorsMid, borderWidth: 0, borderRadius: 4, barThickness: 14, grouped: false });
    if (scanVals) datasets.push({ label: 'Scans', data: scanVals, backgroundColor: bgColors, borderWidth: 0, borderRadius: 4, barThickness: 8, grouped: false });

    const ctx = document.getElementById('chart-diag').getContext('2d');
    state.activeCharts['diag'] = new Chart(ctx, {
        type: 'bar',
        data: { labels, datasets },
        options: {
            indexAxis: 'y', responsive: true, maintainAspectRatio: false,
            layout: { padding: { right: 120 } },
            plugins: {
                legend: { display: !!(scanVals || visitVals), position: 'top', labels: { usePointStyle: true, boxWidth: 10, padding: 14 } },
                tooltip: tooltipStyle(),
            },
            scales: {
                x: { display: false, stacked: false },
                y: { grid: { display: false }, stacked: false, ticks: { color: '#cbd5e1', font: { size: 12, weight: 500 } } },
            },
            onClick(e, els) {
                if (!els.length) return;
                const label = labels[els[0].index];
                state.activeFilter = state.activeFilter?.value === label ? null : { field: 'diagnosis', value: label };
                const sel = $('globalCohortSelect');
                if (sel) sel.value = state.activeFilter ? state.activeFilter.value : '';
                applyFilter();
            },
            animation: { duration: 500 },
        },
        plugins: [{
            afterDatasetsDraw(chart) {
                const ctx2 = chart.ctx;
                const total = patVals.reduce((a, b) => a + b, 0);
                const idxVisits = visitVals ? 1 : null;
                const idxScans = scanVals ? (visitVals ? 2 : 1) : null;
                chart.data.datasets[0].data.forEach((v, i) => {
                    const m = chart.getDatasetMeta(0).data[i]; if (!m) return;
                    const pct = total > 0 ? Math.round(v / total * 100) : 0;
                    const mVisit = idxVisits !== null ? chart.getDatasetMeta(idxVisits).data[i] : null;
                    const mScan = idxScans !== null ? chart.getDatasetMeta(idxScans).data[i] : null;
                    const max_x = Math.max(m.x, mVisit ? mVisit.x : 0, mScan ? mScan.x : 0);
                    ctx2.save(); ctx2.textBaseline = 'middle'; ctx2.font = '500 11px Inter,sans-serif';
                    let curX = max_x + 8;
                    ctx2.fillStyle = '#94a3b8';
                    const tPat = `${v} pat. (${pct}%)`;
                    ctx2.fillText(tPat, curX, m.y); curX += ctx2.measureText(tPat).width;
                    if (visitVals) {
                        ctx2.fillStyle = '#64748b'; const sep = `  ·  `;
                        ctx2.fillText(sep, curX, m.y); curX += ctx2.measureText(sep).width;
                        ctx2.fillStyle = bgColorsMid[i];
                        const tVis = `${visitVals[i]} visits`;
                        ctx2.fillText(tVis, curX, m.y); curX += ctx2.measureText(tVis).width;
                    }
                    if (scanVals) {
                        ctx2.fillStyle = '#64748b'; const sep = `  ·  `;
                        ctx2.fillText(sep, curX, m.y); curX += ctx2.measureText(sep).width;
                        ctx2.fillStyle = bgColors[i]; ctx2.fillText(`${scanVals[i]} scans`, curX, m.y);
                    }
                    ctx2.restore();
                });
            },
        }],
    });
}

// ── Cross-filter ──────────────────────────────────────────────────────────────

export async function applyFilter() {
    renderDiagHighlight();
    if (state.activeFilter?.field === 'diagnosis') {
        showLoading(`Filtering cohort to ${state.activeFilter.value}…`);
        try {
            const p = new URLSearchParams({ csv_path: $('csvSelect').value, cohort: state.activeFilter.value });
            if (state.selectedScanFolders.length) p.set('scan_folders', state.selectedScanFolders.join(','));
            const r = await fetch(`/api/metadata?${p}`);
            state.filteredMeta = await r.json();
        } catch (e) { console.error('Filter failed', e); }
        hideLoading();
    } else {
        state.filteredMeta = state.globalMeta;
    }

    ['age', 'sex', 'scanVisits', 'scansPerSubj', 'metaVisits', 'mmse', 'cdr', 'apoe', 'split'].forEach(id => {
        if (state.activeCharts[id]) { state.activeCharts[id].destroy(); delete state.activeCharts[id]; }
    });
    renderDemo(state.filteredMeta);
    renderScans(state.lastScan, state.filteredMeta);
    renderClinical(state.filteredMeta);
    renderTable(state.filteredMeta, state.lastScan);

    let indicator = $('filterIndicator');
    if (!indicator) {
        indicator = document.createElement('div');
        indicator.id = 'filterIndicator';
        indicator.style.cssText = 'font-size:.75rem;color:var(--amber);margin-left:auto;cursor:pointer;';
        indicator.onclick = () => { state.activeFilter = null; applyFilter(); };
        document.querySelector('.table-header').appendChild(indicator);
    }
    indicator.textContent = state.activeFilter ? `Filtered: ${state.activeFilter.value} — click again to clear` : '';
    indicator.style.display = state.activeFilter ? '' : 'none';
}

export function renderDiagHighlight() {
    const chart = state.activeCharts['diag'];
    if (!chart || !state.globalMeta?.diagnosis_distribution) return;
    const labels = chart.data.labels;
    const baseColors = labels.map(l => diagColor(l));
    const patColors = labels.map((l, i) => !state.activeFilter ? baseColors[i] + '55' : l === state.activeFilter.value ? baseColors[i] + '55' : baseColors[i] + '22');
    const visitColors = labels.map((l, i) => !state.activeFilter ? baseColors[i] + '99' : l === state.activeFilter.value ? baseColors[i] + '99' : baseColors[i] + '22');
    const scanColors = labels.map((l, i) => !state.activeFilter ? baseColors[i] : l === state.activeFilter.value ? baseColors[i] : baseColors[i] + '44');
    chart.data.datasets.forEach(ds => {
        if (ds.label === 'Patients') ds.backgroundColor = patColors;
        if (ds.label === 'Visits') ds.backgroundColor = visitColors;
        if (ds.label === 'Scans') ds.backgroundColor = scanColors;
    });
    chart.update('none');
}

export function filteredRows() {
    const q = $('tableSearch').value.toLowerCase().trim();
    let rows = state.patientData;
    if (q) rows = rows.filter(r => Object.values(r).some(v => v !== null && String(v).toLowerCase().includes(q)));
    return rows;
}

// ── Scans & Visits ────────────────────────────────────────────────────────────

export function renderScans(scan, meta) {
    const sec = $('sectionScans'), cont = $('scanCharts');
    cont.innerHTML = '';
    const has = scan || meta?.visit_distribution;
    sec.style.display = has ? '' : 'none'; if (!has) return;
    if (scan?.visit_distribution) addVBar('scanCharts', 'scanVisits', 'Scans per Visit (on disk)', { labels: Object.keys(scan.visit_distribution), counts: Object.values(scan.visit_distribution) });
    if (scan?.scans_per_subject_distribution) addVBar('scanCharts', 'scansPerSubj', 'Scans per Subject', scan.scans_per_subject_distribution);
    if (meta?.visit_distribution) addVBar('scanCharts', 'metaVisits', 'Visits (from CSV)', meta.visit_distribution);
}

// ── Clinical ──────────────────────────────────────────────────────────────────

export function renderClinical(meta) {
    const sec = $('sectionClinical'), cont = $('clinicalCharts');
    cont.innerHTML = '';
    const has = meta?.mmse_histogram || meta?.cdr_distribution || meta?.apoe_distribution || meta?.split_distribution;
    sec.style.display = has ? '' : 'none'; if (!has) return;
    if (meta.mmse_histogram) addVBar('clinicalCharts', 'mmse', 'MMSE Distribution', meta.mmse_histogram);
    if (meta.cdr_distribution) addHBar('clinicalCharts', 'cdr', 'CDR Global', meta.cdr_distribution, BAR_COLORS);
    if (meta.apoe_distribution) addHBar('clinicalCharts', 'apoe', 'ApoE Genotype', meta.apoe_distribution, BAR_COLORS);
    if (meta.split_distribution) addHBar('clinicalCharts', 'split', 'Train / Val / Test', meta.split_distribution, BAR_COLORS);
}

// ── Chart builders ────────────────────────────────────────────────────────────

export function addHBar(cid, id, title, data, colors) {
    const labels = data.labels || Object.keys(data);
    const values = data.counts || Object.values(data);
    const total = values.reduce((a, b) => a + b, 0);
    const card = makeCard(cid, id, title);
    const ctx = card.querySelector('canvas').getContext('2d');
    state.activeCharts[id] = new Chart(ctx, {
        type: 'bar',
        data: { labels, datasets: [{ data: values, backgroundColor: colors ? labels.map((_, i) => colors[i % colors.length]) : labels.map(l => diagColor(l)), borderWidth: 0, borderRadius: 4, barThickness: 22 }] },
        options: {
            indexAxis: 'y', responsive: true, maintainAspectRatio: false,
            layout: { padding: { right: 50 } },
            plugins: { legend: { display: false }, tooltip: tooltipStyle() },
            scales: { x: { display: false }, y: { grid: { display: false }, ticks: { color: '#cbd5e1', font: { size: 11, weight: 500 } } } },
            animation: { duration: 500 },
        },
        plugins: [{
            afterDatasetsDraw(ch) {
                const ctx2 = ch.ctx, tot = total;
                ch.data.datasets[0].data.forEach((v, i) => {
                    const m = ch.getDatasetMeta(0).data[i]; if (!m) return;
                    const pct = tot > 0 ? Math.round(v / tot * 100) : 0;
                    ctx2.save(); ctx2.fillStyle = '#94a3b8'; ctx2.font = '500 10px Inter,sans-serif';
                    ctx2.textAlign = 'left'; ctx2.textBaseline = 'middle';
                    ctx2.fillText(`${v} (${pct}%)`, m.x + 6, m.y); ctx2.restore();
                });
            },
        }],
    });
}

export function addVBar(cid, id, title, data) {
    const labels = data.labels || Object.keys(data);
    const values = data.counts || Object.values(data);
    const card = makeCard(cid, id, title);
    const ctx = card.querySelector('canvas').getContext('2d');
    state.activeCharts[id] = new Chart(ctx, {
        type: 'bar',
        data: { labels, datasets: [{ data: values, backgroundColor: 'rgba(129,140,248,0.55)', borderColor: 'rgba(129,140,248,0.8)', borderWidth: 1, borderRadius: 3, hoverBackgroundColor: 'rgba(167,139,250,0.7)' }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: tooltipStyle() }, scales: { x: { grid: { display: false }, ticks: { maxRotation: 45, font: { size: 10 } } }, y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.03)' } } }, animation: { duration: 500 } },
    });
}

export function makeCard(cid, id, title) {
    const card = document.createElement('div');
    card.className = 'chart-card';
    card.innerHTML = `<h3>${title}</h3><div class="chart-container"><canvas id="chart-${id}"></canvas></div>`;
    document.getElementById(cid).appendChild(card);
    return card;
}
