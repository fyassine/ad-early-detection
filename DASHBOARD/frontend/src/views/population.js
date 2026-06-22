// Population view — multi-cohort aggregation, epidemiology overlay,
// network-disruption atlas and GELSTM model card. Phase 1 fills in the
// summary, epidemiology table and network heatmap. Phase 2 wires the
// model card to the CLASSIFIER GELSTM ensemble.
import { Chart } from 'chart.js';
import { $ } from '../utils.js';
import { state } from '../state.js';

let _summaryCache = null;
let _atlasCache = null;
let _epiCache = null;
let _modelCardCache = null;
let _activeCharts = {};

export async function renderPopulation() {
    const empty = $('populationEmpty');
    const content = $('populationContent');
    if (!empty || !content) return;

    const csv = state.globalMeta && document.getElementById('csvSelect')?.value;
    const folders = (state.selectedScanFolders || []).join(',');
    const hasData = !!csv;

    empty.style.display = hasData ? 'none' : '';
    content.style.display = hasData ? '' : 'none';
    if (!hasData) return;

    if (!content.dataset.scaffolded) {
        content.innerHTML = scaffoldHtml();
        content.dataset.scaffolded = '1';
        wireSectionToggles(content);
    }

    // Fetch all four payloads in parallel. They are independent so a
    // failure in one shouldn't block the others.
    const [summary, epi, atlas, modelCard] = await Promise.all([
        fetchJson(`/api/population/summary?csv_path=${encodeURIComponent(csv)}&scan_folders=${encodeURIComponent(folders)}`)
            .catch(() => null),
        fetchJson('/api/population/epidemiology').catch(() => null),
        folders
            ? fetchJson(`/api/population/network-atlas?csv_path=${encodeURIComponent(csv)}&scan_folders=${encodeURIComponent(folders)}`).catch(() => null)
            : Promise.resolve(null),
        fetchJson('/api/population/model-card').catch(() => null),
    ]);

    _summaryCache = summary;
    _atlasCache = atlas;
    _epiCache = epi;
    _modelCardCache = modelCard;

    Object.values(_activeCharts).forEach(c => { try { c.destroy(); } catch {} });
    _activeCharts = {};

    renderSummarySection(summary);
    renderEpidemiologySection(epi, summary);
    renderNetworkAtlasSection(atlas);
    renderModelCardSection(modelCard);
}

export function resetPopulation() {
    const content = $('populationContent');
    const empty = $('populationEmpty');
    Object.values(_activeCharts).forEach(c => { try { c.destroy(); } catch {} });
    _activeCharts = {};
    if (content) {
        content.innerHTML = '';
        content.style.display = 'none';
        delete content.dataset.scaffolded;
    }
    if (empty) empty.style.display = '';
    _summaryCache = _atlasCache = _epiCache = _modelCardCache = null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Scaffold (one-time HTML; sections fill via render*Section helpers)
// ─────────────────────────────────────────────────────────────────────────────
function scaffoldHtml() {
    return `
        <section class="summary-cards" id="popSummaryCards"></section>

        <section class="dash-section" id="popSummarySection">
            <div class="section-header">
                <span class="section-icon">📊</span>
                <h2>Multi-cohort breakdown</h2>
                <span class="toggle-icon">▼</span>
            </div>
            <div id="popSummaryTable"></div>
        </section>

        <section class="dash-section" id="popEpiSection">
            <div class="section-header">
                <span class="section-icon">🧮</span>
                <h2>Epidemiology overlay — Fang et al. 2025</h2>
                <span class="toggle-icon">▼</span>
            </div>
            <div class="charts-row cols-2" id="popEpiCharts"></div>
            <div class="placeholder-body" id="popEpiCitation" style="margin-top:.5rem"></div>
        </section>

        <section class="dash-section" id="popNetworkSection">
            <div class="section-header">
                <span class="section-icon">🧠</span>
                <h2>Network-level disruption atlas (Schaefer 7 networks)</h2>
                <span class="toggle-icon">▼</span>
            </div>
            <div id="popNetworkAtlas"></div>
        </section>

        <section class="dash-section" id="popModelSection">
            <div class="section-header">
                <span class="section-icon">🤖</span>
                <h2>GELSTM model card</h2>
                <span class="toggle-icon">▼</span>
            </div>
            <div id="popModelCard"></div>
        </section>
    `;
}

function wireSectionToggles(root) {
    root.querySelectorAll('.section-header').forEach(hdr => {
        hdr.addEventListener('click', () => {
            const sec = hdr.closest('.dash-section');
            if (sec) sec.classList.toggle('collapsed');
        });
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Summary section (top cards + cohort breakdown table)
// ─────────────────────────────────────────────────────────────────────────────
function renderSummarySection(summary) {
    const cards = $('popSummaryCards');
    const tableHost = $('popSummaryTable');
    if (!cards || !tableHost) return;
    if (!summary) {
        cards.innerHTML = `<div class="placeholder-card"><div class="placeholder-body">Summary unavailable.</div></div>`;
        tableHost.innerHTML = '';
        return;
    }

    const totals = summary.totals || {};
    cards.innerHTML = [
        summaryCard('Total subjects', totals.n_subjects ?? '–', '👥'),
        summaryCard('Total visits', totals.n_visits ?? '–', '📅'),
        summaryCard('Cohorts represented', totals.n_cohorts ?? '–', '🧬'),
        summaryCard('MCI → AD conversion',
            totals.mci_conversion_rate != null
                ? `${(totals.mci_conversion_rate * 100).toFixed(1)}%`
                : '–',
            '⏱️', { isText: true }),
    ].join('');

    const cohorts = summary.cohorts || {};
    const rows = Object.entries(cohorts)
        .filter(([, v]) => v && v.n_subjects > 0)
        .map(([name, v]) => `
            <tr>
                <td class="cohort-cell"><span class="cohort-dot" style="background:${cohortColor(name)}"></span>${name}</td>
                <td>${v.n_subjects}</td>
                <td>${v.n_visits}</td>
                <td>${fmtAge(v.age_mean, v.age_std)}</td>
                <td>${fmtPct(v.sex_pct_F)}</td>
                <td>${fmtPct(v.apoe4_pct)}</td>
            </tr>
        `).join('');

    const sites = summary.site || {};
    const siteRows = Object.entries(sites)
        .map(([site, v]) => `<tr><td>${escapeHtml(site)}</td><td>${v.n_subjects}</td><td>${v.n_visits}</td></tr>`)
        .join('');

    tableHost.innerHTML = `
        <div class="population-table-wrap">
            <table class="population-table">
                <thead>
                    <tr><th>Cohort</th><th>Subjects</th><th>Visits</th><th>Age</th><th>% Female</th><th>% APOE4+</th></tr>
                </thead>
                <tbody>${rows || '<tr><td colspan="6" class="muted">No cohort rows</td></tr>'}</tbody>
            </table>
        </div>
        ${siteRows ? `
            <div class="population-table-wrap" style="margin-top:1rem">
                <div class="population-table-title">Site / study breakdown</div>
                <table class="population-table">
                    <thead><tr><th>Site</th><th>Subjects</th><th>Visits</th></tr></thead>
                    <tbody>${siteRows}</tbody>
                </table>
            </div>` : ''}
    `;
}

function summaryCard(label, value, icon, opts = {}) {
    return `
        <div class="summary-card">
            <div class="card-top">
                <div class="card-icon">${icon}</div>
                <div class="card-label">${label}</div>
            </div>
            <div class="card-value ${opts.isText ? 'is-text' : ''}">${value}</div>
        </div>
    `;
}

// ─────────────────────────────────────────────────────────────────────────────
// Epidemiology overlay — Fang 2025 lifetime risk bars
// ─────────────────────────────────────────────────────────────────────────────
function renderEpidemiologySection(epi, summary) {
    const host = $('popEpiCharts');
    const cite = $('popEpiCitation');
    if (!host || !cite) return;
    if (!epi) {
        host.innerHTML = `<div class="placeholder-card"><div class="placeholder-body">Epidemiology table unavailable.</div></div>`;
        return;
    }

    host.innerHTML = `
        <div class="chart-card">
            <h3>Lifetime risk by stratum</h3>
            <div class="chart-container"><canvas id="popEpiStratumChart"></canvas></div>
        </div>
        <div class="chart-card">
            <h3>Residual lifetime risk by attained age</h3>
            <div class="chart-container"><canvas id="popEpiAgeChart"></canvas></div>
        </div>
    `;
    cite.innerHTML = `<strong>Source:</strong> ${escapeHtml(epi.citation || '')}<br><span class="muted">${escapeHtml(epi.notes || '')}</span>`;

    const stratumLabels = [];
    const stratumValues = [];
    const stratumColors = [];
    const pushStratum = (group, colorList) => {
        Object.entries(group).forEach(([label, value], i) => {
            stratumLabels.push(label);
            stratumValues.push(value);
            stratumColors.push(colorList[i % colorList.length]);
        });
    };
    pushStratum(epi.by_sex || {},   ['#d28b9b', '#4f98a3']);
    pushStratum(epi.by_apoe4 || {}, ['#6daa45', '#e8af34', '#d05c5c']);
    pushStratum(epi.by_race || {},  ['#7a7976', '#9a9893', '#bdbbb5', '#5d5d5b']);

    const ctxStrat = document.getElementById('popEpiStratumChart');
    if (ctxStrat) {
        _activeCharts.epiStratum = new Chart(ctxStrat, {
            type: 'bar',
            data: {
                labels: stratumLabels,
                datasets: [{
                    label: 'Lifetime AD risk (Fang 2025)',
                    data: stratumValues.map(v => +(v * 100).toFixed(1)),
                    backgroundColor: stratumColors,
                    borderWidth: 0,
                }],
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: { callbacks: { label: c => `${c.parsed.x.toFixed(1)}%` } },
                },
                scales: {
                    x: { title: { display: true, text: 'Lifetime risk (%)' }, suggestedMax: 70 },
                    y: { ticks: { autoSkip: false } },
                },
            },
        });
    }

    const ages = epi.by_age_residual || [];
    const ctxAge = document.getElementById('popEpiAgeChart');
    if (ctxAge && ages.length) {
        _activeCharts.epiAge = new Chart(ctxAge, {
            type: 'line',
            data: {
                labels: ages.map(a => a.age),
                datasets: [{
                    label: 'Residual lifetime risk',
                    data: ages.map(a => +(a.risk * 100).toFixed(1)),
                    borderColor: '#4f98a3',
                    backgroundColor: 'rgba(79,152,163,.15)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 4,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: { callbacks: { label: c => `Age ${c.label}: ${c.parsed.y.toFixed(1)}%` } },
                },
                scales: {
                    y: { title: { display: true, text: 'Risk (%)' }, suggestedMax: 50, suggestedMin: 0 },
                    x: { title: { display: true, text: 'Attained age (years)' } },
                },
            },
        });
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Network disruption atlas — Schaefer 7-network × cohort-pair Cohen's d
// ─────────────────────────────────────────────────────────────────────────────
function renderNetworkAtlasSection(atlas) {
    const host = $('popNetworkAtlas');
    if (!host) return;
    if (!atlas || atlas.available === false || !atlas.networks?.length) {
        host.innerHTML = `<div class="placeholder-card"><div class="placeholder-body">
            ${escapeHtml(atlas?.note || 'Network atlas unavailable. Make sure scan folders are loaded and the cohort warmup has finished computing per-network FC.')}
        </div></div>`;
        return;
    }

    const { networks, cohorts, matrix } = atlas;
    const pairs = [];
    for (let i = 0; i < cohorts.length; i++) {
        for (let j = i + 1; j < cohorts.length; j++) {
            pairs.push([cohorts[i], cohorts[j]]);
        }
    }

    let dMax = 0;
    networks.forEach(net => {
        Object.values(matrix[net] || {}).forEach(d => {
            if (d != null && isFinite(d) && Math.abs(d) > dMax) dMax = Math.abs(d);
        });
    });
    if (!dMax) dMax = 1;

    const rows = networks.map(net => {
        const cells = pairs.map(([a, b]) => {
            const d = (matrix[net] || {})[`${a}__${b}`];
            const value = (d != null && isFinite(d)) ? d : null;
            const bg = value == null ? 'transparent' : diverging(value, dMax);
            const txt = value == null ? '–' : value.toFixed(2);
            return `<td class="atlas-cell" style="background:${bg}" title="${escapeHtml(a)} vs ${escapeHtml(b)}: d=${txt}">${txt}</td>`;
        }).join('');
        return `<tr><th class="atlas-row-label">${escapeHtml(net)}</th>${cells}</tr>`;
    }).join('');

    const headerCells = pairs.map(([a, b]) =>
        `<th class="atlas-col-label" title="${escapeHtml(a)} vs ${escapeHtml(b)}">${escapeHtml(a)}<br>vs<br>${escapeHtml(b)}</th>`
    ).join('');

    host.innerHTML = `
        <div class="atlas-wrap">
            <table class="atlas-table">
                <thead><tr><th></th>${headerCells}</tr></thead>
                <tbody>${rows}</tbody>
            </table>
            <div class="atlas-legend">
                <span class="atlas-legend-label">Cohen's d (approximated from pooled stats)</span>
                <div class="atlas-legend-bar">
                    <span style="background:${diverging(-dMax, dMax)}"></span>
                    <span style="background:${diverging(-dMax / 2, dMax)}"></span>
                    <span style="background:${diverging(0, dMax)}"></span>
                    <span style="background:${diverging(dMax / 2, dMax)}"></span>
                    <span style="background:${diverging(dMax, dMax)}"></span>
                </div>
                <div class="atlas-legend-ticks">
                    <span>${(-dMax).toFixed(2)}</span>
                    <span>0</span>
                    <span>${dMax.toFixed(2)}</span>
                </div>
            </div>
        </div>
        <div class="placeholder-body" style="margin-top:.75rem">
            Cells show pooled Cohen's d for within-network mean FC between
            cohort pairs (cohort A − cohort B). Negative values mean A has
            lower within-network FC than B. Bootstrap CIs land in Phase 3 once
            per-subject network values are cached.
        </div>
    `;
}

// ─────────────────────────────────────────────────────────────────────────────
// Model card — GELSTM ensemble performance card
// ─────────────────────────────────────────────────────────────────────────────
function renderModelCardSection(modelCard) {
    const host = $('popModelCard');
    if (!host) return;
    if (!modelCard || modelCard.available === false) {
        const sec = document.getElementById('popModelSection');
        if (sec) sec.classList.add('collapsed');
        host.innerHTML = `<div class="placeholder-card">
            <div class="placeholder-title" style="color:var(--text-2);font-size:.85rem">Model not yet configured</div>
            <div class="placeholder-body">GELSTM fold checkpoints have not been placed in the expected directory. This section will populate once the ensemble is trained and deployed.</div>
        </div>`;
        return;
    }

    const rocAuc = modelCard.roc?.auc;
    const prAuc = modelCard.pr?.auc;
    const cm = modelCard.cm;
    const perDataset = modelCard.per_dataset_auc || modelCard.per_cohort_auc;
    const cards = [
        cardCell('Model version', modelCard.model_version || '–', '🧾', { mono: true }),
        cardCell('Ensemble folds', modelCard.n_folds ?? '–', '🧬'),
        cardCell('ROC AUC',  rocAuc != null ? rocAuc.toFixed(3) : '–', '📈'),
        cardCell('PR AUC',   prAuc  != null ? prAuc.toFixed(3) : '–', '📊'),
    ];

    const cmHtml = (cm && cm.tp != null) ? `
        <div class="chart-card">
            <h3>Confusion matrix (held-out)</h3>
            <table class="cm-table">
                <thead><tr><th></th><th>Predicted +</th><th>Predicted −</th></tr></thead>
                <tbody>
                    <tr><th>Actual +</th><td class="cm-cell cm-tp">${cm.tp ?? '–'}</td><td class="cm-cell">${cm.fn ?? '–'}</td></tr>
                    <tr><th>Actual −</th><td class="cm-cell">${cm.fp ?? '–'}</td><td class="cm-cell cm-tn">${cm.tn ?? '–'}</td></tr>
                </tbody>
            </table>
        </div>` : '';

    const perDatasetHtml = perDataset ? `
        <div class="chart-card">
            <h3>Per-dataset generalisation</h3>
            <table class="population-table">
                <thead><tr><th>Dataset</th><th>AUC</th><th>N</th></tr></thead>
                <tbody>${Object.entries(perDataset).map(([k, v]) => `
                    <tr><td>${escapeHtml(k)}</td><td>${typeof v === 'object' ? (v.auc?.toFixed(3) ?? '–') : Number(v).toFixed(3)}</td>
                    <td>${typeof v === 'object' ? (v.n ?? '–') : '–'}</td></tr>`).join('')}
                </tbody>
            </table>
        </div>` : '';

    host.innerHTML = `
        <section class="summary-cards" style="margin-bottom:1rem">${cards.join('')}</section>
        ${(cmHtml || perDatasetHtml)
            ? `<div class="charts-row cols-2">${cmHtml}${perDatasetHtml}</div>`
            : ''}
        ${modelCard.note ? `<div class="placeholder-body" style="margin-top:.75rem"><em>${escapeHtml(modelCard.note)}</em></div>` : ''}
    `;
}

function cardCell(label, value, icon, opts = {}) {
    return `
        <div class="summary-card">
            <div class="card-top">
                <div class="card-icon">${icon}</div>
                <div class="card-label">${label}</div>
            </div>
            <div class="card-value ${opts.mono ? 'is-text' : ''}" style="${opts.mono ? 'font-family:\'JetBrains Mono\',monospace;font-size:.78rem' : ''}">${value}</div>
        </div>`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
async function fetchJson(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url}: ${r.status}`);
    return r.json();
}

function escapeHtml(s) {
    return String(s ?? '').replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
}

function fmtAge(mean, std) {
    if (mean == null) return '–';
    const m = mean.toFixed(1);
    if (std == null) return m;
    return `${m} ± ${std.toFixed(1)}`;
}

function fmtPct(p) {
    if (p == null) return '–';
    return `${(p * 100).toFixed(0)}%`;
}

function cohortColor(name) {
    const palette = {
        healthy: '#6daa45',
        scd: '#4f98a3',
        mci: '#e8af34',
        converter: '#d28b9b',
        ad: '#d05c5c',
    };
    return palette[name] || '#7a7976';
}

// Red-white-blue diverging palette (negative = blue, positive = red).
function diverging(value, max) {
    const t = Math.max(-1, Math.min(1, value / max));
    if (t >= 0) {
        const a = t;
        const r = Math.round(245 + (208 - 245) * a);
        const g = Math.round(245 + (92  - 245) * a);
        const b = Math.round(245 + (92  - 245) * a);
        return `rgba(${r},${g},${b},${Math.max(0.15, Math.abs(a))})`;
    } else {
        const a = -t;
        const r = Math.round(245 + (79  - 245) * a);
        const g = Math.round(245 + (152 - 245) * a);
        const b = Math.round(245 + (163 - 245) * a);
        return `rgba(${r},${g},${b},${Math.max(0.15, Math.abs(a))})`;
    }
}
