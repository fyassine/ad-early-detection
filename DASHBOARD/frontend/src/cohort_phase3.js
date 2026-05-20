// Cohort tab — Phase 3 sections.
//
// Fetches:
//   - /api/cohort/network-disruption  → per-network effect-size heatmap
//   - /api/cohort/graph-topology      → small-worldness, clustering, path-len bars
//   - /api/cohort/risk-distribution   → GELSTM probability histograms
//   - /api/cohort/dfc-states          → dFC stub (timeseries not yet cached)
//
// Each section is hidden until its endpoint returns data; degraded paths
// render a calm explanatory placeholder instead of an error.
import { Chart } from 'chart.js';
import { $ } from './utils.js';
import { state } from './state.js';

const COHORT_COLORS = {
    healthy: '#6daa45',
    scd: '#4f98a3',
    mci: '#e8af34',
    converter: '#d28b9b',
    ad: '#d05c5c',
};

let _charts = {};

export async function renderCohortPhase3() {
    const csv = $('csvSelect')?.value;
    const folders = (state.selectedScanFolders || []).join(',');
    if (!csv || !folders) {
        hideAllSections();
        return;
    }

    Object.values(_charts).forEach(c => { try { c.destroy(); } catch {} });
    _charts = {};

    showAllSections();
    setLoading('networkPanelContent', 'Loading per-network effect sizes…');
    setLoading('graphTopologyContent', 'Computing graph metrics (~30s for first run)…');
    setLoading('dfcStatesContent', 'Checking dynamic FC availability…');
    setLoading('riskDistContent', 'Querying GELSTM risk distribution…');

    const params = `csv_path=${encodeURIComponent(csv)}&scan_folders=${encodeURIComponent(folders)}`;
    const [netDis, topo, dfc, riskDist] = await Promise.all([
        fetchJson(`/api/cohort/network-disruption?${params}`).catch(() => null),
        fetchJson(`/api/cohort/graph-topology?${params}`).catch(() => null),
        fetchJson(`/api/cohort/dfc-states?${params}`).catch(() => null),
        fetchJson(`/api/cohort/risk-distribution?${params}`).catch(() => null),
    ]);

    renderNetworkPanel(netDis);
    renderGraphTopology(topo);
    renderDfcStates(dfc, { csv, folders });
    renderRiskDistribution(riskDist);
}

function showAllSections() {
    ['sectionNetworkPanel', 'sectionGraphTopology', 'sectionDfcStates', 'sectionRiskDist'].forEach(id => {
        const el = $(id); if (el) el.style.display = '';
    });
}
function hideAllSections() {
    ['sectionNetworkPanel', 'sectionGraphTopology', 'sectionDfcStates', 'sectionRiskDist'].forEach(id => {
        const el = $(id); if (el) el.style.display = 'none';
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-network effect sizes panel
// ─────────────────────────────────────────────────────────────────────────────
function renderNetworkPanel(atlas) {
    const host = $('networkPanelContent');
    if (!host) return;
    if (!atlas || atlas.available === false || !atlas.networks?.length) {
        host.innerHTML = placeholder(atlas?.note || 'Network effect sizes unavailable. Make sure the cohort warmup has finished computing per-network FC.');
        return;
    }
    const { networks, cohorts, matrix, global_fc_by_network } = atlas;

    // Per-cohort, per-network FC line chart (one line per network, x = cohort, y = mean FC).
    const cohortOrder = ['healthy', 'scd', 'mci', 'converter', 'ad'].filter(c => cohorts.includes(c));
    const networkLines = networks.map(net => ({
        label: net,
        data: cohortOrder.map(c => (global_fc_by_network?.[c] || {})[net] ?? null),
        borderColor: networkColor(net),
        backgroundColor: networkColor(net) + '22',
        borderWidth: 2,
        pointRadius: 4,
        tension: 0.25,
        spanGaps: true,
    }));

    host.innerHTML = `
        <div class="charts-row cols-1">
            <div class="chart-card">
                <h3>Within-network FC across cohorts</h3>
                <div class="chart-container tall"><canvas id="netFCLine"></canvas></div>
            </div>
        </div>
        <div class="atlas-wrap" style="margin-top:1rem">
            <div class="placeholder-body" style="margin-bottom:.5rem">
                Per-network pooled Cohen's d (network × cohort pair). Negative values mean the
                row's cohort A has lower within-network FC than cohort B.
            </div>
        </div>
    `;

    // Pull the heatmap from Population's atlas rendering style:
    host.querySelector('.atlas-wrap').insertAdjacentHTML('beforeend', renderAtlasTable(atlas));

    const ctx = document.getElementById('netFCLine');
    if (ctx) {
        _charts.netFCLine = new Chart(ctx, {
            type: 'line',
            data: { labels: cohortOrder, datasets: networkLines },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'right', labels: { boxWidth: 10, font: { size: 11 } } },
                    tooltip: { callbacks: { label: c => `${c.dataset.label}: ${c.parsed.y?.toFixed(3) ?? '–'}` } },
                },
                scales: {
                    y: { title: { display: true, text: 'Mean within-network FC' } },
                    x: { ticks: { autoSkip: false } },
                },
            },
        });
    }
}

function renderAtlasTable(atlas) {
    const { networks, cohorts, matrix } = atlas;
    const pairs = [];
    for (let i = 0; i < cohorts.length; i++) {
        for (let j = i + 1; j < cohorts.length; j++) pairs.push([cohorts[i], cohorts[j]]);
    }
    let dMax = 0;
    networks.forEach(net => Object.values(matrix[net] || {}).forEach(d => {
        if (d != null && isFinite(d) && Math.abs(d) > dMax) dMax = Math.abs(d);
    }));
    if (!dMax) dMax = 1;
    const headerCells = pairs.map(([a, b]) => `<th class="atlas-col-label">${a}<br>vs<br>${b}</th>`).join('');
    const rows = networks.map(net => {
        const cells = pairs.map(([a, b]) => {
            const d = (matrix[net] || {})[`${a}__${b}`];
            const value = d != null && isFinite(d) ? d : null;
            const bg = value == null ? 'transparent' : diverging(value, dMax);
            const txt = value == null ? '–' : value.toFixed(2);
            return `<td class="atlas-cell" style="background:${bg}">${txt}</td>`;
        }).join('');
        return `<tr><th class="atlas-row-label">${net}</th>${cells}</tr>`;
    }).join('');
    return `<table class="atlas-table"><thead><tr><th></th>${headerCells}</tr></thead><tbody>${rows}</tbody></table>`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Graph topology bars
// ─────────────────────────────────────────────────────────────────────────────
function renderGraphTopology(topo) {
    const host = $('graphTopologyContent');
    if (!host) return;
    if (!topo || topo.available === false) {
        host.innerHTML = placeholder(topo?.note || 'Graph topology unavailable.');
        return;
    }
    const cohorts = (topo.cohorts || []).filter(c => topo.metrics_by_cohort?.[c]?.n);
    const METRICS = [
        { key: 'small_worldness', label: 'Small-worldness σ' },
        { key: 'clustering', label: 'Clustering coefficient' },
        { key: 'path_length', label: 'Char. path length' },
        { key: 'global_efficiency', label: 'Global efficiency' },
    ];

    host.innerHTML = `
        <div class="placeholder-body" style="margin-bottom:.6rem">
            Density-thresholded binary graphs (top ${(topo.density * 100).toFixed(0)}% of edges by |r|).
            Subjects per cohort capped at ${topo.max_subjects} (random seed=42). Bars show mean ± std.
        </div>
        <div class="charts-row cols-2">
            ${METRICS.map(m => `
                <div class="chart-card">
                    <h3>${m.label}</h3>
                    <div class="chart-container"><canvas id="topo-${m.key}"></canvas></div>
                </div>`).join('')}
        </div>
    `;

    METRICS.forEach(m => {
        const ctx = document.getElementById(`topo-${m.key}`);
        if (!ctx) return;
        const means = cohorts.map(c => topo.metrics_by_cohort[c].metrics[m.key]?.mean ?? null);
        const stds = cohorts.map(c => topo.metrics_by_cohort[c].metrics[m.key]?.std ?? null);
        const colors = cohorts.map(c => COHORT_COLORS[c] || '#7a7976');
        _charts[`topo-${m.key}`] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: cohorts,
                datasets: [{
                    label: m.label,
                    data: means,
                    backgroundColor: colors.map(c => c + 'aa'),
                    borderColor: colors,
                    borderWidth: 1,
                    errorBars: stds,
                }],
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: c => {
                                const i = c.dataIndex;
                                const std = stds[i];
                                const mean = means[i];
                                if (mean == null) return '–';
                                return std != null ? `${mean.toFixed(3)} ± ${std.toFixed(3)}` : mean.toFixed(3);
                            },
                        },
                    },
                },
                scales: { y: { beginAtZero: false, title: { display: false } } },
            },
        });
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Dynamic FC stub
// ─────────────────────────────────────────────────────────────────────────────
const _dfcPoll = { timer: null };

function _renderDfcProgress(host, dfc, ctx) {
    const job = dfc?.job;
    const pct = Math.round(((job?.progress || 0)) * 100);
    const stage = String(job?.stage || 'running').replace(/_/g, ' ');
    host.innerHTML = `<div class="placeholder-card">
        <div class="placeholder-body">
            ${escapeHtml(dfc?.note || 'Dynamic FC is being computed.')}
            <div style="margin-top:.6rem;font-size:.72rem;color:var(--text-2)">
                Warmup stage: <em>${escapeHtml(stage)}</em>
            </div>
            <div style="margin-top:.4rem;background:rgba(255,255,255,.06);border-radius:6px;overflow:hidden;height:14px">
                <div id="dfcProgressFill" style="width:${pct}%;height:100%;background:linear-gradient(90deg,var(--indigo,#6366f1),var(--green,#22c55e));transition:width .35s ease"></div>
            </div>
            <div style="text-align:right;font-size:.72rem;color:var(--text-2);margin-top:.25rem">
                <span id="dfcProgressPct">${pct}</span>%
            </div>
        </div>
    </div>`;
}

function renderDfcStates(dfc, ctx) {
    const host = $('dfcStatesContent');
    if (!host) return;
    if (_dfcPoll.timer) { clearInterval(_dfcPoll.timer); _dfcPoll.timer = null; }

    if (!dfc || dfc.available === false) {
        const job = dfc?.job;
        const isRunning = job && (job.status === 'running' || job.status === 'starting');
        if (isRunning && ctx?.csv) {
            _renderDfcProgress(host, dfc, ctx);
            const url = `/api/cohort/dfc-states?csv_path=${encodeURIComponent(ctx.csv)}&scan_folders=${encodeURIComponent(ctx.folders || '')}`;
            _dfcPoll.timer = setInterval(async () => {
                const r = await fetch(url).then(res => res.ok ? res.json() : null).catch(() => null);
                if (!r) return;
                if (r.available) {
                    clearInterval(_dfcPoll.timer); _dfcPoll.timer = null;
                    renderDfcStates(r, ctx);
                    return;
                }
                const j = r.job;
                if (!j || (j.status !== 'running' && j.status !== 'starting')) {
                    // Warmup ended without producing dFC results — fall back to static placeholder.
                    clearInterval(_dfcPoll.timer); _dfcPoll.timer = null;
                    host.innerHTML = placeholder(r.note || 'Dynamic FC unavailable.');
                    return;
                }
                const pct = Math.round((j.progress || 0) * 100);
                const fill = document.getElementById('dfcProgressFill');
                const txt  = document.getElementById('dfcProgressPct');
                if (fill) fill.style.width = `${pct}%`;
                if (txt)  txt.textContent = `${pct}`;
            }, 2000);
            return;
        }
        host.innerHTML = placeholder(dfc?.note || 'Dynamic FC unavailable.');
        return;
    }
    host.innerHTML = placeholder('Dynamic FC rendering wires in once time-series caching ships.');
}

// ─────────────────────────────────────────────────────────────────────────────
// GELSTM risk distribution
// ─────────────────────────────────────────────────────────────────────────────
function renderRiskDistribution(rd) {
    const host = $('riskDistContent');
    if (!host) return;
    if (!rd || rd.available === false) {
        host.innerHTML = placeholder(rd?.note || 'Risk distribution unavailable.');
        return;
    }

    const cohorts = rd.cohorts.filter(c => rd.histograms?.[c]?.n);
    const edges = rd.bin_edges;
    const labels = edges.slice(0, -1).map((e, i) =>
        `${(e * 100).toFixed(0)}–${(edges[i + 1] * 100).toFixed(0)}%`);

    const datasets = cohorts.map(c => ({
        label: `${c} (n=${rd.histograms[c].n})`,
        data: rd.histograms[c].counts,
        backgroundColor: (COHORT_COLORS[c] || '#7a7976') + 'aa',
        borderColor: COHORT_COLORS[c] || '#7a7976',
        borderWidth: 1,
        stack: 'stack-1',
    }));

    host.innerHTML = `
        <div class="placeholder-body" style="margin-bottom:.6rem">
            Model version <code>${rd.model_version || '–'}</code>. Histograms stacked by
            diagnosis cohort.
        </div>
        <div class="chart-card">
            <h3>Predicted P(converter) per cohort</h3>
            <div class="chart-container tall"><canvas id="riskDistChart"></canvas></div>
        </div>
    `;

    const ctx = document.getElementById('riskDistChart');
    if (ctx) {
        _charts.riskDist = new Chart(ctx, {
            type: 'bar',
            data: { labels, datasets },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'top', labels: { boxWidth: 10, font: { size: 11 } } },
                },
                scales: {
                    x: { stacked: true, title: { display: true, text: 'GELSTM probability bin' } },
                    y: { stacked: true, title: { display: true, text: 'Subjects' } },
                },
            },
        });
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
async function fetchJson(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url}: ${r.status}`);
    return r.json();
}
function placeholder(message) {
    return `<div class="placeholder-card"><div class="placeholder-body">${escapeHtml(message)}</div></div>`;
}
function setLoading(id, msg) {
    const el = $(id); if (el) el.innerHTML = `<div class="placeholder-body" style="padding:1rem">${escapeHtml(msg)}</div>`;
}
function escapeHtml(s) {
    return String(s ?? '').replace(/[&<>"']/g, c => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c]));
}
function networkColor(name) {
    const palette = {
        Default:'#e08040', Cont:'#6daa45', SalVentAttn:'#e8af34',
        DorsAttn:'#4f98a3', Limbic:'#d163a7', SomMot:'#7a7976', Vis:'#9f7fbf',
    };
    return palette[name] || '#7a7976';
}
function diverging(value, max) {
    const t = Math.max(-1, Math.min(1, value / max));
    if (t >= 0) {
        const a = t;
        const r = Math.round(245 + (208 - 245) * a);
        const g = Math.round(245 + (92  - 245) * a);
        const b = Math.round(245 + (92  - 245) * a);
        return `rgba(${r},${g},${b},${Math.max(0.15, a)})`;
    }
    const a = -t;
    const r = Math.round(245 + (79  - 245) * a);
    const g = Math.round(245 + (152 - 245) * a);
    const b = Math.round(245 + (163 - 245) * a);
    return `rgba(${r},${g},${b},${Math.max(0.15, a)})`;
}
