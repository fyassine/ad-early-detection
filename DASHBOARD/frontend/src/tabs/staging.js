// Patient Staging tab — 2023+ analytics (A/T/N, brain-age, EBM, time-shift,
// per-Schaefer-network FC + system segregation). Backed by:
//   GET /api/patient/{sid}/staging
//   GET /api/cohort/network-stats
import { Chart } from 'chart.js';
import { $, C, tooltipStyle, diagColor } from '../utils.js';
import { state } from '../state.js';
import { setSelectedVisit } from '../modal.js';

const NETWORK_ORDER = ['Default', 'Cont', 'SalVentAttn', 'DorsAttn', 'Limbic', 'SomMot', 'Vis'];
const NETWORK_COLORS = {
    Default: '#e08040', Cont: '#6daa45', SalVentAttn: '#e8af34',
    DorsAttn: '#4f98a3', Limbic: '#d163a7', SomMot: '#7a7976', Vis: '#9f7fbf',
};

const STAGE_LABELS = {
    '0': 'Stage 0 — no AD pathology',
    '1': 'Stage 1 — Aβ pathologic change',
    '2': 'Stage 2 — Alzheimer’s disease (biological)',
    '3': 'Stage 3 — AD with neurodegeneration',
    'S': 'SNAP — non-AD pathology',
};

function _badge(symbol, isPositive) {
    if (isPositive === null || isPositive === undefined) {
        return `<span class="atn-badge atn-unknown" title="missing">${symbol}?</span>`;
    }
    return isPositive
        ? `<span class="atn-badge atn-pos">${symbol}+</span>`
        : `<span class="atn-badge atn-neg">${symbol}−</span>`;
}

function _atnRowHtml(atn) {
    if (!atn) return `<span class="atn-empty">—</span>`;
    return `<span class="atn-pill">${_badge('A', atn.a)}${_badge('T', atn.t)}${_badge('N', atn.n)}</span>`;
}

function _stagePillHtml(stage) {
    if (stage == null) return `<span class="stage-pill stage-unknown">stage ?</span>`;
    const cls = stage === 'S' ? 'stage-snap' : `stage-${stage}`;
    const label = STAGE_LABELS[String(stage)] || `stage ${stage}`;
    return `<span class="stage-pill ${cls}" title="${label}">stage ${stage}</span>`;
}

function _fmtMonths(m) {
    if (m == null) return '—';
    const sign = m >= 0 ? '+' : '−';
    const abs = Math.abs(m);
    if (abs < 12) return `${sign}${abs.toFixed(1)} mo`;
    return `${sign}${(abs / 12).toFixed(2)} yr`;
}

export async function renderStagingTab() {
    if (!state.currentPatient) return;
    const el = $('tab-staging');
    el.innerHTML = `<div class="staging-wrap"><div class="loading-text" style="padding:2rem;text-align:center">Computing biological stage, brain-age, and time-shift…</div></div>`;

    const { sid } = state.currentPatient;
    const csvPath = $('csvSelect').value;
    const folders = state.selectedScanFolders.join(',');

    let payload = null;
    try {
        const r = await fetch(`/api/patient/${sid}/staging?csv_path=${encodeURIComponent(csvPath)}&scan_folders=${encodeURIComponent(folders)}`);
        if (!r.ok) throw new Error(`staging request failed (${r.status})`);
        payload = await r.json();
    } catch (e) {
        el.innerHTML = `<div class="staging-wrap"><p class="no-trajectory">Failed to load staging data: ${e.message}</p></div>`;
        state.currentPatient.tabRendered.staging = true;
        return;
    }

    let netStats = null;
    try {
        const r = await fetch(`/api/cohort/network-stats?csv_path=${encodeURIComponent(csvPath)}&scan_folders=${encodeURIComponent(folders)}`);
        if (r.ok) netStats = await r.json();
    } catch { /* non-fatal */ }

    const visits = payload.visits || [];
    const lastVisit = visits[visits.length - 1] || {};
    const lastAtn = lastVisit.atn;
    const lastStage = lastAtn?.stage ?? null;
    const lastBag = lastVisit.brain_age?.brain_age_gap_corrected ?? lastVisit.brain_age?.brain_age_gap ?? null;
    const lastEbm = lastVisit.ebm_stage;
    const tau = payload.time_shift?.tau_months;

    // ── Hero strip: latest A/T/N + stage + brain-age gap + EBM + time-shift ────
    const heroCards = [];
    heroCards.push(`<div class="dev-card"><div class="dev-label">A/T/N (latest)</div><div class="dev-value">${_atnRowHtml(lastAtn)}</div><div class="dev-sub">NIA-AA 2024 criteria</div></div>`);
    heroCards.push(`<div class="dev-card"><div class="dev-label">Biological stage</div><div class="dev-value">${_stagePillHtml(lastStage)}</div><div class="dev-sub">Jack et al. 2024</div></div>`);

    if (lastBag !== null && lastBag !== undefined) {
        const cls = lastBag > 2 ? 'below' : lastBag < -2 ? 'above' : 'normal';
        heroCards.push(`<div class="dev-card ${cls}"><div class="dev-label">Brain-age gap</div><div class="dev-value">${lastBag > 0 ? '+' : ''}${lastBag.toFixed(1)} yr</div><div class="dev-sub">predicted − chronological</div></div>`);
    } else {
        heroCards.push(`<div class="dev-card normal"><div class="dev-label">Brain-age gap</div><div class="dev-value">—</div><div class="dev-sub">CN cohort too small</div></div>`);
    }

    if (lastEbm) {
        const seqLen = payload.ebm_sequence?.length || 0;
        const cls = lastEbm.stage >= 3 ? 'below' : lastEbm.stage >= 1 ? 'above' : 'normal';
        heroCards.push(`<div class="dev-card ${cls}"><div class="dev-label">EBM stage</div><div class="dev-value">${lastEbm.stage} / ${lastEbm.stage_max}</div><div class="dev-sub">${lastEbm.abnormalities?.length || 0} abnormal markers</div></div>`);
    }

    if (tau != null) {
        const cls = tau > 12 ? 'below' : tau < -12 ? 'above' : 'normal';
        heroCards.push(`<div class="dev-card ${cls}"><div class="dev-label">Time-shift</div><div class="dev-value">${_fmtMonths(tau)}</div><div class="dev-sub">vs cohort mean trajectory</div></div>`);
    }

    // ── Per-visit A/T/N + stage table ──────────────────────────────────────────
    let stageTable = '';
    if (visits.length) {
        const rows = visits.map(v => {
            const atn = v.atn;
            const ebm = v.ebm_stage;
            const bag = v.brain_age?.brain_age_gap_corrected ?? v.brain_age?.brain_age_gap;
            const seg = v.system_segregation;
            return `<tr data-visit="${v.visit}"${v.visit === state.currentPatient.selectedVisit ? ' class="selected"' : ''}>
                <td>${v.visit}</td>
                <td>${_atnRowHtml(atn)}</td>
                <td>${_stagePillHtml(atn?.stage)}</td>
                <td>${ebm ? `${ebm.stage}/${ebm.stage_max}` : '—'}</td>
                <td>${bag != null ? (bag > 0 ? '+' : '') + bag.toFixed(1) : '—'}</td>
                <td>${seg != null ? seg.toFixed(3) : '—'}</td>
            </tr>`;
        }).join('');
        stageTable = `<div class="session-table-wrap" style="margin-top:1rem"><table class="session-table">
            <thead><tr><th>Visit</th><th>A/T/N</th><th>Bio stage</th><th>EBM</th><th>BAG (yr)</th><th>Seg index</th></tr></thead>
            <tbody>${rows}</tbody>
        </table></div>`;
    }

    // ── EBM sequence visualisation ──
    let ebmHtml = '';
    if (payload.ebm_sequence?.length && lastEbm?.posteriors) {
        const seq = payload.ebm_sequence;
        const posteriors = lastEbm.posteriors;
        const nodes = seq.map((key, i) => {
            const p = posteriors[key];
            const cls = p == null ? 'unknown' : p > 0.5 ? 'abnormal' : 'normal';
            const pct = p == null ? '—' : `${Math.round(p * 100)}%`;
            return `<div class="ebm-node ${cls}" title="P(abnormal | x) = ${pct}">
                <div class="ebm-pos">${i + 1}</div>
                <div class="ebm-key">${key}</div>
                <div class="ebm-prob">${pct}</div>
            </div>`;
        }).join('<div class="ebm-arrow">›</div>');
        ebmHtml = `<div class="staging-card">
            <h3>Event-Based Model — biomarker abnormality sequence</h3>
            <p class="staging-sub">Estimated CN → AD ordering across the cohort. Cells highlight which biomarkers are currently abnormal in this patient at the latest visit.</p>
            <div class="ebm-sequence">${nodes}</div>
        </div>`;
    }

    // ── System-segregation trajectory ──
    const segTrajectory = visits.map(v => v.system_segregation ?? null);
    const segLabels = visits.map(v => v.visit);
    const hasSeg = segTrajectory.some(v => v != null);

    // ── Network FC small-multiples ──
    let netSmallMultiplesHtml = '';
    const refStats = netStats?.network_fc_stats?.mci || netStats?.network_fc_stats?.healthy || null;
    const subjectHasNet = visits.some(v => v.network_fc);
    if (subjectHasNet) {
        netSmallMultiplesHtml = `<div class="staging-card">
            <h3>Per-network FC trajectories <span style="font-size:.65rem;color:var(--text-3);font-weight:400">Schaefer 7 networks</span></h3>
            <div class="network-grid" id="networkSmallMultiples"></div>
        </div>`;
    }

    // ── Time-shift visualisation ──
    let tauHtml = '';
    if (tau != null) {
        const sign = tau >= 0 ? '+' : '';
        const yr = (tau / 12).toFixed(2);
        tauHtml = `<div class="staging-card">
            <h3>Disease-course time-shift</h3>
            <p class="staging-sub">Lightweight Disease Course Mapping (Couronné, Ortholand, Schiratti 2023–2024) — τ is the offset that best aligns this patient’s biomarkers to the cohort-mean disease curve.</p>
            <div class="tau-row">
                <div class="tau-value">${sign}${tau.toFixed(1)} mo (${sign}${yr} yr)</div>
                <div class="tau-bar" id="tauBar"></div>
            </div>
            <div class="tau-legend">
                <span>← ${'younger / less advanced'}</span>
                <span>cohort mean</span>
                <span>${'further into disease'} →</span>
            </div>
        </div>`;
    }

    // ── Brain-age summary ──
    let bagHtml = '';
    if (payload.brain_age_summary?.available) {
        const sum = payload.brain_age_summary;
        bagHtml = `<div class="staging-card">
            <h3>Brain-age model</h3>
            <p class="staging-sub">Ridge regressor mapping vectorised FC to chronological age, fitted on ${sum.n_train} CN baselines (cross-validated).</p>
            <div class="bag-stats">
                <span>CV MAE: <strong>${sum.cv_mae?.toFixed(2) ?? '—'} yr</strong></span>
                <span>CV R²: <strong>${sum.cv_r2?.toFixed(2) ?? '—'}</strong></span>
                <span>n: <strong>${sum.n_train}</strong></span>
            </div>
            <div class="chart-container" style="height:160px"><canvas id="bagTrajectoryChart"></canvas></div>
        </div>`;
    }

    el.innerHTML = `
        <div class="staging-wrap">
            <div class="deviation-strip">${heroCards.join('')}</div>
            ${ebmHtml}
            ${tauHtml}
            ${bagHtml}
            ${hasSeg ? `<div class="staging-card">
                <h3>System segregation index <span style="font-size:.65rem;color:var(--text-3);font-weight:400">Setton 2023 / Chan 2014</span></h3>
                <p class="staging-sub">(W − B) / W across Schaefer 7 networks. Drops with cognitive decline.</p>
                <div class="chart-container" style="height:200px"><canvas id="segTrajectoryChart"></canvas></div>
            </div>` : ''}
            ${netSmallMultiplesHtml}
            <div class="staging-card">
                <h3>Per-visit details</h3>
                ${stageTable}
            </div>
        </div>
    `;

    // Wire row clicks for visit selection
    el.querySelectorAll('.session-table tbody tr').forEach(tr => {
        tr.addEventListener('click', () => setSelectedVisit(tr.dataset.visit));
    });

    // ── Render system-segregation trajectory ──
    if (hasSeg) {
        const refMean = refStats ? null : null; // network-FC ref, not segregation directly
        state.activeCharts['segTraj'] = new Chart($('segTrajectoryChart').getContext('2d'), {
            type: 'line',
            data: {
                labels: segLabels,
                datasets: [{
                    label: 'System segregation',
                    data: segTrajectory,
                    borderColor: C.indigo, backgroundColor: 'rgba(79,152,163,0.15)',
                    borderWidth: 2.5, pointRadius: 5, tension: 0.3, fill: true, spanGaps: true,
                }],
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false }, tooltip: tooltipStyle() },
                scales: {
                    x: { grid: { display: false } },
                    y: { grid: { color: 'rgba(255,255,255,0.03)' }, grace: '15%' },
                },
            },
        });
    }

    // ── Render per-network FC small-multiples ──
    if (subjectHasNet) {
        const grid = $('networkSmallMultiples');
        if (grid) {
            NETWORK_ORDER.forEach(net => {
                const series = visits.map(v => v.network_fc?.[net] ?? null);
                if (!series.some(v => v !== null)) return;
                const card = document.createElement('div');
                card.className = 'net-card';
                card.innerHTML = `<div class="net-label" style="color:${NETWORK_COLORS[net]}">${net}</div>
                    <canvas id="netChart-${net}"></canvas>`;
                grid.appendChild(card);
                const ctx = document.getElementById(`netChart-${net}`).getContext('2d');
                const datasets = [{
                    label: net,
                    data: series,
                    borderColor: NETWORK_COLORS[net],
                    backgroundColor: NETWORK_COLORS[net] + '22',
                    borderWidth: 2, pointRadius: 3, tension: 0.3, fill: true, spanGaps: true,
                }];
                // Reference cohort band: MCI mean ± std for this network
                const refCoh = netStats?.network_fc_stats?.mci?.[net] || netStats?.network_fc_stats?.healthy?.[net];
                if (refCoh && refCoh.mean != null && refCoh.std != null) {
                    const upper = Array(series.length).fill(refCoh.mean + refCoh.std);
                    const lower = Array(series.length).fill(refCoh.mean - refCoh.std);
                    datasets.push({
                        label: '_band_upper', data: upper, borderColor: 'transparent',
                        backgroundColor: NETWORK_COLORS[net] + '14', pointRadius: 0,
                        fill: '+1', order: 99, spanGaps: true,
                    });
                    datasets.push({
                        label: '_band_lower', data: lower, borderColor: 'transparent',
                        backgroundColor: 'transparent', pointRadius: 0, fill: false,
                        order: 100, spanGaps: true,
                    });
                }
                state.activeCharts[`net-${net}`] = new Chart(ctx, {
                    type: 'line',
                    data: { labels: segLabels, datasets },
                    options: {
                        responsive: true, maintainAspectRatio: false,
                        plugins: {
                            legend: { display: false },
                            tooltip: { ...tooltipStyle(), filter: c => !String(c.dataset?.label || '').startsWith('_band_') },
                        },
                        scales: {
                            x: { grid: { display: false }, ticks: { font: { size: 9 } } },
                            y: { grid: { color: 'rgba(255,255,255,0.03)' }, grace: '15%', ticks: { font: { size: 9 } } },
                        },
                    },
                });
            });
        }
    }

    // ── Render brain-age trajectory ──
    if (payload.brain_age_summary?.available) {
        const bagSeries = visits.map(v => v.brain_age?.brain_age_gap_corrected ?? v.brain_age?.brain_age_gap ?? null);
        if (bagSeries.some(v => v !== null)) {
            const ctx = $('bagTrajectoryChart')?.getContext('2d');
            if (ctx) {
                state.activeCharts['bagTraj'] = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: segLabels,
                        datasets: [{
                            label: 'Brain-age gap (yr)',
                            data: bagSeries,
                            borderColor: C.orange,
                            backgroundColor: 'rgba(224,128,64,0.12)',
                            borderWidth: 2.5, pointRadius: 5, tension: 0.3, fill: true, spanGaps: true,
                        }],
                    },
                    options: {
                        responsive: true, maintainAspectRatio: false,
                        plugins: { legend: { display: false }, tooltip: tooltipStyle() },
                        scales: {
                            x: { grid: { display: false } },
                            y: { grid: { color: 'rgba(255,255,255,0.03)' }, grace: '15%' },
                        },
                    },
                });
            }
        }
    }

    // ── Render time-shift bar ──
    if (tau != null) {
        const bar = $('tauBar');
        if (bar) {
            const range = 60; // ±60 months full scale
            const pct = Math.max(-range, Math.min(range, tau)) / range * 50 + 50;
            bar.innerHTML = `<div class="tau-bar-fill" style="left:${Math.min(50, pct)}%; right:${Math.max(0, 100 - Math.max(50, pct))}%; background: ${tau >= 0 ? 'var(--rose)' : 'var(--green)'}"></div>
                <div class="tau-marker" style="left:${pct}%"></div>
                <div class="tau-zero"></div>`;
        }
    }

    state.currentPatient.tabRendered.staging = true;
}
