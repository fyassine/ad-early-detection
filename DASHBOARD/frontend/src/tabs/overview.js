import { Chart } from 'chart.js';
import { $, C, tooltipStyle } from '../utils.js';
import { state } from '../state.js';
import { setSelectedVisit } from '../modal.js';

// Chart.js plugin marking the MCI → AD conversion boundary.
function makeConvPlugin(convVisit) {
    return {
        id: 'convLine',
        afterDraw(chart) {
            const labels = chart.data.labels;
            if (!convVisit || !labels) return;
            const idx = labels.indexOf(convVisit);
            if (idx < 0) return;
            const { ctx, chartArea, scales: { x } } = chart;
            const x1 = x.getPixelForValue(labels[idx]);
            const x0 = idx > 0 ? x.getPixelForValue(labels[idx - 1]) : x1;
            const xPx = idx > 0 ? (x0 + x1) / 2 : x1;
            ctx.save();
            ctx.beginPath();
            ctx.setLineDash([5, 4]);
            ctx.strokeStyle = 'rgba(248,113,113,0.65)';
            ctx.lineWidth = 1.5;
            ctx.moveTo(xPx, chartArea.top);
            ctx.lineTo(xPx, chartArea.bottom);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.font = 'bold 9px sans-serif';
            ctx.textAlign = 'right';
            ctx.fillStyle = 'rgba(156,163,175,0.85)';
            ctx.fillText('◄ conv.', xPx - 5, chartArea.top + 13);
            ctx.textAlign = 'left';
            ctx.fillStyle = 'rgba(248,113,113,0.9)';
            ctx.fillText('AD ►', xPx + 5, chartArea.top + 13);
            ctx.restore();
        },
    };
}

function _normativeRefCohort(stats) {
    if (!stats?.biomarker_stats) return null;
    if (stats.biomarker_stats.mci && Object.keys(stats.biomarker_stats.mci).length) return 'mci';
    if (stats.biomarker_stats.healthy && Object.keys(stats.biomarker_stats.healthy).length) return 'healthy';
    return null;
}

function _devCard(label, value, refMean, format = v => v?.toFixed(3) ?? '—') {
    if (value == null || refMean == null || !isFinite(refMean) || refMean === 0) {
        return `<div class="dev-card normal">
            <div class="dev-label">${label}</div>
            <div class="dev-value">${format(value)}</div>
            <div class="dev-sub">no reference</div>
        </div>`;
    }
    const pct = ((value - refMean) / Math.abs(refMean)) * 100;
    const cls = pct < -5 ? 'below' : pct > 5 ? 'above' : 'normal';
    const sign = pct >= 0 ? '+' : '';
    return `<div class="dev-card ${cls}">
        <div class="dev-label">${label}</div>
        <div class="dev-value">${format(value)}</div>
        <div class="dev-sub">${sign}${pct.toFixed(0)}% vs MCI-NC</div>
    </div>`;
}

export function renderOverviewTab() {
    if (!state.currentPatient) return;
    const el = $('tab-overview');
    const { traj, clinical, manifold, cohortStats, allVisits, mergedVisits, diagC } = state.currentPatient;
    const hasFmri = traj?.sessions?.length > 0;
    const hasClinical = clinical?.visits?.length > 0;

    if (!hasFmri && !hasClinical) {
        el.innerHTML = '<p class="no-trajectory">No longitudinal data found for this subject</p>';
        state.currentPatient.tabRendered.overview = true;
        return;
    }

    const refCohort = _normativeRefCohort(cohortStats);
    const refStats = refCohort ? cohortStats.biomarker_stats[refCohort] : null;
    const refLabel = refCohort === 'mci' ? 'MCI-NC' : (refCohort || '—');

    let stripHtml = '';
    if (refStats && hasFmri) {
        const last = traj.sessions[traj.sessions.length - 1];
        const cards = [
            _devCard('Global FC',  last.global_fc,  refStats.global_fc?.mean),
            _devCard('DMN FC',     last.dmn_fc,     refStats.dmn_fc?.mean),
            _devCard('Modularity', last.modularity, refStats.modularity?.mean),
        ];
        const score = manifold?.trajectory?.[manifold.trajectory.length - 1]?.conversion_score;
        if (score != null) {
            const cls = score > 0.5 ? 'below' : score > 0.2 ? 'above' : 'normal';
            cards.push(`<div class="dev-card ${cls}">
                <div class="dev-label">Conversion Score</div>
                <div class="dev-value">${score.toFixed(2)}</div>
                <div class="dev-sub">0=MCI-NC · 1=AD</div>
            </div>`);
        }
        stripHtml = `<div class="deviation-strip">${cards.join('')}</div>`;
    }

    let convHtml = '';
    if (manifold?.trajectory?.some(t => t.conversion_score != null)) {
        const vals = manifold.trajectory.map(t => t.conversion_score != null ? t.conversion_score.toFixed(2) : '—');
        const visits = manifold.trajectory.map(t => t.visit);
        const pairs = visits.map((v, i) => `<span><span class="v">${v}</span>: ${vals[i]}</span>`).join(' · ');
        convHtml = `<div class="conv-score-row">
            <div>
                <div class="conv-label">Conversion Score (fMRI-only)</div>
                <div class="conv-vals">${pairs}</div>
            </div>
            <canvas id="chart-convScore"></canvas>
        </div>`;
    }

    let html = stripHtml + convHtml;

    if (hasFmri) {
        html += `
        <div class="trajectory-chart-wrap">
            <h3 style="margin-bottom:.5rem;font-size:.75rem">Global FC &amp; DMN FC ${refLabel !== '—' ? `<span style="color:var(--text-3);font-weight:400">· band = ${refLabel} ±1σ</span>` : ''}</h3>
            <div class="chart-container"><canvas id="chart-trajFC"></canvas></div>
        </div>
        <div class="trajectory-chart-wrap">
            <h3 style="margin-bottom:.5rem;font-size:.75rem">Network Modularity Q ${refLabel !== '—' ? `<span style="color:var(--text-3);font-weight:400">· band = ${refLabel} ±1σ</span>` : ''}</h3>
            <div class="chart-container"><canvas id="chart-trajMod"></canvas></div>
        </div>`;
    }

    if (hasClinical) {
        html += `
        <div class="trajectory-chart-wrap">
            <h3 style="margin-bottom:.5rem;font-size:.75rem">Cognitive Scores</h3>
            <div class="chart-container"><canvas id="chart-trajCog"></canvas></div>
        </div>
        <div class="trajectory-chart-wrap">
            <h3 style="margin-bottom:.5rem;font-size:.75rem">CSF Biomarkers</h3>
            <div class="chart-container"><canvas id="chart-trajCSF"></canvas></div>
        </div>`;
    }

    html += `
        <div class="session-table-wrap">
            <table class="session-table">
                <thead><tr><th>Visit</th><th>Global FC</th><th>Modularity</th><th>Conv. Score</th><th>MMSE</th><th>CDR</th><th>Aβ42</th><th>tTau</th></tr></thead>
                <tbody>${mergedVisits.map(s => `<tr data-visit="${s.visit}"${s.visit === state.currentPatient.selectedVisit ? ' class="selected"' : ''}>
                    <td>${s.visit}</td>
                    <td>${s.global_fc?.toFixed(4) || '—'}</td>
                    <td>${s.modularity?.toFixed(4) || '—'}</td>
                    <td>${s.conversion_score != null ? s.conversion_score.toFixed(2) : '—'}</td>
                    <td>${s.mmse != null ? s.mmse : '—'}</td>
                    <td>${s.cdr != null ? s.cdr : '—'}</td>
                    <td>${s.abeta42 != null ? s.abeta42 : '—'}</td>
                    <td>${s.tau != null ? s.tau : '—'}</td>
                </tr>`).join('')}
                </tbody>
            </table>
        </div>`;

    el.innerHTML = html;

    el.querySelectorAll('.session-table tbody tr').forEach(tr => {
        tr.addEventListener('click', () => setSelectedVisit(tr.dataset.visit));
    });

    const visitNum = v => { const m = String(v).match(/M(\d+)/i); return m ? parseInt(m[1]) : 999; };
    const sharedY = { grid: { color: 'rgba(255,255,255,0.03)' }, grace: '15%', afterFit: a => { a.width = 45; } };
    const sharedOpts = { responsive: true, maintainAspectRatio: false, plugins: { tooltip: tooltipStyle() }, animation: { duration: 600 } };
    const sharedX = { grid: { display: false }, ticks: { font: { size: 11, weight: 500 } }, labels: allVisits };

    let conversionVisit = hasFmri
        ? (traj.sessions.find(s => s.file && s.file.split('/').some(seg => {
              const sl = seg.toLowerCase();
              return sl === 'ad' || sl.startsWith('ad_');
          }))?.visit ?? null)
        : null;
    const isConverter = hasClinical && (clinical.diagnosis?.some(d => String(d).toLowerCase() === 'converter') ?? false);
    if (!conversionVisit && isConverter && hasFmri && hasClinical) {
        const lastFmriNum = visitNum(traj.sessions[traj.sessions.length - 1]?.visit ?? '');
        if (lastFmriNum < 999) {
            const firstAfter = clinical.visits.find(v => visitNum(v) > lastFmriNum);
            if (firstAfter) conversionVisit = firstAfter;
        }
    }
    const convPlugin = makeConvPlugin(conversionVisit);

    const makeBand = (metric, color) => {
        if (!refStats || !refStats[metric]) return [];
        const { mean, std } = refStats[metric];
        if (mean == null || std == null) return [];
        const upper = Array(allVisits.length).fill(mean + std);
        const lower = Array(allVisits.length).fill(mean - std);
        const meanLine = Array(allVisits.length).fill(mean);
        return [
            { label: '_band_upper_' + metric, data: upper, borderColor: 'transparent', backgroundColor: color + '22', pointRadius: 0, fill: '+1', order: 99, spanGaps: true },
            { label: '_band_lower_' + metric, data: lower, borderColor: 'transparent', backgroundColor: 'transparent', pointRadius: 0, fill: false, order: 100, spanGaps: true },
            { label: refLabel + ' mean', data: meanLine, borderColor: color + 'aa', borderDash: [4, 3], borderWidth: 1.2, pointRadius: 0, fill: false, order: 98, spanGaps: true },
        ];
    };

    const bandLegendFilter = (item, data) => {
        const ds = data.datasets[item.datasetIndex];
        return ds && !String(ds.label || '').startsWith('_band_');
    };

    if (hasFmri) {
        const sessMap = new Map(traj.sessions.map(s => [s.visit, s]));
        const gFC = allVisits.map(v => sessMap.get(v)?.global_fc ?? null);
        const dFC = allVisits.map(v => sessMap.get(v)?.dmn_fc ?? null);
        const mod = allVisits.map(v => sessMap.get(v)?.modularity ?? null);
        const accentColor = diagC || C.indigo;

        const fcDatasets = [
            ...makeBand('global_fc', '#6daa45'),
            { label: 'Global FC', data: gFC, borderColor: C.indigo, backgroundColor: 'rgba(129,140,248,0.1)', borderWidth: 2.5, pointRadius: 5, pointBackgroundColor: C.indigo, pointBorderColor: C.indigo, tension: 0.3, fill: false, spanGaps: true },
            { label: 'DMN FC', data: dFC, borderColor: accentColor, backgroundColor: accentColor + '18', borderWidth: 2.5, pointRadius: 5, pointBackgroundColor: accentColor, pointBorderColor: accentColor, tension: 0.3, fill: false, spanGaps: true },
        ];
        state.activeCharts['trajFC'] = new Chart($('chart-trajFC').getContext('2d'), {
            type: 'line',
            data: { labels: allVisits, datasets: fcDatasets },
            options: { ...sharedOpts, plugins: { ...sharedOpts.plugins, legend: { position: 'top', labels: { usePointStyle: true, boxWidth: 10, padding: 12, font: { size: 11 }, filter: bandLegendFilter } }, tooltip: { ...tooltipStyle(), filter: c => !String(c.dataset?.label || '').startsWith('_band_') } }, scales: { x: sharedX, y: sharedY } },
            plugins: [convPlugin],
        });

        const modDatasets = [
            ...makeBand('modularity', '#6daa45'),
            { label: 'Modularity Q', data: mod, borderColor: C.amber, backgroundColor: 'rgba(251,191,36,0.08)', borderWidth: 2.5, pointRadius: 5, pointBackgroundColor: C.amber, pointBorderColor: C.amber, tension: 0.3, fill: false, spanGaps: true },
        ];
        state.activeCharts['trajMod'] = new Chart($('chart-trajMod').getContext('2d'), {
            type: 'line',
            data: { labels: allVisits, datasets: modDatasets },
            options: { ...sharedOpts, plugins: { ...sharedOpts.plugins, legend: { position: 'top', labels: { usePointStyle: true, boxWidth: 10, padding: 12, font: { size: 11 }, filter: bandLegendFilter } }, tooltip: { ...tooltipStyle(), filter: c => !String(c.dataset?.label || '').startsWith('_band_') } }, scales: { x: sharedX, y: sharedY } },
            plugins: [convPlugin],
        });
    }

    if (hasClinical) {
        const clinIdxMap = new Map(clinical.visits.map((v, i) => [v, i]));
        const mapClin = arr => allVisits.map(v => { const i = clinIdxMap.get(v); return i !== undefined ? (arr[i] ?? null) : null; });
        const c_mmse  = mapClin(clinical.cognitive.mmse);
        const c_cdr   = mapClin(clinical.cognitive.cdr);
        const c_pacc5 = mapClin(clinical.cognitive.pacc5);

        state.activeCharts['trajCog'] = new Chart($('chart-trajCog').getContext('2d'), {
            type: 'line',
            data: { labels: allVisits, datasets: [
                { label: 'MMSE (0-30)', data: c_mmse, borderColor: C.green, backgroundColor: C.green + '18', borderWidth: 2.5, pointRadius: 5, pointBackgroundColor: C.green, pointBorderColor: C.green, tension: 0.3, fill: true, spanGaps: true, yAxisID: 'y' },
                { label: 'CDR', data: c_cdr, borderColor: C.rose, backgroundColor: 'transparent', borderWidth: 2.5, pointRadius: 5, pointBackgroundColor: C.rose, pointBorderColor: C.rose, tension: 0.3, fill: false, spanGaps: true, yAxisID: 'y1' },
                { label: 'PACC5', data: c_pacc5, borderColor: C.cyan, backgroundColor: 'transparent', borderWidth: 2.5, pointRadius: 5, pointBackgroundColor: C.cyan, pointBorderColor: C.cyan, tension: 0.3, fill: false, spanGaps: true, yAxisID: 'y2' },
            ]},
            options: { ...sharedOpts, plugins: { ...sharedOpts.plugins, legend: { position: 'top', labels: { usePointStyle: true, boxWidth: 10, padding: 12, font: { size: 11 } } } },
                scales: {
                    x: sharedX,
                    y:  { ...sharedY, position: 'left', title: { display: true, text: 'MMSE', font: { size: 9 } }, min: 0, max: 30 },
                    y1: { grid: { display: false }, position: 'right', title: { display: true, text: 'CDR', font: { size: 9 } }, min: 0, max: 3 },
                    y2: { grid: { display: false }, position: 'right', title: { display: true, text: 'PACC5', font: { size: 9 } } },
                },
            },
            plugins: [convPlugin],
        });

        const c_abeta = mapClin(clinical.csf.abeta42);
        const c_tau   = mapClin(clinical.csf.tau);
        const c_ptau  = mapClin(clinical.csf.ptau);
        state.activeCharts['trajCSF'] = new Chart($('chart-trajCSF').getContext('2d'), {
            type: 'line',
            data: { labels: allVisits, datasets: [
                { label: 'Aβ42', data: c_abeta, borderColor: C.indigo, backgroundColor: 'transparent', borderWidth: 2.5, pointRadius: 5, pointBackgroundColor: C.indigo, pointBorderColor: C.indigo, tension: 0.3, fill: false, spanGaps: true, yAxisID: 'y' },
                { label: 't-Tau', data: c_tau, borderColor: C.amber, backgroundColor: 'transparent', borderWidth: 2.5, pointRadius: 5, pointBackgroundColor: C.amber, pointBorderColor: C.amber, tension: 0.3, fill: false, spanGaps: true, yAxisID: 'y1' },
                { label: 'p-Tau', data: c_ptau, borderColor: C.violet, backgroundColor: 'transparent', borderWidth: 2.5, pointRadius: 5, pointBackgroundColor: C.violet, pointBorderColor: C.violet, tension: 0.3, fill: false, spanGaps: true, yAxisID: 'y1' },
            ]},
            options: { ...sharedOpts, plugins: { ...sharedOpts.plugins, legend: { position: 'top', labels: { usePointStyle: true, boxWidth: 10, padding: 12, font: { size: 11 } } } },
                scales: {
                    x: sharedX,
                    y:  { ...sharedY, position: 'left', title: { display: true, text: 'Aβ42 (pg/ml)', font: { size: 9 } }, grace: '15%' },
                    y1: { grid: { display: false }, position: 'right', title: { display: true, text: 'Tau (pg/ml)', font: { size: 9 } }, grace: '15%' },
                },
            },
            plugins: [convPlugin],
        });
    }

    if (manifold?.trajectory?.some(t => t.conversion_score != null)) {
        const visits = manifold.trajectory.map(t => t.visit);
        const data = manifold.trajectory.map(t => t.conversion_score);
        state.activeCharts['convScore'] = new Chart($('chart-convScore').getContext('2d'), {
            type: 'line',
            data: { labels: visits, datasets: [{
                data, borderColor: C.orange, backgroundColor: 'rgba(224,128,64,0.18)',
                borderWidth: 2, pointRadius: 3, pointBackgroundColor: C.orange,
                tension: 0.25, fill: true, spanGaps: true,
            }]},
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: tooltipStyle() }, scales: { x: { display: false }, y: { display: false, suggestedMin: 0, suggestedMax: 1 } } },
        });
    }

    state.currentPatient.tabRendered.overview = true;
}
