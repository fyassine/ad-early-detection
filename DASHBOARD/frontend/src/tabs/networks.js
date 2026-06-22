// Patient Networks tab — per-Schaefer-7-network FC trajectory.
//
// Renders seven small multiples, one per network (Default, Cont, SalVentAttn,
// DorsAttn, Limbic, SomMot, Vis). Each chart shows the subject's per-visit
// within-network FC against a normative band (cohort mean ± 1σ) drawn from
// the active workspace's CN (or MCI fallback).
import { Chart } from 'chart.js';
import { $ } from '../utils.js';
import { state } from '../state.js';

const NETWORK_ORDER = ['Default', 'Cont', 'SalVentAttn', 'DorsAttn', 'Limbic', 'SomMot', 'Vis'];
const NETWORK_COLORS = {
    Default: '#e08040', Cont: '#6daa45', SalVentAttn: '#e8af34',
    DorsAttn: '#4f98a3', Limbic: '#d163a7', SomMot: '#7a7976', Vis: '#9f7fbf',
};

let _charts = {};

export async function renderNetworksTab() {
    if (!state.currentPatient) return;
    const host = $('tab-networks');
    if (!host) return;

    Object.values(_charts).forEach(c => { try { c.destroy(); } catch {} });
    _charts = {};

    host.innerHTML = `<div class="staging-wrap"><div class="loading-text" style="padding:2rem;text-align:center">Computing per-network FC per visit…</div></div>`;

    const { sid } = state.currentPatient;
    const csv = $('csvSelect').value;
    const folders = state.selectedScanFolders.join(',');
    if (!csv || !folders) {
        host.innerHTML = placeholder('CSV + scan folders required.');
        return;
    }

    let data = null;
    try {
        const r = await fetch(`/api/patient/${sid}/network-trajectory?csv_path=${encodeURIComponent(csv)}&scan_folders=${encodeURIComponent(folders)}`);
        if (!r.ok) throw new Error(`status ${r.status}`);
        data = await r.json();
    } catch (e) {
        host.innerHTML = placeholder(`Network trajectory failed: ${escapeHtml(e.message)}`);
        return;
    }

    if (!data || data.available === false) {
        host.innerHTML = placeholder('Network trajectory unavailable.');
        state.currentPatient.tabRendered.networks = true;
        return;
    }

    const visits = data.visits || [];
    if (!visits.length) {
        host.innerHTML = placeholder('No visits found for this subject.');
        state.currentPatient.tabRendered.networks = true;
        return;
    }

    const labels = visits.map(v => v.visit);
    const networks = NETWORK_ORDER.filter(n => visits.some(v => v.network_fc?.[n] != null));
    const normative = data.normative || {};

    host.innerHTML = `
        <div class="staging-wrap">
            <div class="placeholder-body" style="margin-bottom:.75rem">
                Per-network mean within-network FC across the subject's visits.
                Dashed band = healthy normative mean ± 1σ for each network
                (falls back to MCI non-converters if CN sample is too small).
            </div>
            <div class="charts-row cols-3" id="patNetworksGrid">
                ${networks.map(net => `
                    <div class="chart-card">
                        <h3 style="color:${NETWORK_COLORS[net] || '#7a7976'}">${net}</h3>
                        <div class="chart-container"><canvas id="pat-net-${net}"></canvas></div>
                    </div>`).join('')}
            </div>
        </div>
    `;

    networks.forEach(net => {
        const ctx = document.getElementById(`pat-net-${net}`);
        if (!ctx) return;
        const color = NETWORK_COLORS[net] || '#7a7976';
        const series = visits.map(v => v.network_fc?.[net] ?? null);
        const ref = normative[net] || {};
        const refMean = ref.mean ?? null;
        const refStd = ref.std ?? null;

        const datasets = [];
        if (refMean != null && refStd != null) {
            const upper = labels.map(() => refMean + refStd);
            const lower = labels.map(() => refMean - refStd);
            const mid = labels.map(() => refMean);
            datasets.push(
                { label: '_band_upper', data: upper, borderColor: 'transparent', backgroundColor: color + '22', pointRadius: 0, fill: '+1', order: 99 },
                { label: '_band_lower', data: lower, borderColor: 'transparent', pointRadius: 0, fill: false, order: 100 },
                { label: 'normative mean', data: mid, borderColor: color + '88', borderDash: [4, 3], borderWidth: 1.2, pointRadius: 0, fill: false, order: 98 },
            );
        }
        datasets.push({
            label: `${net} FC`,
            data: series,
            borderColor: color,
            backgroundColor: color + '22',
            borderWidth: 2.5,
            pointRadius: 5,
            tension: 0.25,
            spanGaps: true,
        });

        _charts[`pat-net-${net}`] = new Chart(ctx, {
            type: 'line',
            data: { labels, datasets },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { filter: (i, d) => !String(d.datasets[i.datasetIndex]?.label || '').startsWith('_band_') }
                    },
                },
                scales: { y: { beginAtZero: false } },
            },
        });
    });

    state.currentPatient.tabRendered.networks = true;
}

function placeholder(message) {
    return `<div class="staging-wrap"><div class="placeholder-card"><div class="placeholder-body">${escapeHtml(message)}</div></div></div>`;
}
function escapeHtml(s) {
    return String(s ?? '').replace(/[&<>"']/g, c => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c]));
}
