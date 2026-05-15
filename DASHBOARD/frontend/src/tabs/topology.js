// Patient Topology tab — per-visit graph-theoretic trajectory.
//
// Charts small-worldness σ, clustering coefficient, char. path length,
// and global efficiency across the subject's visits. Lists the top-10
// DomiRank hubs at the most recent visit.
import { Chart } from 'chart.js';
import { $ } from '../utils.js';
import { state } from '../state.js';

const METRICS = [
    { key: 'small_worldness',   label: 'Small-worldness σ',     color: '#e08040' },
    { key: 'clustering',        label: 'Clustering coef',        color: '#6daa45' },
    { key: 'path_length',       label: 'Char. path length',      color: '#4f98a3' },
    { key: 'global_efficiency', label: 'Global efficiency',      color: '#d163a7' },
];

let _charts = {};

export async function renderTopologyTab() {
    if (!state.currentPatient) return;
    const host = $('tab-topology');
    if (!host) return;

    Object.values(_charts).forEach(c => { try { c.destroy(); } catch {} });
    _charts = {};

    host.innerHTML = `<div class="staging-wrap"><div class="loading-text" style="padding:2rem;text-align:center">Computing graph metrics per visit…</div></div>`;

    const { sid } = state.currentPatient;
    const csv = $('csvSelect').value;
    const folders = state.selectedScanFolders.join(',');
    if (!csv || !folders) {
        host.innerHTML = placeholder('CSV + scan folders required to compute graph metrics.');
        return;
    }

    let data = null;
    try {
        const r = await fetch(`/api/patient/${sid}/graph-trajectory?csv_path=${encodeURIComponent(csv)}&scan_folders=${encodeURIComponent(folders)}`);
        if (!r.ok) throw new Error(`status ${r.status}`);
        data = await r.json();
    } catch (e) {
        host.innerHTML = placeholder(`Graph trajectory failed: ${escapeHtml(e.message)}`);
        return;
    }

    if (!data || data.available === false) {
        host.innerHTML = placeholder(data?.note || 'Graph metrics unavailable.');
        state.currentPatient.tabRendered.topology = true;
        return;
    }

    const visits = data.visits || [];
    if (!visits.length) {
        host.innerHTML = placeholder('No visits found for this subject.');
        state.currentPatient.tabRendered.topology = true;
        return;
    }

    const labels = visits.map(v => v.visit);
    host.innerHTML = `
        <div class="staging-wrap">
            <div class="placeholder-body" style="margin-bottom:.75rem">
                Density-thresholded binary graphs at top ${(data.density * 100).toFixed(0)}% of |r| edges.
            </div>
            <div class="charts-row cols-2">
                ${METRICS.map(m => `
                    <div class="chart-card">
                        <h3>${m.label}</h3>
                        <div class="chart-container"><canvas id="pat-topo-${m.key}"></canvas></div>
                    </div>`).join('')}
            </div>
            <div class="chart-card" style="margin-top:1rem">
                <h3>Top-10 DomiRank hubs (most recent visit)</h3>
                <div id="patHubs"></div>
            </div>
        </div>
    `;

    METRICS.forEach(m => {
        const ctx = document.getElementById(`pat-topo-${m.key}`);
        if (!ctx) return;
        _charts[`pat-topo-${m.key}`] = new Chart(ctx, {
            type: 'line',
            data: {
                labels,
                datasets: [{
                    label: m.label,
                    data: visits.map(v => v[m.key] ?? null),
                    borderColor: m.color,
                    backgroundColor: m.color + '22',
                    borderWidth: 2.5,
                    pointRadius: 5,
                    tension: 0.25,
                    spanGaps: true,
                }],
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: { y: { beginAtZero: false } },
            },
        });
    });

    // Top-K DomiRank hubs from the most recent visit
    const lastWithHubs = [...visits].reverse().find(v => v.domirank_top_k?.length);
    const hubsHost = document.getElementById('patHubs');
    if (hubsHost) {
        if (!lastWithHubs) {
            hubsHost.innerHTML = `<div class="placeholder-body">No hub data available.</div>`;
        } else {
            hubsHost.innerHTML = `
                <table class="population-table">
                    <thead><tr><th>Rank</th><th>ROI index</th><th>DomiRank score</th></tr></thead>
                    <tbody>
                        ${lastWithHubs.domirank_top_k.map((h, i) => `
                            <tr><td>${i + 1}</td><td>${h.roi}</td><td>${h.score?.toFixed(3) ?? '–'}</td></tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
        }
    }
    state.currentPatient.tabRendered.topology = true;
}

function placeholder(message) {
    return `<div class="staging-wrap"><div class="placeholder-card"><div class="placeholder-body">${escapeHtml(message)}</div></div></div>`;
}

function escapeHtml(s) {
    return String(s ?? '').replace(/[&<>"']/g, c => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c]));
}
