import { $ } from '../utils.js';
import { state } from '../state.js';
import { _resolveQueryFolders } from '../modal.js';

export function renderConnectivityTab() {
    if (!state.currentPatient) return;
    const el = $('tab-connectivity');
    el.innerHTML = `
        <div class="heatmap-wrap">
            <div class="visit-selector" style="justify-content:flex-end">
                <label style="display:inline-flex;align-items:center">
                    <input type="checkbox" id="heatmapGroup" style="margin-right:.3rem">
                    Group by Schaefer network
                    <span class="help-tip" tabindex="0" title="Reorders the matrix so ROIs from the same Schaefer 7-network sit next to each other (Default → Cont → SalVentAttn → DorsAttn → Limbic → SomMot → Vis). Diagonal blocks then show within-network connectivity; off-diagonal blocks show between-network coupling. The yellow dividers mark the DMN boundary.">ⓘ</span>
                </label>
            </div>
            <div class="heatmap-canvas-wrap">
                <canvas class="heatmap-canvas" id="heatmapCanvas" width="540" height="540"></canvas>
                <div class="heatmap-tooltip" id="heatmapTooltip"></div>
            </div>
            <div class="heatmap-colorbar">
                <span>−0.5</span>
                <div class="bar"></div>
                <span>+0.5</span>
            </div>
        </div>`;
    $('heatmapGroup').addEventListener('change', renderConnectivityHeatmap);
    state.currentPatient.tabRendered.connectivity = true;
    renderConnectivityHeatmap();
}

export async function renderConnectivityHeatmap() {
    if (!state.currentPatient) return;
    const visit = state.currentPatient.selectedVisit;
    if (!visit) return;

    const canvas = $('heatmapCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.fillStyle = '#161614';
    ctx.fillRect(0, 0, W, H);

    let payload = state._matrixCache.get(visit);
    if (!payload) {
        ctx.fillStyle = 'rgba(122,121,118,0.7)';
        ctx.font = '12px Inter';
        ctx.textAlign = 'center';
        ctx.fillText('Loading…', W / 2, H / 2);
        try {
            const folders = _resolveQueryFolders(state.currentPatient.includeAD);
            const r = await fetch(`/api/patient/${state.currentPatient.sid}/matrix?scan_folders=${encodeURIComponent(folders.join(','))}&visit=${encodeURIComponent(visit)}`);
            if (!r.ok) throw new Error('fetch failed');
            payload = await r.json();
            state._matrixCache.set(visit, payload);
        } catch (e) {
            ctx.fillStyle = '#d163a7';
            ctx.fillText('Failed to load matrix', W / 2, H / 2);
            return;
        }
    }

    const grouped = $('heatmapGroup').checked;
    const m = payload.matrix;
    const n = m.length;
    let order = Array.from({ length: n }, (_, i) => i);
    if (grouped) {
        const dmn = new Set(payload.dmn_indices || []);
        order = [...order.filter(i => dmn.has(i)), ...order.filter(i => !dmn.has(i))];
    }

    const cell = Math.max(1, Math.floor(540 / n));
    canvas.width = cell * n;
    canvas.height = cell * n;

    const img = ctx.createImageData(canvas.width, canvas.height);
    const cmap = (v) => {
        const t = Math.max(-1, Math.min(1, v / 0.5));
        if (t >= 0) {
            return [
                Math.round(22 + (214 - 22) * t),
                Math.round(22 + (90 - 22) * t),
                Math.round(20 + (59 - 20) * t),
            ];
        }
        const a = -t;
        return [
            Math.round(22 + (59 - 22) * a),
            Math.round(22 + (108 - 22) * a),
            Math.round(20 + (214 - 20) * a),
        ];
    };

    for (let i = 0; i < n; i++) {
        const ri = order[i];
        for (let j = 0; j < n; j++) {
            const cj = order[j];
            const v = m[ri][cj];
            const [r, g, b] = cmap(v);
            for (let dy = 0; dy < cell; dy++) {
                for (let dx = 0; dx < cell; dx++) {
                    const px = ((i * cell + dy) * canvas.width + (j * cell + dx)) * 4;
                    img.data[px] = r;
                    img.data[px + 1] = g;
                    img.data[px + 2] = b;
                    img.data[px + 3] = 255;
                }
            }
        }
    }
    ctx.putImageData(img, 0, 0);

    if (grouped) {
        const dmnSize = (payload.dmn_indices || []).length;
        if (dmnSize > 0 && dmnSize < n) {
            const px = dmnSize * cell;
            ctx.strokeStyle = 'rgba(232,175,52,0.6)';
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.moveTo(px, 0); ctx.lineTo(px, canvas.height);
            ctx.moveTo(0, px); ctx.lineTo(canvas.width, px);
            ctx.stroke();
        }
    }

    const tip = $('heatmapTooltip');
    canvas.onmousemove = e => {
        const r = canvas.getBoundingClientRect();
        const x = (e.clientX - r.left) * (canvas.width / r.width);
        const y = (e.clientY - r.top) * (canvas.height / r.height);
        const i = Math.floor(y / cell), j = Math.floor(x / cell);
        if (i < 0 || i >= n || j < 0 || j >= n) { tip.style.display = 'none'; return; }
        const ri = order[i], cj = order[j];
        const v = m[ri][cj];
        tip.textContent = `ROI ${ri} ↔ ROI ${cj}  ·  r = ${v.toFixed(3)}`;
        tip.style.left = (e.clientX - r.left + 12) + 'px';
        tip.style.top = (e.clientY - r.top + 12) + 'px';
        tip.style.display = 'block';
    };
    canvas.onmouseleave = () => { tip.style.display = 'none'; };
}
