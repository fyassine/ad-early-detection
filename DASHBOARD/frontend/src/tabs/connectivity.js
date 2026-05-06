import { $ } from '../utils.js';
import { state } from '../state.js';
import { _resolveQueryFolders } from '../modal.js';

const NET_ORDER = ['Default', 'Cont', 'SalVentAttn', 'DorsAttn', 'Limbic', 'SomMot', 'Vis'];
const NET_COLORS = {
    Default: '#e08040', Cont: '#6daa45', SalVentAttn: '#e8af34',
    DorsAttn: '#4f98a3', Limbic: '#d163a7', SomMot: '#7a7976', Vis: '#9f7fbf',
};

function _hexToRgb(hex) {
    return [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)];
}

// Returns the .rois array (handles the {rois:[...]} wrapper the API sends)
async function _getAtlasRois() {
    if (state._atlasCoords) return state._atlasCoords.rois ?? state._atlasCoords;
    try {
        const r = await fetch('/api/atlas/schaefer/coords?n_parcels=200');
        if (!r.ok) return null;
        state._atlasCoords = await r.json();
        return state._atlasCoords.rois ?? state._atlasCoords;
    } catch { return null; }
}

// Build 2-D projected points from MNI coordinates for one view.
// xKey/yKey: which MNI axis to use; flipY: positive Y = top of canvas.
function _buildProjection(rois, xKey, yKey, flipY, W, H) {
    const pad = 8;
    const xs = rois.map(c => c[xKey]);
    const ys = rois.map(c => c[yKey]);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);
    const scaleX = (W - 2 * pad) / (xMax - xMin || 1);
    const scaleY = (H - 2 * pad) / (yMax - yMin || 1);
    const scale = Math.min(scaleX, scaleY);
    const ox = pad + ((W - 2 * pad) - (xMax - xMin) * scale) / 2;
    const oy = pad + ((H - 2 * pad) - (yMax - yMin) * scale) / 2;
    return rois.map(c => ({
        cx: ox + (c[xKey] - xMin) * scale,
        cy: flipY ? H - (oy + (c[yKey] - yMin) * scale) : oy + (c[yKey] - yMin) * scale,
        net: c.network || 'Unknown',
    }));
}

function _drawProjection(canvasId, projPts, highlightIdx) {
    const canvas = $(canvasId);
    if (!canvas || !projPts?.length) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0f0f0e';
    ctx.fillRect(0, 0, W, H);
    // Faint brain-oval outline for spatial context
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.ellipse(W / 2, H / 2, W / 2 - 5, H / 2 - 5, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
    // Draw ROI dots
    for (let i = 0; i < projPts.length; i++) {
        const { cx, cy, net } = projPts[i];
        const color = NET_COLORS[net] || '#888';
        const isHL = i === highlightIdx;
        ctx.beginPath();
        ctx.arc(cx, cy, isHL ? 6 : 2.5, 0, Math.PI * 2);
        ctx.fillStyle = isHL ? '#ffffff' : color + 'cc';
        ctx.fill();
        if (isHL) {
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    }
}

// Module-level state for projections and selection
let _projAxial   = null;
let _projCoronal = null;
let _selectedRoi = -1;  // currently selected ROI index
let _currentOrder = [];
let _currentCell  = 1;

function _updateMatrixHighlight() {
    const canvas = $('heatmapCanvas');
    if (!canvas) return;
    let ov = canvas.parentElement.querySelector('#matrixHLOverlay');
    if (!ov) {
        ov = document.createElement('canvas');
        ov.id = 'matrixHLOverlay';
        ov.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;';
        canvas.parentElement.appendChild(ov);
    }
    ov.width  = canvas.width;
    ov.height = canvas.height;
    const octx = ov.getContext('2d');
    octx.clearRect(0, 0, ov.width, ov.height);
    if (_selectedRoi < 0 || !_currentOrder.length) return;
    const pos = _currentOrder.indexOf(_selectedRoi);
    if (pos < 0) return;
    const px = pos * _currentCell;
    const cell = _currentCell;
    const net = _projAxial?.[_selectedRoi]?.net;
    const color = (net && NET_COLORS[net]) ? NET_COLORS[net] : '#ffffff';
    // Highlight band
    octx.fillStyle = color + '28';
    octx.fillRect(0, px, ov.width, cell);
    octx.fillRect(px, 0, cell, ov.height);
    // Border lines
    octx.strokeStyle = color + 'cc';
    octx.lineWidth = 1.5;
    octx.beginPath();
    octx.moveTo(0, px); octx.lineTo(ov.width, px);
    octx.moveTo(0, px + cell); octx.lineTo(ov.width, px + cell);
    octx.moveTo(px, 0); octx.lineTo(px, ov.height);
    octx.moveTo(px + cell, 0); octx.lineTo(px + cell, ov.height);
    octx.stroke();
}

function _attachProjClicks(projPts, canvasId) {
    const canvas = $(canvasId);
    if (!canvas || !projPts?.length) return;
    canvas.style.cursor = 'crosshair';
    canvas.onclick = e => {
        const r = canvas.getBoundingClientRect();
        const mx = (e.clientX - r.left) * (canvas.width  / r.width);
        const my = (e.clientY - r.top)  * (canvas.height / r.height);
        let minD = 18 * 18, found = -1;
        projPts.forEach(({ cx, cy }, i) => {
            const d = (cx - mx) ** 2 + (cy - my) ** 2;
            if (d < minD) { minD = d; found = i; }
        });
        if (found < 0) return;
        _selectedRoi = (_selectedRoi === found) ? -1 : found; // toggle off on re-click
        _drawProjection('projAxial',   _projAxial,   _selectedRoi);
        _drawProjection('projCoronal', _projCoronal, _selectedRoi);
        _updateMatrixHighlight();
        // Show tooltip with ROI info
        const roi = _selectedRoi >= 0 ? projPts[_selectedRoi] : null;
        const info = $('projRoiInfo');
        if (info) info.textContent = roi ? `ROI ${_selectedRoi} · ${roi.net}` : '';
    };
}

export function renderConnectivityTab() {
    if (!state.currentPatient) return;
    _selectedRoi = -1;
    _projAxial = null;
    _projCoronal = null;
    const el = $('tab-connectivity');
    el.innerHTML = `
        <div class="heatmap-wrap">
            <div style="display:flex;gap:1rem;align-items:flex-start;flex-wrap:wrap">
                <div style="flex:1 1 300px;min-width:0">
                    <div class="visit-selector" style="justify-content:flex-end;margin-bottom:.5rem">
                        <label style="display:inline-flex;align-items:center">
                            <input type="checkbox" id="heatmapGroup" style="margin-right:.3rem">
                            Group by Schaefer network
                            <span class="help-tip" tabindex="0"
                                title="Reorders the matrix so ROIs from the same Schaefer 7-network sit next to each other.">ⓘ</span>
                        </label>
                    </div>
                    <div class="heatmap-canvas-wrap">
                        <canvas class="heatmap-canvas" id="heatmapCanvas" width="540" height="540"></canvas>
                        <div class="heatmap-tooltip" id="heatmapTooltip"></div>
                    </div>
                    <div class="heatmap-colorbar">
                        <span>−0.5</span><div class="bar"></div><span>+0.5</span>
                    </div>
                </div>
                <div style="flex:0 0 210px">
                    <div style="font-size:.68rem;font-weight:700;color:var(--text-2);
                        text-transform:uppercase;letter-spacing:.04em;margin-bottom:.35rem">
                        Network regions
                    </div>
                    <div style="font-size:.6rem;color:var(--text-3);margin-bottom:.15rem">Axial (top–down)</div>
                    <canvas id="projAxial" width="210" height="150"
                        style="width:210px;height:150px;background:#0f0f0e;border-radius:6px;
                               border:1px solid var(--border);display:block"></canvas>
                    <div style="font-size:.6rem;color:var(--text-3);margin:.35rem 0 .15rem">Coronal (front)</div>
                    <canvas id="projCoronal" width="210" height="150"
                        style="width:210px;height:150px;background:#0f0f0e;border-radius:6px;
                               border:1px solid var(--border);display:block"></canvas>
                    <div id="projRoiInfo" style="font-size:.65rem;color:var(--text-2);
                        min-height:1.1em;margin-top:.3rem;font-family:'JetBrains Mono',monospace"></div>
                    <div id="projLegend" style="margin-top:.4rem;display:flex;flex-direction:column;gap:.18rem"></div>
                </div>
            </div>
        </div>`;

    $('heatmapGroup').addEventListener('change', renderConnectivityHeatmap);
    state.currentPatient.tabRendered.connectivity = true;

    // Load atlas coords, build projections, draw them
    _getAtlasRois().then(rois => {
        if (!rois?.length) return;
        const W = 210, H = 150;
        _projAxial   = _buildProjection(rois, 'x_mni', 'y_mni', true, W, H);
        _projCoronal = _buildProjection(rois, 'x_mni', 'z_mni', true, W, H);
        _drawProjection('projAxial',   _projAxial,   -1);
        _drawProjection('projCoronal', _projCoronal, -1);
        _attachProjClicks(_projAxial,   'projAxial');
        _attachProjClicks(_projCoronal, 'projCoronal');
        // Network legend
        const box = $('projLegend');
        if (box) box.innerHTML = NET_ORDER.map(net =>
            `<div style="display:flex;align-items:center;gap:.3rem;font-size:.62rem;color:var(--text-1)">
                <span style="width:8px;height:8px;border-radius:50%;background:${NET_COLORS[net]};flex-shrink:0"></span>${net}
            </div>`
        ).join('');
    });

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
        ctx.font = '12px Inter'; ctx.textAlign = 'center';
        ctx.fillText('Loading…', W / 2, H / 2);
        try {
            const folders = _resolveQueryFolders(state.currentPatient.includeAD);
            const r = await fetch(
                `/api/patient/${state.currentPatient.sid}/matrix` +
                `?scan_folders=${encodeURIComponent(folders.join(','))}&visit=${encodeURIComponent(visit)}`
            );
            if (!r.ok) throw new Error('fetch failed');
            payload = await r.json();
            state._matrixCache.set(visit, payload);
        } catch {
            ctx.fillStyle = '#d163a7';
            ctx.fillText('Failed to load matrix', W / 2, H / 2);
            return;
        }
    }

    const grouped = $('heatmapGroup')?.checked;
    const m = payload.matrix;
    const n = m.length;
    let order = Array.from({ length: n }, (_, i) => i);

    let netBlocks = null;
    let roiNetName = null;

    if (grouped) {
        const rois = await _getAtlasRois();
        if (rois?.length === n) {
            roiNetName = rois.map(c => c.network || 'Unknown');
            if (!_projAxial) {
                _projAxial   = _buildProjection(rois, 'x_mni', 'y_mni', true, 210, 150);
                _projCoronal = _buildProjection(rois, 'x_mni', 'z_mni', true, 210, 150);
                _attachProjClicks(_projAxial,   'projAxial');
                _attachProjClicks(_projCoronal, 'projCoronal');
            }
            order.sort((a, b) => {
                const ia = NET_ORDER.indexOf(roiNetName[a]);
                const ib = NET_ORDER.indexOf(roiNetName[b]);
                return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib) || a - b;
            });
            netBlocks = [];
            let pos = 0;
            for (const net of NET_ORDER) {
                const count = order.filter(i => roiNetName[i] === net).length;
                if (count > 0) { netBlocks.push({ net, start: pos, count }); pos += count; }
            }
        } else {
            const dmn = new Set(payload.dmn_indices || []);
            order = [...order.filter(i => dmn.has(i)), ...order.filter(i => !dmn.has(i))];
        }
    } else {
        const rois = state._atlasCoords?.rois ?? (Array.isArray(state._atlasCoords) ? state._atlasCoords : null);
        if (rois?.length === n) roiNetName = rois.map(c => c.network || 'Unknown');
    }

    const cell = Math.max(1, Math.floor(540 / n));
    canvas.width  = cell * n;
    canvas.height = cell * n;
    _currentOrder = order;
    _currentCell  = cell;

    const img = ctx.createImageData(canvas.width, canvas.height);
    const cmap = v => {
        const t = Math.max(-1, Math.min(1, v / 0.5));
        if (t >= 0) return [Math.round(22+(214-22)*t), Math.round(22+(90-22)*t),  Math.round(20+(59-20)*t)];
        const a = -t;
        return [Math.round(22+(59-22)*a), Math.round(22+(108-22)*a), Math.round(20+(214-20)*a)];
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
                    img.data[px]=r; img.data[px+1]=g; img.data[px+2]=b; img.data[px+3]=255;
                }
            }
        }
    }

    // Colored edge strips per network band
    if (netBlocks) {
        netBlocks.forEach(({ net, start, count }) => {
            const [cr, cg, cb] = _hexToRgb(NET_COLORS[net] || '#ffffff');
            for (let row = start*cell; row < (start+count)*cell; row++) {
                for (let col = 0; col < 2; col++) {
                    const px = (row*canvas.width+col)*4;
                    img.data[px]=cr; img.data[px+1]=cg; img.data[px+2]=cb; img.data[px+3]=210;
                }
            }
            for (let row = 0; row < 2; row++) {
                for (let col = start*cell; col < (start+count)*cell; col++) {
                    const px = (row*canvas.width+col)*4;
                    img.data[px]=cr; img.data[px+1]=cg; img.data[px+2]=cb; img.data[px+3]=210;
                }
            }
        });
    }

    ctx.putImageData(img, 0, 0);

    if (netBlocks) {
        netBlocks.forEach(({ start }, i) => {
            if (i === 0) return;
            const px = start * cell;
            ctx.strokeStyle = 'rgba(255,255,255,0.20)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(px, 0); ctx.lineTo(px, canvas.height);
            ctx.moveTo(0, px); ctx.lineTo(canvas.width, px);
            ctx.stroke();
        });
        // Label overlay
        const wrap = canvas.closest('.heatmap-canvas-wrap');
        let ov = wrap.querySelector('#heatmapNetOverlay');
        if (!ov) {
            ov = document.createElement('div');
            ov.id = 'heatmapNetOverlay';
            ov.style.cssText = 'position:absolute;top:0;left:0;right:0;bottom:0;pointer-events:none;overflow:hidden;';
            wrap.appendChild(ov);
        }
        ov.innerHTML = netBlocks.map(({ net, start, count }) => {
            const mid = ((start + count / 2) / n * 100).toFixed(2);
            return `<span style="position:absolute;left:3px;top:${mid}%;transform:translateY(-50%);
                font-size:7.5px;font-weight:700;color:${NET_COLORS[net]};
                text-shadow:0 1px 3px rgba(0,0,0,.95);white-space:nowrap">${net}</span>`;
        }).join('');
    } else if (grouped) {
        const dmnSize = (payload.dmn_indices || []).length;
        if (dmnSize > 0 && dmnSize < n) {
            const px = dmnSize * cell;
            ctx.strokeStyle = 'rgba(232,175,52,0.6)'; ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.moveTo(px,0); ctx.lineTo(px,canvas.height);
            ctx.moveTo(0,px); ctx.lineTo(canvas.width,px);
            ctx.stroke();
        }
    }

    if (!grouped) canvas.closest('.heatmap-canvas-wrap').querySelector('#heatmapNetOverlay')?.remove();

    // Re-apply selection highlight if one is active
    _updateMatrixHighlight();

    // Tooltip + hover highlight
    const tip = $('heatmapTooltip');
    canvas.onmousemove = e => {
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) * (canvas.width  / rect.width);
        const y = (e.clientY - rect.top)  * (canvas.height / rect.height);
        const ci = Math.floor(y / cell), cj = Math.floor(x / cell);
        if (ci < 0 || ci >= n || cj < 0 || cj >= n) { tip.style.display = 'none'; return; }
        const ri = order[ci], rcj = order[cj];
        const v = m[ri][rcj];
        const nA = roiNetName ? ` (${roiNetName[ri]})` : '';
        const nB = roiNetName ? ` (${roiNetName[rcj]})` : '';
        tip.textContent = `ROI ${ri}${nA} ↔ ROI ${rcj}${nB}  ·  r = ${v.toFixed(3)}`;
        tip.style.left = (e.clientX - rect.left + 12) + 'px';
        tip.style.top  = (e.clientY - rect.top  + 12) + 'px';
        tip.style.display = 'block';
        // Hover highlight on brain projections (only if no ROI is selected)
        if (_selectedRoi < 0 && _projAxial) {
            _drawProjection('projAxial',   _projAxial,   ri);
            _drawProjection('projCoronal', _projCoronal, ri);
        }
    };
    canvas.onmouseleave = () => {
        tip.style.display = 'none';
        if (_selectedRoi < 0 && _projAxial) {
            _drawProjection('projAxial',   _projAxial,   -1);
            _drawProjection('projCoronal', _projCoronal, -1);
        }
    };
}
