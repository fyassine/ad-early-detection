import { $, DIAG_COLORS } from '../utils.js';
import { state } from '../state.js';
import { setSelectedVisit } from '../modal.js';

function _hexToRgba(hex, alpha) {
    const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    if (!m) return `rgba(127,127,127,${alpha})`;
    return `rgba(${parseInt(m[1], 16)},${parseInt(m[2], 16)},${parseInt(m[3], 16)},${alpha})`;
}

const MANIFOLD_COLORS = ['healthy', 'scd', 'mci', 'converter', 'ad'].reduce((acc, k) => {
    const hex = DIAG_COLORS[k];
    acc[k] = { fill: _hexToRgba(hex, 0.42), stroke: hex };
    return acc;
}, {});

export function renderManifoldTab() {
    if (!state.currentPatient) return;
    const el = $('tab-manifold');
    const { cohortStats } = state.currentPatient;

    if (cohortStats === null) {
        el.innerHTML = `<div class="manifold-wrap">
            <div class="loading-text" style="padding:2rem;text-align:center">
                Fitting baseline UMAP… this only happens once per dataset.<br>
                <span style="font-size:.7rem;color:var(--text-3)">First fit on a full cohort can take a few minutes; subsequent opens are instant.</span>
            </div>
        </div>`;
        return;
    }
    if (!cohortStats?.manifold?.points?.length) {
        el.innerHTML = `<div class="manifold-wrap">
            <p class="no-trajectory">Manifold could not be computed — make sure the cohort has enough baseline scans across CN/SCD/MCI/AD groups.</p>
        </div>`;
        state.currentPatient.tabRendered.manifold = true;
        return;
    }

    const legendItems = Object.keys(MANIFOLD_COLORS)
        .filter(k => cohortStats.manifold.centroids?.[k])
        .map(k => `<span class="leg-item"><span class="leg-dot" style="background:${MANIFOLD_COLORS[k].stroke}"></span>${k}</span>`)
        .join('');

    el.innerHTML = `
        <div class="manifold-wrap">
            <h3 style="font-size:.78rem;color:var(--text-2);text-transform:uppercase;letter-spacing:.04em;margin-bottom:.5rem">2D UMAP — baselines + patient trajectory</h3>
            <canvas class="manifold-canvas" id="manifoldCanvas"></canvas>
            <div class="manifold-legend">${legendItems}
                <span class="leg-item"><span class="leg-dot" style="background:#fff;outline:1px solid #fff"></span>this patient (visits)</span>
            </div>
            <div class="manifold-info">
                Conversion axis runs from <strong>MCI-NC</strong> (= mci) toward <strong>AD</strong>. The patient's
                visits are projected into the same UMAP space and connected chronologically.
                Click a visit dot to sync the other tabs to that visit.
            </div>
        </div>`;

    drawManifold();
    state.currentPatient.tabRendered.manifold = true;
}

export function drawManifold() {
    if (!state.currentPatient) return;
    const canvas = $('manifoldCanvas');
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.floor(rect.width * dpr);
    canvas.height = Math.floor(rect.height * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const W = rect.width, H = rect.height;
    ctx.clearRect(0, 0, W, H);

    const { cohortStats, manifold, selectedVisit } = state.currentPatient;
    const points = cohortStats.manifold.points || [];
    const centroids = cohortStats.manifold.centroids || {};
    const axis = cohortStats.manifold.conversion_axis || {};

    let xs = points.map(p => p.x).filter(v => v != null);
    let ys = points.map(p => p.y).filter(v => v != null);
    Object.values(centroids).forEach(c => { if (c.x != null) xs.push(c.x); if (c.y != null) ys.push(c.y); });
    (manifold?.trajectory || []).forEach(t => { if (t.x != null) xs.push(t.x); if (t.y != null) ys.push(t.y); });
    if (!xs.length || !ys.length) return;
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const padX = (maxX - minX) * 0.08 || 0.5, padY = (maxY - minY) * 0.08 || 0.5;
    const x0 = minX - padX, x1 = maxX + padX, y0 = minY - padY, y1 = maxY + padY;
    const PAD = 32;
    const sx = x => PAD + (x - x0) / (x1 - x0) * (W - 2 * PAD);
    const sy = y => H - PAD - (y - y0) / (y1 - y0) * (H - 2 * PAD);

    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 1;
    for (let i = 1; i < 8; i++) {
        const x = PAD + i * (W - 2 * PAD) / 8;
        ctx.beginPath(); ctx.moveTo(x, PAD); ctx.lineTo(x, H - PAD); ctx.stroke();
    }
    for (let i = 1; i < 6; i++) {
        const y = PAD + i * (H - 2 * PAD) / 6;
        ctx.beginPath(); ctx.moveTo(PAD, y); ctx.lineTo(W - PAD, y); ctx.stroke();
    }

    points.forEach(p => {
        if (p.x == null || p.y == null) return;
        const col = MANIFOLD_COLORS[p.cohort];
        if (!col) return;
        ctx.fillStyle = col.fill;
        ctx.beginPath(); ctx.arc(sx(p.x), sy(p.y), 4, 0, Math.PI * 2); ctx.fill();
    });

    if (axis.origin && axis.target) {
        ctx.save();
        ctx.strokeStyle = 'rgba(255,255,255,0.4)';
        ctx.setLineDash([6, 5]);
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(sx(axis.origin.x), sy(axis.origin.y));
        ctx.lineTo(sx(axis.target.x), sy(axis.target.y));
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(255,255,255,0.55)';
        ctx.font = '10px Inter, sans-serif';
        ctx.fillText('disease axis ▶', sx(axis.target.x) + 6, sy(axis.target.y) - 4);
        ctx.restore();
    }

    Object.entries(centroids).forEach(([cohort, c]) => {
        if (c.x == null || c.y == null) return;
        const col = MANIFOLD_COLORS[cohort];
        if (!col) return;
        const cx = sx(c.x), cy = sy(c.y);
        ctx.fillStyle = col.stroke;
        ctx.strokeStyle = '#0f0f0e';
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(cx, cy, 9, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 10px Inter, sans-serif';
        ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        ctx.fillText(cohort.toUpperCase(), cx, cy + 12);
        ctx.textAlign = 'start'; ctx.textBaseline = 'alphabetic';
    });

    const traj = manifold?.trajectory || [];
    const validTraj = traj.filter(t => t.x != null && t.y != null);
    if (validTraj.length >= 2) {
        ctx.strokeStyle = 'rgba(255,255,255,0.55)';
        ctx.setLineDash([5, 4]);
        ctx.lineWidth = 1.7;
        ctx.beginPath();
        ctx.moveTo(sx(validTraj[0].x), sy(validTraj[0].y));
        for (let i = 1; i < validTraj.length; i++) ctx.lineTo(sx(validTraj[i].x), sy(validTraj[i].y));
        ctx.stroke();
        ctx.setLineDash([]);
        const a = validTraj[validTraj.length - 2], b = validTraj[validTraj.length - 1];
        const ax = sx(a.x), ay = sy(a.y), bx = sx(b.x), by = sy(b.y);
        const ang = Math.atan2(by - ay, bx - ax);
        const head = 8;
        ctx.beginPath();
        ctx.moveTo(bx, by);
        ctx.lineTo(bx - head * Math.cos(ang - 0.4), by - head * Math.sin(ang - 0.4));
        ctx.lineTo(bx - head * Math.cos(ang + 0.4), by - head * Math.sin(ang + 0.4));
        ctx.closePath();
        ctx.fillStyle = 'rgba(255,255,255,0.85)';
        ctx.fill();
    }
    validTraj.forEach((t, i) => {
        const cx = sx(t.x), cy = sy(t.y);
        const isSel = t.visit === selectedVisit;
        ctx.fillStyle = '#fff';
        ctx.strokeStyle = '#0f0f0e';
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(cx, cy, isSel ? 8 : 6, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        if (isSel) {
            ctx.strokeStyle = '#ffcb6b';
            ctx.lineWidth = 2;
            ctx.beginPath(); ctx.arc(cx, cy, 11, 0, Math.PI * 2); ctx.stroke();
        }
        ctx.fillStyle = '#0f0f0e';
        ctx.font = 'bold 9px Inter, sans-serif';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(`${i + 1}`, cx, cy);
        ctx.textAlign = 'start'; ctx.textBaseline = 'alphabetic';
        ctx.fillStyle = 'rgba(255,255,255,0.75)';
        ctx.font = '10px Inter, sans-serif';
        ctx.fillText(t.visit, cx + 10, cy + 3);
    });

    canvas.onclick = e => {
        const r = canvas.getBoundingClientRect();
        const mx = e.clientX - r.left, my = e.clientY - r.top;
        for (const t of validTraj) {
            const cx = sx(t.x), cy = sy(t.y);
            if (Math.hypot(mx - cx, my - cy) <= 9) { setSelectedVisit(t.visit); return; }
        }
    };
}

export function redrawManifold() {
    if (state.currentPatient?.tabRendered.manifold) drawManifold();
}
