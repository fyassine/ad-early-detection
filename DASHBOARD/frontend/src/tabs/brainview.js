import { Niivue } from '@niivue/niivue';
import { $ } from '../utils.js';
import { state } from '../state.js';
import { _resolveQueryFolders, setSelectedVisit } from '../modal.js';

const NETWORK_COLORS = {
    Default:    '#e08040',
    Cont:       '#6daa45',
    SalVentAttn:'#e8af34',
    DorsAttn:   '#4f98a3',
    Limbic:     '#d163a7',
    SomMot:     '#7a7976',
    Vis:        '#9f7fbf',
};

const NETWORK_INDEX = {
    Vis: 0, SomMot: 1, DorsAttn: 2, SalVentAttn: 3, Limbic: 4, Cont: 5, Default: 6,
};

const BRAIN_MESH_LH = '/static/data/BrainMesh_ICBM152.lh.mz3';
const BRAIN_MESH_RH = null;

const BRAIN_MODES = [
    { id: 'raw',            title: 'Raw',         sub: 'Strongest correlations at this visit.' },
    { id: 'vs-cn',          title: 'Δ vs CN',     sub: 'Deviation from healthy-control baseline mean.' },
    { id: 'delta-baseline', title: 'Δ since M0',  sub: "Progression — change vs the patient's first visit." },
];

async function _getAtlasCoords() {
    if (state._atlasCoords) return state._atlasCoords;
    try {
        const r = await fetch('/api/atlas/schaefer/coords?n_parcels=200');
        if (!r.ok) return null;
        state._atlasCoords = await r.json();
        return state._atlasCoords;
    } catch { return null; }
}

async function _getCohortReferenceMatrix(cohort = 'healthy') {
    if (state._cohortRefMatrices[cohort]) return state._cohortRefMatrices[cohort];
    const csv = $('csvSelect').value;
    if (!csv || !state.selectedScanFolders.length) return null;
    try {
        const r = await fetch(
            `/api/cohort/reference?cohort=${encodeURIComponent(cohort)}` +
            `&csv_path=${encodeURIComponent(csv)}` +
            `&scan_folders=${encodeURIComponent(state.selectedScanFolders.join(','))}`
        );
        if (!r.ok) return null;
        const json = await r.json();
        state._cohortRefMatrices[cohort] = json.matrix;
        return state._cohortRefMatrices[cohort];
    } catch { return null; }
}

async function _ensureMatrix(visit) {
    let payload = state._matrixCache.get(visit);
    if (payload) return payload;
    const folders = _resolveQueryFolders(state.currentPatient.includeAD);
    try {
        const r = await fetch(
            `/api/patient/${state.currentPatient.sid}/matrix` +
            `?scan_folders=${encodeURIComponent(folders.join(','))}` +
            `&visit=${encodeURIComponent(visit)}`
        );
        if (!r.ok) return null;
        payload = await r.json();
        state._matrixCache.set(visit, payload);
        return payload;
    } catch { return null; }
}

export function renderBrainViewTab() {
    if (!state.currentPatient) return;
    const el = $('tab-brainview');
    const modeCardsHtml = BRAIN_MODES.map((m, i) => `
        <div class="mode-card${i === 0 ? ' selected' : ''}" data-mode="${m.id}">
            <div class="mode-title"><span class="dot"></span>${m.title}</div>
            <div class="mode-sub">${m.sub}</div>
        </div>`).join('');

    el.innerHTML = `
        <div class="brain-wrap">
            <div class="mode-cards" id="brainModeCards">${modeCardsHtml}</div>
            <div class="brain-controls">
                <label>Top edges:</label>
                <input type="range" id="brainTopPct" min="0.5" max="10" value="2" step="0.5">
                <span id="brainTopPctVal" style="font-size:.78rem;color:var(--text-1);min-width:2.5em">2%</span>
                <label style="margin-left:.6rem">|w| ≥</label>
                <input type="number" id="brainAbsThr" min="0" max="1" step="0.01" placeholder="auto" title="Optional: only show edges whose absolute weight is at least this value (overrides Top % if set)">
                <label class="brain-toggle"><input type="checkbox" id="brainShowCortex" checked>Show cortex</label>
                <div class="actions">
                    <button class="brain-action-btn" id="brainPlayBtn" title="Cycle through visits (1 / s)">▶ Play</button>
                    <button class="brain-action-btn" id="brainResetBtn" title="Reset rotation (r)">⟳ Reset</button>
                    <button class="brain-action-btn" id="brainSnapBtn" title="Download PNG of the current view">📷 Snapshot</button>
                </div>
            </div>
            <div id="brainNetworkFilters" class="network-filters"></div>
            <div style="font-size:.7rem;color:var(--text-3);margin:.4rem 0 .25rem;text-align:right">drag to rotate · scroll to zoom · press r to reset</div>
            <canvas id="brainNvCanvas" class="brain-canvas"></canvas>
            <div class="brain-colorbar" id="brainColorbar">
                <span id="brainCbarMin" class="tick-mid">−</span>
                <div class="scale"></div>
                <span id="brainCbarMax" class="tick-mid">+</span>
            </div>
            <div id="brainStatus" style="font-size:.75rem;color:var(--text-2);margin-top:.5rem;line-height:1.55"></div>
            <div id="brainMeshStatus" class="brain-mesh-status"></div>
        </div>`;

    let _brainSliderTimer = null;
    $('brainTopPct').addEventListener('input', e => {
        $('brainTopPctVal').textContent = e.target.value + '%';
        clearTimeout(_brainSliderTimer);
        _brainSliderTimer = setTimeout(renderBrainGraph, 60);
    });
    $('brainAbsThr').addEventListener('input', () => {
        clearTimeout(_brainSliderTimer);
        _brainSliderTimer = setTimeout(renderBrainGraph, 120);
    });

    const cortexToggle = $('brainShowCortex');
    if (cortexToggle) {
        cortexToggle.checked = state.currentPatient._cortexVisible !== false;
        cortexToggle.addEventListener('change', () => {
            state.currentPatient._cortexVisible = cortexToggle.checked;
            _applyCortexVisibility();
            _updateBrainMeshStatus();
        });
    }

    el.querySelectorAll('#brainModeCards .mode-card').forEach(card => {
        card.addEventListener('click', () => {
            el.querySelectorAll('#brainModeCards .mode-card').forEach(c =>
                c.classList.toggle('selected', c === card));
            renderBrainGraph();
        });
    });

    $('brainResetBtn').addEventListener('click', resetBrainView);
    $('brainSnapBtn').addEventListener('click', snapshotBrainView);
    $('brainPlayBtn').addEventListener('click', toggleBrainPlayback);

    try {
        const nv = new Niivue({
            backColor: [0.06, 0.06, 0.055, 1],
            show3Dcrosshair: false,
            isOrientCube: true,
        });
        nv.attachTo('brainNvCanvas');
        try {
            if (nv.opts) {
                nv.opts.showLegend = false;
                nv.opts.isColorbar = false;
                nv.opts.isCornerOrientationText = false;
                nv.opts.show3Dcrosshair = false;
            }
            if (typeof nv.setMeshThicknessOn2D === 'function') {
                nv.setMeshThicknessOn2D(0);
            }
        } catch (_) {}
        state.currentPatient.brainNv = nv;
    } catch (e) {
        $('brainStatus').textContent = 'NiiVue init failed: ' + e.message;
    }

    state.currentPatient.tabRendered.brainview = true;
    _updateBrainMeshStatus();
    renderBrainGraph();
}

function _selectedBrainMode() {
    const card = document.querySelector('#brainModeCards .mode-card.selected');
    return card?.dataset.mode || 'raw';
}

export function resetBrainView() {
    const nv = state.currentPatient?.brainNv;
    if (!nv?.scene) return;
    nv.scene.renderAzimuth = -45;
    nv.scene.renderElevation = 15;
    if (typeof nv.scene.renderZoom === 'number') nv.scene.renderZoom = 1.0;
    if (typeof nv.drawScene === 'function') nv.drawScene();
}

async function snapshotBrainView() {
    const nv = state.currentPatient?.brainNv;
    if (!nv) return;
    const canvas = nv.canvas || $('brainNvCanvas');
    if (!canvas) return;
    if (typeof nv.drawScene === 'function') nv.drawScene();
    canvas.toBlob(blob => {
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        const visit = state.currentPatient.selectedVisit || 'visit';
        const mode = _selectedBrainMode();
        a.href = url;
        a.download = `sub-${state.currentPatient.sid}_${visit}_${mode}_brainview.png`;
        document.body.appendChild(a); a.click();
        setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 0);
    }, 'image/png');
}

function toggleBrainPlayback() {
    if (!state.currentPatient) return;
    const btn = $('brainPlayBtn');
    if (state.currentPatient._brainPlayTimer) {
        clearInterval(state.currentPatient._brainPlayTimer);
        state.currentPatient._brainPlayTimer = null;
        btn.textContent = '▶ Play';
        btn.classList.remove('active');
        return;
    }
    const visits = state.currentPatient.allVisits || [];
    if (visits.length < 2) return;
    btn.textContent = '⏸ Pause';
    btn.classList.add('active');
    let idx = visits.indexOf(state.currentPatient.selectedVisit);
    if (idx < 0) idx = 0;
    state.currentPatient._brainPlayTimer = setInterval(() => {
        idx = (idx + 1) % visits.length;
        setSelectedVisit(visits[idx]);
        if (idx === 0) toggleBrainPlayback();
    }, 1100);
}

export async function renderBrainGraph() {
    if (!state.currentPatient) return;
    const visit = state.currentPatient.selectedVisit;

    const status = $('brainStatus');
    const coords = await _getAtlasCoords();
    if (!coords?.rois?.length) {
        if (status) status.innerHTML =
            `Schaefer ROI coordinates not generated yet. Run <code>python -m app.generate_schaefer_coords --parcellation … --labels …</code> once.`;
        return;
    }

    const payload = await _ensureMatrix(visit);
    if (!payload) {
        if (status) status.textContent = 'Failed to load matrix for ' + visit;
        return;
    }

    const networks = Array.from(new Set(coords.rois.map(r => r.network).filter(Boolean))).sort();
    const filtersEl = $('brainNetworkFilters');
    if (!filtersEl.dataset.populated) {
        filtersEl.innerHTML = networks.map(n =>
            `<label><input type="checkbox" data-net="${n}" checked>
                <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${NETWORK_COLORS[n] || '#7a7976'};margin-right:.25rem;vertical-align:middle"></span>${n}</label>`
        ).join('');
        filtersEl.dataset.populated = '1';
        filtersEl.addEventListener('change', renderBrainGraph);
    }
    const enabledNetworks = new Set(
        Array.from(filtersEl.querySelectorAll('input[type=checkbox]:checked')).map(i => i.dataset.net)
    );

    const m = payload.matrix;
    const n = m.length;
    if (n !== coords.rois.length) {
        if (status) status.innerHTML =
            `Matrix is ${n}×${n} but coords are ${coords.rois.length}×${coords.rois.length}. ` +
            `Brain View requires the Schaefer parcellation to match the .npz size.`;
        return;
    }

    const mode = _selectedBrainMode();
    let displayMatrix = m;
    let modeLabel = `raw correlation @ ${visit}`;

    if (mode === 'vs-cn') {
        const cn = await _getCohortReferenceMatrix('healthy');
        if (!cn) { status.textContent = 'CN reference unavailable — fitting cohort statistics first…'; return; }
        if (cn.length !== n) { status.textContent = `CN reference is ${cn.length}×${cn.length}, current matrix is ${n}×${n}.`; return; }
        displayMatrix = m.map((row, i) => row.map((v, j) => v - cn[i][j]));
        modeLabel = `Δ vs CN baseline mean @ ${visit}`;
    } else if (mode === 'delta-baseline') {
        const firstVisit = state.currentPatient.allVisits.find(v =>
            (state.currentPatient.traj?.sessions || []).some(s => s.visit === v));
        if (!firstVisit) { status.textContent = 'No fMRI baseline visit available.'; return; }
        if (firstVisit === visit) {
            displayMatrix = m.map(row => row.map(() => 0));
            modeLabel = `Δ since ${firstVisit} (= 0 — pick a later visit)`;
        } else {
            const basePayload = await _ensureMatrix(firstVisit);
            if (!basePayload || basePayload.matrix.length !== n) {
                status.textContent = `Could not load baseline matrix (${firstVisit}).`;
                return;
            }
            const base = basePayload.matrix;
            displayMatrix = m.map((row, i) => row.map((v, j) => v - base[i][j]));
            modeLabel = `Δ since ${firstVisit} → ${visit} (progression)`;
        }
    }

    const topPct = parseFloat($('brainTopPct').value) / 100;
    const absThrInput = parseFloat($('brainAbsThr').value);
    const useAbsThr = Number.isFinite(absThrInput) && absThrInput > 0;
    const totalPairs = n * (n - 1) / 2;
    const keepCount = Math.max(5, Math.floor(totalPairs * topPct));
    const edgeKey = `${visit}|${mode}|${[...enabledNetworks].sort().join(',')}`;
    let sortedEdges = (payload._sortedEdgesKey === edgeKey) ? payload._sortedEdges : null;
    if (!sortedEdges) {
        sortedEdges = [];
        for (let i = 0; i < n; i++) {
            if (!enabledNetworks.has(coords.rois[i].network)) continue;
            for (let j = i + 1; j < n; j++) {
                if (!enabledNetworks.has(coords.rois[j].network)) continue;
                const w = displayMatrix[i][j];
                if (w === 0) continue;
                sortedEdges.push({ i, j, w });
            }
        }
        sortedEdges.sort((a, b) => Math.abs(b.w) - Math.abs(a.w));
        payload._sortedEdges = sortedEdges;
        payload._sortedEdgesKey = edgeKey;
    }
    const top = useAbsThr
        ? sortedEdges.filter(e => Math.abs(e.w) >= absThrInput)
        : sortedEdges.slice(0, keepCount);

    const minKept = top.length > 0 ? Math.abs(top[top.length - 1].w).toFixed(3) : '—';
    const maxKept = top.length > 0 ? Math.abs(top[0].w).toFixed(3) : '—';
    const nPos = top.filter(e => e.w > 0).length;
    const nNeg = top.length - nPos;
    const selectorBlurb = useAbsThr ? `|w| ≥ ${absThrInput}` : `top ${(topPct * 100).toFixed(1)}%`;
    if (status) status.innerHTML =
        `<strong>${modeLabel}</strong><br>` +
        `${top.length} edges (${selectorBlurb}) · |w| ∈ [${minKept}, ${maxKept}]` +
        ` &nbsp; <span class="edge-pills">` +
            `<span class="edge-pill pos"><span class="pill-dot"></span>${nPos} increased</span>` +
            `<span class="edge-pill neg"><span class="pill-dot"></span>${nNeg} decreased</span>` +
        `</span>`;

    const sparseEdges = new Float32Array(n * n);
    top.forEach(e => {
        sparseEdges[e.i * n + e.j] = e.w;
        sparseEdges[e.j * n + e.i] = e.w;
    });
    const activeNodes = new Set();
    top.forEach(e => { activeNodes.add(e.i); activeNodes.add(e.j); });

    const edgeMaxAbs = Math.max(0.05, Math.abs(top[0]?.w ?? 0.05));
    const connectome = {
        name: `sub-${state.currentPatient.sid} ${visit} ${mode}`,
        nodeColormap: 'warm',
        nodeColormapNegative: 'winter',
        nodeMinColor: 0,
        nodeMaxColor: 6,
        nodeScale: 2.5,
        edgeColormap: 'warm',
        edgeColormapNegative: 'winter',
        edgeMin: edgeMaxAbs * 0.05,
        edgeMax: edgeMaxAbs,
        edgeScale: 1.0,
        nodes: {
            names: coords.rois.map(() => ''),
            prefilled: [],
            X: coords.rois.map(r => r.x_mni),
            Y: coords.rois.map(r => r.y_mni),
            Z: coords.rois.map(r => r.z_mni),
            Color: coords.rois.map(r => NETWORK_INDEX[r.network] ?? 0),
            Size: coords.rois.map((r, i) =>
                enabledNetworks.has(r.network)
                    ? (activeNodes.has(i) ? 3.0 : 1.2)
                    : 0.4
            ),
        },
        edges: Array.from(sparseEdges),
    };

    const cbarMin = $('brainCbarMin'), cbarMax = $('brainCbarMax');
    if (cbarMin && cbarMax) {
        cbarMin.textContent = `−${edgeMaxAbs.toFixed(2)}`;
        cbarMax.textContent = `+${edgeMaxAbs.toFixed(2)}`;
    }

    const nv = state.currentPatient.brainNv;
    if (!nv) { status.innerHTML += '<br>NiiVue instance unavailable — cannot render 3D view.'; return; }
    try {
        await _ensureBrainMeshUnderlay(nv);
        await _renderConnectomeKeepingMesh(nv, connectome);

        if (!state.currentPatient._brainViewAngled && nv.scene) {
            nv.scene.renderAzimuth = -45;
            nv.scene.renderElevation = 15;
            state.currentPatient._brainViewAngled = true;
            if (typeof nv.drawScene === 'function') nv.drawScene();
        }
    } catch (e) {
        console.warn('connectome load failed', e);
        status.innerHTML += '<br>Connectome load failed: ' + (e?.message || e);
    }
}

async function _ensureBrainMeshUnderlay(nv) {
    if (!state.currentPatient || !nv) return;
    if (state.currentPatient._brainMeshesLoaded || state.currentPatient._brainMeshesFailed) return;
    const meshList = [];
    if (BRAIN_MESH_LH) meshList.push({ url: BRAIN_MESH_LH, rgba255: [200, 200, 200, 60] });
    if (BRAIN_MESH_RH) meshList.push({ url: BRAIN_MESH_RH, rgba255: [200, 200, 200, 60] });
    if (!meshList.length) {
        state.currentPatient._brainMeshesFailed = true;
        _updateBrainMeshStatus();
        return;
    }
    try {
        await nv.loadMeshes(meshList);
        state.currentPatient._brainMeshesLoaded = true;
        state.currentPatient._cortexMeshIds = Array.isArray(nv.meshes) ? nv.meshes.map(m => m.id) : [];
        state.currentPatient._cortexMeshCount = state.currentPatient._cortexMeshIds.length;
        _applyCortexVisibility();
        _updateBrainMeshStatus();
    } catch (e) {
        console.warn('brain-mesh underlay unavailable (offline?):', e);
        state.currentPatient._brainMeshesFailed = true;
        _updateBrainMeshStatus();
    }
}

async function _renderConnectomeKeepingMesh(nv, connectome) {
    if (!nv) return;
    const tryAppendPath = typeof nv.loadConnectomeAsMesh === 'function' && typeof nv.addMesh === 'function';

    if (tryAppendPath) {
        try {
            if (state.currentPatient._connectomeMeshId && Array.isArray(nv.meshes)) {
                const old = nv.meshes.find(m => m && m.id === state.currentPatient._connectomeMeshId);
                if (old && typeof nv.removeMesh === 'function') {
                    try { nv.removeMesh(old); } catch (_) {}
                }
            }
            const mesh = nv.loadConnectomeAsMesh(connectome);
            if (mesh) {
                if (typeof mesh.updateMesh === 'function') {
                    try { mesh.updateMesh(nv.gl); } catch (_) {}
                }
                nv.addMesh(mesh);
                state.currentPatient._connectomeMeshId = mesh.id;
                if (typeof nv.drawScene === 'function') nv.drawScene();
                return;
            }
        } catch (e) {
            console.warn('append-mesh connectome path failed; falling back', e);
        }
    }

    if (typeof nv.loadConnectomeFromJSON === 'function') {
        await nv.loadConnectomeFromJSON(connectome);
    } else if (typeof nv.loadConnectome === 'function') {
        await nv.loadConnectome(connectome);
    } else {
        throw new Error('NiiVue version too old — needs loadConnectome(FromJSON)');
    }
    state.currentPatient._brainMeshesLoaded = false;
    state.currentPatient._cortexMeshIds = [];
    state.currentPatient._cortexMeshCount = 0;
    _updateBrainMeshStatus();
}

function _applyCortexVisibility() {
    const nv = state.currentPatient?.brainNv;
    if (!nv || !Array.isArray(nv.meshes)) return;
    const visible = state.currentPatient._cortexVisible !== false;
    const alpha = visible ? 60 : 0;
    const ids = new Set(state.currentPatient._cortexMeshIds || []);
    nv.meshes.forEach(m => {
        if (m && ids.has(m.id)) m.rgba255 = [200, 200, 200, alpha];
    });
    if (typeof nv.drawScene === 'function') nv.drawScene();
}

function _updateBrainMeshStatus() {
    const el = $('brainMeshStatus');
    if (!el || !state.currentPatient) return;
    const toggle = $('brainShowCortex');
    if (toggle) toggle.disabled = !state.currentPatient._brainMeshesLoaded;
    if (state.currentPatient._brainMeshesLoaded) {
        const count = state.currentPatient._cortexMeshCount || 0;
        const suffix = count === 1 ? ' (LH)' : (count > 1 ? '' : '');
        el.textContent = state.currentPatient._cortexVisible !== false
            ? `cortex underlay loaded${suffix}`
            : 'cortex underlay hidden';
    } else if (state.currentPatient._brainMeshesFailed) {
        el.textContent = 'cortex underlay unavailable';
    } else {
        el.textContent = '';
    }
}
