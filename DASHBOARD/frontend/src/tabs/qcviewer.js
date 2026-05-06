import { Niivue } from '@niivue/niivue';
import { $ } from '../utils.js';
import { state } from '../state.js';
import { _resolveQueryFolders } from '../modal.js';

export function renderQCViewerTab() {
    if (!state.currentPatient) return;
    const el = $('tab-qcviewer');
    el.innerHTML = `
        <div class="qc-wrap">
            <div class="visit-selector" style="justify-content:flex-end">
                <label class="qc-toggle" style="display:inline-flex;align-items:center;gap:.4rem;font-size:.7rem;color:var(--text-2);text-transform:uppercase;letter-spacing:.05em;cursor:pointer">
                    <input type="checkbox" id="qcFull4D">
                    Load full 4D timeseries
                </label>
            </div>
            <canvas class="qc-canvas" id="qcCanvas"></canvas>
            <div class="qc-status" id="qcStatus"></div>
        </div>`;

    $('qcFull4D').addEventListener('change', () => loadQCViewerVolume());

    try {
        // Pre-size the canvas with explicit pixel dimensions before NiiVue attaches.
        // Without this, NiiVue reads clientHeight=0 when the tab panel has display:none,
        // causing the GL viewport to be initialised at 0×0.
        const qcCanvas = $('qcCanvas');
        const targetH = Math.min(Math.floor(window.innerHeight * 0.72), 780);
        const targetW = Math.max(qcCanvas.parentElement?.clientWidth || 0, Math.floor(window.innerWidth * 0.55), 400);
        qcCanvas.width = targetW;
        qcCanvas.height = targetH;

        const nv = new Niivue({
            backColor: [0, 0, 0, 1],
            show3Dcrosshair: false,
            isColorbar: true,
            isOrientCube: true,
        });
        nv.attachTo('qcCanvas');
        // Sync NiiVue's GL viewport with the pre-sized canvas after one layout frame
        requestAnimationFrame(() => {
            try { if (typeof nv.resizeListener === 'function') nv.resizeListener(); } catch (_) {}
        });
        try {
            if (nv.opts) {
                nv.opts.show3Dcrosshair = false;
                nv.opts.isCornerOrientationText = true;
            }
            if (typeof nv.setSliceType === 'function' && typeof nv.sliceTypeMultiplanar !== 'undefined') {
                nv.setSliceType(nv.sliceTypeMultiplanar);
            }
        } catch (_) {}
        state.currentPatient.niiVue = nv;
    } catch (e) {
        $('qcStatus').textContent = 'NiiVue init failed: ' + e.message;
    }

    state.currentPatient.tabRendered.qcviewer = true;
    loadQCViewerVolume();
}

function _qcVolumeUrl(visit, full4D) {
    if (!state.currentPatient) return null;
    const folders = _resolveQueryFolders(state.currentPatient.includeAD);
    let url = `/api/patient/${state.currentPatient.sid}/scan?scan_folders=${encodeURIComponent(folders.join(','))}&visit=${encodeURIComponent(visit)}&ext=.nii.gz`;
    if (!full4D) url += '&reduce=mean';
    return url;
}

export function _prefetchQCVolume(visit) {
    if (!state.currentPatient || !visit) return;
    const key = `${state.currentPatient.sid}|${visit}|mean`;
    if (state._qcPrefetched.has(key)) return;
    const hasNifti = state.currentPatient.scansList.some(s =>
        String(s.visit).toUpperCase() === String(visit).toUpperCase());
    if (!hasNifti) return;
    state._qcPrefetched.add(key);
    const url = _qcVolumeUrl(visit, false);
    if (!url) return;
    fetch(url, { method: 'GET', cache: 'force-cache' }).catch(() => {
        state._qcPrefetched.delete(key);
    });
}

export async function loadQCViewerVolume() {
    if (!state.currentPatient || !state.currentPatient.niiVue) return;
    const status = $('qcStatus');

    const visit = state.currentPatient.selectedVisit;
    const hasNifti = state.currentPatient.scansList.some(s =>
        String(s.visit).toUpperCase() === String(visit).toUpperCase());
    if (!hasNifti) {
        status.textContent = `No .nii.gz on disk for ${visit}.` +
            (state.currentPatient.scansList.length === 0
                ? ' Selected scan folders contain only parcellated .npz files — switch to a NIfTI folder for QC.'
                : '');
        return;
    }
    const full4D = $('qcFull4D')?.checked;
    const modeLabel = full4D ? 'full 4D timeseries' : 'temporal σ — signal variability (cached)';
    status.textContent = `Loading ${visit} · ${modeLabel}…`;
    const url = _qcVolumeUrl(visit, full4D);
    try {
        await state.currentPatient.niiVue.loadVolumes([{ url, name: `${state.currentPatient.sid}_${visit}.nii.gz` }]);
        try {
            const nv = state.currentPatient.niiVue;
            const vol = nv.volumes?.[0];
            if (vol) {
                if (typeof nv.setColormap === 'function') nv.setColormap(vol.id, 'gray');
                // Fallback windowing: if the NIfTI header has no valid range (cal_min ≈ cal_max),
                // compute p2/p98 from the raw voxel buffer so NiiVue renders visible contrast.
                if (Math.abs((vol.cal_max ?? 0) - (vol.cal_min ?? 0)) < 0.01) {
                    const raw = vol.img;
                    if (raw?.length) {
                        const stride = Math.max(1, Math.floor(raw.length / 80000));
                        const vals = [];
                        for (let i = 0; i < raw.length; i += stride) {
                            const v = raw[i];
                            if (Number.isFinite(v) && v > 0) vals.push(v);
                        }
                        if (vals.length > 10) {
                            vals.sort((a, b) => a - b);
                            vol.cal_min = vals[Math.floor(vals.length * 0.02)];
                            vol.cal_max = vals[Math.floor(vals.length * 0.98)];
                        }
                    }
                }
                if (typeof nv.updateGLVolume === 'function') nv.updateGLVolume();
                if (typeof nv.resizeListener === 'function') { try { nv.resizeListener(); } catch (_) {} }
                if (typeof nv.drawScene === 'function') nv.drawScene();
            }
        } catch (e) { console.warn('QC post-load', e); }
        status.textContent = `Showing ${visit} · ${modeLabel} · scroll/drag to navigate slices`;
    } catch (e) {
        status.textContent = 'Failed to load volume: ' + (e?.message || e);
    }
}
