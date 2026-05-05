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
        const nv = new Niivue({
            backColor: [0, 0, 0, 1],
            show3Dcrosshair: false,
            isColorbar: true,
            isOrientCube: true,
        });
        nv.attachTo('qcCanvas');
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
    const modeLabel = full4D ? 'full 4D timeseries' : '3D temporal mean (cached)';
    status.textContent = `Loading ${visit} · ${modeLabel}…`;
    const url = _qcVolumeUrl(visit, full4D);
    try {
        await state.currentPatient.niiVue.loadVolumes([{ url, name: `${state.currentPatient.sid}_${visit}.nii.gz` }]);
        try {
            const nv = state.currentPatient.niiVue;
            const vol = nv.volumes?.[0];
            if (vol) {
                if (typeof nv.setColormap === 'function') nv.setColormap(vol.id, 'gray');
                const data = vol.img2RAS ? vol.img2RAS() : vol.img;
                if (data && data.length) {
                    const N = data.length;
                    const stride = Math.max(1, Math.floor(N / 100000));
                    const sample = [];
                    for (let i = 0; i < N; i += stride) {
                        const v = data[i];
                        if (Number.isFinite(v) && v !== 0) sample.push(v);
                    }
                    if (sample.length) {
                        sample.sort((a, b) => a - b);
                        const p2 = sample[Math.floor(sample.length * 0.02)];
                        const p98 = sample[Math.floor(sample.length * 0.98)];
                        vol.cal_min = p2;
                        vol.cal_max = p98;
                    }
                }
                if (typeof nv.updateGLVolume === 'function') nv.updateGLVolume();
                if (typeof nv.drawScene === 'function') nv.drawScene();
            }
        } catch (e) { console.warn('QC windowing fallback', e); }
        status.textContent = `Showing ${visit} · ${modeLabel} · scroll/drag to navigate slices`;
    } catch (e) {
        status.textContent = 'Failed to load volume: ' + (e?.message || e);
    }
}
