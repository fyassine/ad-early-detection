import { $, diagColor, fmtCol, fmtVal } from './utils.js';
import { state } from './state.js';
import { renderOverviewTab } from './tabs/overview.js';
import { renderStagingTab } from './tabs/staging.js';
import { renderManifoldTab, redrawManifold } from './tabs/manifold.js';
import { renderConnectivityTab, renderConnectivityHeatmap } from './tabs/connectivity.js';
import { renderQCViewerTab, loadQCViewerVolume, _prefetchQCVolume } from './tabs/qcviewer.js';
import { renderBrainViewTab, renderBrainGraph } from './tabs/brainview.js';

export const TAB_DEFS = [
    { id: 'overview',     label: 'Overview' },
    { id: 'staging',      label: 'Staging' },
    { id: 'manifold',     label: 'Manifold' },
    { id: 'connectivity', label: 'Connectivity' },
    { id: 'qcviewer',     label: 'QC Viewer' },
    { id: 'brainview',    label: 'Brain View' },
];

function _syncPatientModalSize(activeTab) {
    const modal = $('patientModal');
    if (!modal) return;
    modal.classList.toggle('brainview-expanded', activeTab === 'brainview');
}

export async function openPatient(sid) {
    if (!sid) return;
    document.querySelectorAll('.data-table tbody tr').forEach(tr =>
        tr.classList.toggle('selected', tr.dataset.sid === sid));
    const patient = state.patientData.find(p => p.subject_id === sid) || {};
    $('modalTitle').textContent = `sub-${sid}`;
    const diagC = diagColor(String(patient.diagnosis || ''));
    $('modalSubhead').innerHTML = patient.diagnosis
        ? `<span style="color:${diagC};font-weight:600">${patient.diagnosis}</span>` : '';

    const fields = ['sex', 'age', 'n_visits', 'apoe'];
    const metaHtml = fields.filter(k => patient[k] != null).map(k =>
        `<div class="meta-item"><div class="meta-label">${fmtCol(k)}</div><div class="meta-value">${fmtVal(patient[k])}</div></div>`).join('');

    const visitHtml = (patient.visits || []).map(v => `<span class="visit-tag">${v}</span>`).join('') || '—';
    const isConverter = String(patient.diagnosis || '').toLowerCase() === 'converter';

    const toggleHtml = isConverter ? `
        <div style="margin-left:auto;display:flex;align-items:center;">
            <label class="switch">
                <input type="checkbox" id="adScanToggle" class="switch-input">
                <span class="switch-track"><span class="switch-thumb"></span></span>
                <span class="switch-label">Include AD scans</span>
            </label>
        </div>` : '';

    const tabBarHtml = TAB_DEFS.map((t, i) =>
        `<button class="modal-tab${i === 0 ? ' active' : ''}" data-tab="${t.id}">${t.label}</button>`).join('');
    const tabPanelsHtml = TAB_DEFS.map((t, i) =>
        `<div class="tab-panel${i === 0 ? ' active' : ''}" data-tab="${t.id}" id="tab-${t.id}"></div>`).join('');

    $('modalBody').innerHTML = `
        <div class="patient-summary-bar">
            <div class="patient-meta-grid">${metaHtml}</div>
            <div class="visit-bar-row">
                <span class="vbr-label">Visit:</span>
                <div id="globalVisitPills" class="vbr-pills"><span class="vbr-empty">— loading visits —</span></div>
                <span class="vbr-spacer"></span>
                <div class="vbr-csv">
                    <span class="vbr-csv-label">in CSV</span>
                    <div class="visit-tags">${visitHtml}</div>
                </div>
                ${toggleHtml}
            </div>
        </div>
        <div class="modal-tabs" id="modalTabBar">${tabBarHtml}</div>
        ${tabPanelsHtml}
    `;

    const headerSubhead = $('modalSubhead');
    if (headerSubhead && !document.getElementById('currentVisitBadge')) {
        const badge = document.createElement('span');
        badge.id = 'currentVisitBadge';
        badge.className = 'current-visit-badge';
        badge.style.display = 'none';
        badge.innerHTML = `<span class="label">visit</span><span id="currentVisitText"></span>`;
        headerSubhead.appendChild(badge);
    } else if (document.getElementById('currentVisitBadge')) {
        document.getElementById('currentVisitBadge').style.display = 'none';
    }

    state.currentPatient = {
        sid, diagC, isConverter, includeAD: false,
        traj: null, clinical: null, manifold: null,
        cohortStats: null, allVisits: [], mergedVisits: [],
        selectedVisit: (patient.visits && patient.visits.length) ? patient.visits[patient.visits.length - 1] : null,
        activeTab: 'overview',
        tabRendered: { overview: false, staging: false, manifold: false, connectivity: false, qcviewer: false, brainview: false },
        niiVue: null,
        scansList: [],
        _cortexVisible: true,
        _cortexMeshIds: [],
        _cortexMeshCount: 0,
        _brainMeshesLoaded: false,
        _brainMeshesFailed: false,
    };
    state._matrixCache.clear();

    $('modalBackdrop').classList.add('open');
    _syncPatientModalSize('overview');

    $('modalTabBar').addEventListener('click', e => {
        const btn = e.target.closest('.modal-tab');
        if (!btn) return;
        switchTab(btn.dataset.tab);
    });

    if (isConverter) {
        $('adScanToggle').addEventListener('change', e => {
            state.currentPatient.includeAD = e.target.checked;
            state.currentPatient.tabRendered = { overview: false, staging: false, manifold: false, connectivity: false, qcviewer: false, brainview: false };
            loadPatientData();
        });
    }

    loadPatientData();
}

export function switchTab(tabId) {
    if (!state.currentPatient) return;
    document.querySelectorAll('#modalTabBar .modal-tab').forEach(b =>
        b.classList.toggle('active', b.dataset.tab === tabId));
    document.querySelectorAll('.tab-panel').forEach(p =>
        p.classList.toggle('active', p.dataset.tab === tabId));
    state.currentPatient.activeTab = tabId;
    _syncPatientModalSize(tabId);
    renderActiveTab();
}

function renderActiveTab() {
    if (!state.currentPatient) return;
    const t = state.currentPatient.activeTab;
    if (t === 'overview')          renderOverviewTab();
    else if (t === 'staging')      renderStagingTab();
    else if (t === 'manifold')     renderManifoldTab();
    else if (t === 'connectivity') renderConnectivityTab();
    else if (t === 'qcviewer')     renderQCViewerTab();
    else if (t === 'brainview')    renderBrainViewTab();
}

// Compute folders to query, optionally adding sibling AD/Converter folders.
export function _resolveQueryFolders(includeAD) {
    let folders = [...state.selectedScanFolders];
    if (includeAD && state.discoveryData?.scan_folders) {
        const baseDatasets = new Set(state.selectedScanFolders.map(f => f.split('/')[0]));
        const sib = state.discoveryData.scan_folders.filter(f => {
            if (!baseDatasets.has(f.path.split('/')[0])) return false;
            return f.path.split('/').some(seg => {
                const s = seg.toLowerCase();
                return s === 'ad' || s.startsWith('ad_') || s === 'converter' || s.startsWith('converter_');
            });
        }).map(f => f.path);
        sib.forEach(f => { if (!folders.includes(f)) folders.push(f); });
    }
    return folders;
}

async function _fetchCohortStats(csvPath, folders) {
    const key = JSON.stringify([csvPath, [...folders].sort()]);
    if (state._cohortStatsCache.has(key)) return state._cohortStatsCache.get(key);
    if (!csvPath || !folders.length) return null;
    try {
        const r = await fetch(`/api/cohort/stats?csv_path=${encodeURIComponent(csvPath)}&scan_folders=${encodeURIComponent(folders.join(','))}`);
        if (!r.ok) return null;
        const json = await r.json();
        state._cohortStatsCache.set(key, json);
        return json;
    } catch (e) { console.warn('cohort/stats failed', e); return null; }
}

function _setPatientLoadingText(text) {
    const panel = document.querySelector('.tab-panel.active');
    if (!panel) return;
    const tabId = panel.dataset.tab;
    const skeletons = (tabId === 'overview') ? `
        <div class="skeleton skeleton-card"></div>
        <div class="skeleton skeleton-card"></div>
        <div class="skeleton skeleton-line medium"></div>
        <div class="skeleton skeleton-line short"></div>` :
        (tabId === 'manifold') ? `<div class="skeleton" style="height:480px;border-radius:12px"></div>` :
        (tabId === 'connectivity') ? `<div class="skeleton" style="height:540px;border-radius:8px;max-width:540px;margin:0 auto"></div>` :
        (tabId === 'qcviewer' || tabId === 'brainview') ? `<div class="skeleton" style="height:520px;border-radius:8px"></div>` :
        `<div class="skeleton skeleton-card"></div>`;
    panel.innerHTML = `
        <div style="font-size:.75rem;color:var(--text-2);text-align:center;margin-bottom:.6rem">${text}</div>
        ${skeletons}`;
}

async function _fetchTrajectoryStream(sid, folders, token, preferredVisit, signal) {
    if (!folders.length) return { sessions: [] };
    const query = new URLSearchParams({ scan_folders: folders.join(',') });
    if (preferredVisit) query.set('prioritize_visit', preferredVisit);
    const resp = await fetch(`/api/patient/${sid}/trajectory?${query.toString()}`, { signal });
    if (!resp.ok) throw new Error(`trajectory request failed (${resp.status})`);

    let trajectory = null;
    const processLine = (line) => {
        const trimmed = line.trim();
        if (!trimmed) return;
        const msg = JSON.parse(trimmed);
        if (!state.currentPatient || state.currentPatient._loadToken !== token) return;
        if (msg.type === 'progress') {
            const visit = msg.visit || 'unknown';
            _setPatientLoadingText(`Computing biomarkers for ${visit} (${msg.current}/${msg.total})…`);
            return;
        }
        if (msg.type === 'complete') trajectory = msg.data || { sessions: [] };
    };

    if (!resp.body || !resp.body.getReader) {
        const text = await resp.text();
        text.split('\n').forEach(processLine);
        return trajectory || { sessions: [] };
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let split = buffer.indexOf('\n');
        while (split >= 0) {
            processLine(buffer.slice(0, split));
            buffer = buffer.slice(split + 1);
            split = buffer.indexOf('\n');
        }
    }
    buffer += decoder.decode();
    if (buffer.trim()) processLine(buffer);
    return trajectory || { sessions: [] };
}

function _rebuildMergedVisits() {
    if (!state.currentPatient) return;
    const { traj, clinical, manifold } = state.currentPatient;
    const visitNum = v => { const m = String(v).match(/M(\d+)/i); return m ? parseInt(m[1]) : 999; };
    const allVisitSet = new Set();
    (traj?.sessions || []).forEach(s => allVisitSet.add(s.visit));
    (clinical?.visits || []).forEach(v => allVisitSet.add(v));
    const allVisits = Array.from(allVisitSet).sort((a, b) => visitNum(a) - visitNum(b));

    const mergedMap = new Map();
    (traj?.sessions || []).forEach(s => mergedMap.set(s.visit, { ...s }));
    if (clinical?.visits) {
        clinical.visits.forEach((v, i) => {
            if (!mergedMap.has(v)) mergedMap.set(v, { visit: v });
            const s = mergedMap.get(v);
            s.mmse    = clinical.cognitive?.mmse?.[i];
            s.cdr     = clinical.cognitive?.cdr?.[i];
            s.pacc5   = clinical.cognitive?.pacc5?.[i];
            s.abeta42 = clinical.csf?.abeta42?.[i];
            s.tau     = clinical.csf?.tau?.[i];
            s.ptau    = clinical.csf?.ptau?.[i];
        });
    }
    if (manifold?.trajectory) {
        manifold.trajectory.forEach(t => {
            const s = mergedMap.get(t.visit) || { visit: t.visit };
            s.manifold_x = t.x;
            s.manifold_y = t.y;
            s.conversion_score = t.conversion_score;
            mergedMap.set(t.visit, s);
        });
    }
    state.currentPatient.allVisits = allVisits;
    state.currentPatient.mergedVisits = Array.from(mergedMap.values())
        .sort((a, b) => visitNum(a.visit) - visitNum(b.visit));
}

function _computeAdVisits(clinical) {
    const set = new Set();
    if (clinical?.diagnosis && clinical?.visits) {
        clinical.diagnosis.forEach((diag, i) => {
            const isAd = String(diag).toLowerCase() === 'ad' || String(diag) === '5';
            if (isAd) set.add(clinical.visits[i]);
        });
    }
    return set;
}

function _applyIncludeADFilter(traj, clinical, manifold, includeAD, adVisits) {
    if (!adVisits) adVisits = _computeAdVisits(clinical);
    if (clinical) clinical.adVisits = Array.from(adVisits);
    if (includeAD) return adVisits;

    if (clinical?.visits) {
        const allowed = clinical.visits.map((v, i) => adVisits.has(v) ? -1 : i).filter(i => i !== -1);
        clinical.visits = allowed.map(i => clinical.visits[i]);
        if (clinical.diagnosis) clinical.diagnosis = allowed.map(i => clinical.diagnosis[i]);
        if (clinical.cognitive) {
            clinical.cognitive.mmse  = allowed.map(i => clinical.cognitive.mmse[i]);
            clinical.cognitive.cdr   = allowed.map(i => clinical.cognitive.cdr[i]);
            clinical.cognitive.pacc5 = allowed.map(i => clinical.cognitive.pacc5[i]);
        }
        if (clinical.csf) {
            clinical.csf.abeta42 = allowed.map(i => clinical.csf.abeta42[i]);
            clinical.csf.tau     = allowed.map(i => clinical.csf.tau[i]);
            clinical.csf.ptau    = allowed.map(i => clinical.csf.ptau[i]);
        }
    }
    if (traj?.sessions) traj.sessions = traj.sessions.filter(s => !adVisits.has(s.visit));
    if (manifold?.trajectory) manifold.trajectory = manifold.trajectory.filter(t => !adVisits.has(t.visit));
    return adVisits;
}

export async function loadPatientData() {
    if (!state.currentPatient) return;
    const { sid, includeAD } = state.currentPatient;
    const myToken = ++state.currentPatient._loadToken || (state.currentPatient._loadToken = 1);
    const csvPath = $('csvSelect').value;
    const folders = _resolveQueryFolders(includeAD);
    const preferredVisit = state.currentPatient.selectedVisit;
    if (state._activeTrajectoryController) state._activeTrajectoryController.abort();
    const trajectoryController = new AbortController();
    state._activeTrajectoryController = trajectoryController;

    _setPatientLoadingText('Loading patient data…');

    try {
        const trajP = folders.length
            ? _fetchTrajectoryStream(sid, folders, myToken, preferredVisit, trajectoryController.signal)
            : Promise.resolve({ sessions: [] });
        const clinP = csvPath
            ? fetch(`/api/patient/${sid}/clinical?csv_path=${encodeURIComponent(csvPath)}`).then(r => r.ok ? r.json() : {})
            : Promise.resolve({});
        const scansP = folders.length
            ? fetch(`/api/patient/${sid}/scans?scan_folders=${encodeURIComponent(folders.join(','))}`).then(r => r.ok ? r.json() : { scans: [] }).catch(() => ({ scans: [] }))
            : Promise.resolve({ scans: [] });

        const [traj, clinical, scansResp] = await Promise.all([trajP, clinP, scansP]);
        if (!traj) return;
        if (!state.currentPatient || state.currentPatient._loadToken !== myToken) return;

        const adVisits = _applyIncludeADFilter(traj, clinical, null, includeAD);

        state.currentPatient.traj = traj;
        state.currentPatient.clinical = clinical;
        state.currentPatient.manifold = null;
        state.currentPatient.cohortStats = null;
        state.currentPatient.scansList = scansResp?.scans || [];
        state.currentPatient._adVisits = adVisits;

        _rebuildMergedVisits();
        state.currentPatient.selectedVisit = state.currentPatient.mergedVisits.length
            ? state.currentPatient.mergedVisits[state.currentPatient.mergedVisits.length - 1].visit
            : null;
        _updateCurrentVisitBadge(state.currentPatient.selectedVisit);
        _renderGlobalVisitPills();
        state.currentPatient.tabRendered = { overview: false, staging: false, manifold: false, connectivity: false, qcviewer: false, brainview: false };
        renderActiveTab();
    } catch (e) {
        if (e?.name === 'AbortError') return;
        console.error('fast patient load failed', e);
        const ap = document.querySelector('.tab-panel.active');
        if (ap) ap.innerHTML = '<p class="no-trajectory">Error loading patient data</p>';
        return;
    } finally {
        if (state._activeTrajectoryController === trajectoryController) {
            state._activeTrajectoryController = null;
        }
    }

    if (!csvPath || !folders.length) return;
    try {
        const [cohortStats, manResp] = await Promise.all([
            _fetchCohortStats(csvPath, folders),
            fetch(`/api/patient/${sid}/manifold?csv_path=${encodeURIComponent(csvPath)}&scan_folders=${encodeURIComponent(folders.join(','))}`)
                .then(r => r.ok ? r.json() : null).catch(() => null),
        ]);
        if (!state.currentPatient || state.currentPatient._loadToken !== myToken) return;

        _applyIncludeADFilter(null, null, manResp, includeAD, state.currentPatient._adVisits);

        state.currentPatient.cohortStats = cohortStats;
        state.currentPatient.manifold = manResp;
        _rebuildMergedVisits();

        state.currentPatient.tabRendered.overview = false;
        state.currentPatient.tabRendered.manifold = false;
        const t = state.currentPatient.activeTab;
        if (t === 'overview' || t === 'manifold') renderActiveTab();
    } catch (e) {
        console.warn('manifold/cohort load failed', e);
    }
}

function _updateCurrentVisitBadge(visit) {
    const badge = document.getElementById('currentVisitBadge');
    const text = document.getElementById('currentVisitText');
    if (!badge || !text) return;
    if (visit) { text.textContent = visit; badge.style.display = 'inline-flex'; }
    else { badge.style.display = 'none'; }
}

function _renderGlobalVisitPills() {
    const host = document.getElementById('globalVisitPills');
    if (!host || !state.currentPatient) return;
    const visits = state.currentPatient.allVisits || [];
    if (!visits.length) { host.innerHTML = `<span class="vbr-empty">— no visits available —</span>`; return; }
    const sel = state.currentPatient.selectedVisit;
    host.innerHTML = visits.map(v =>
        `<button class="visit-pill${v === sel ? ' active' : ''}" data-visit="${v}">${v}</button>`).join('');
    host.querySelectorAll('.visit-pill').forEach(p => {
        p.addEventListener('click', () => setSelectedVisit(p.dataset.visit));
        p.addEventListener('mouseenter', () => _prefetchQCVolume(p.dataset.visit));
    });
}

export function setSelectedVisit(visit) {
    if (!state.currentPatient || !visit) return;
    if (state.currentPatient.selectedVisit === visit) return;
    state.currentPatient.selectedVisit = visit;
    _updateCurrentVisitBadge(visit);
    document.querySelectorAll('#globalVisitPills .visit-pill').forEach(p =>
        p.classList.toggle('active', p.dataset.visit === visit));
    document.querySelectorAll('#tab-overview .session-table tbody tr').forEach(tr => {
        tr.classList.toggle('selected', tr.dataset.visit === visit);
    });
    if (state.currentPatient.tabRendered.connectivity) renderConnectivityHeatmap();
    if (state.currentPatient.tabRendered.qcviewer)     loadQCViewerVolume();
    if (state.currentPatient.tabRendered.brainview)    renderBrainGraph();
    if (state.currentPatient.tabRendered.manifold)     redrawManifold();
}

export function closeModal() {
    $('modalBackdrop').classList.remove('open');
    $('patientModal').classList.remove('brainview-expanded');
    document.querySelectorAll('.data-table tbody tr.selected').forEach(tr => tr.classList.remove('selected'));
    ['trajFC', 'trajMod', 'trajCog', 'trajCSF', 'convScore'].forEach(k => {
        if (state.activeCharts[k]) { state.activeCharts[k].destroy(); delete state.activeCharts[k]; }
    });
    if (state.currentPatient) {
        if (state.currentPatient._brainPlayTimer) {
            clearInterval(state.currentPatient._brainPlayTimer);
            state.currentPatient._brainPlayTimer = null;
        }
        state.currentPatient.niiVue = null;
        state.currentPatient.brainNv = null;
    }
    state.currentPatient = null;
}
