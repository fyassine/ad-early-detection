import { $, showLoading, hideLoading } from './utils.js';
import { state } from './state.js';
import { analyze, applyFilter, filteredRows } from './dashboard.js';
import { closeModal, setSelectedVisit, switchTab } from './modal.js';
import { resetBrainView } from './tabs/brainview.js';
import { renderRows } from './table.js';

// ── Init ──────────────────────────────────────────────────────────────────────

export async function init() {
    showLoading('Discovering data…');
    try {
        const r = await fetch('/api/discover');
        state.discoveryData = await r.json();
        populateCSVs(state.discoveryData.csvs);
        setupFolderDropdown(state.discoveryData.scan_folders);
        $('statusText').textContent =
            `${state.discoveryData.csvs.length} CSVs · ${state.discoveryData.scan_folders.length} scan folders`;
    } catch {
        $('statusText').textContent = 'Connection error';
    }
    hideLoading();

    renderRecentSearches();
    $('csvSelect').addEventListener('change', checkReady);
    $('btnAnalyze').addEventListener('click', analyze);
    $('tableSearch').addEventListener('input', () => renderRows(filteredRows()));
    $('globalCohortSelect').addEventListener('change', e => {
        state.activeFilter = e.target.value ? { field: 'diagnosis', value: e.target.value } : null;
        applyFilter();
    });
    $('btnCloseModal').addEventListener('click', closeModal);
    $('modalBackdrop').addEventListener('click', e => {
        if (e.target === $('modalBackdrop')) closeModal();
    });

    document.addEventListener('keydown', _patientModalKeyHandler);

    document.querySelectorAll('.section-header').forEach(hdr => {
        hdr.addEventListener('click', () => {
            const sec = hdr.closest('.dash-section');
            if (sec) sec.classList.toggle('collapsed');
        });
    });
}

const TAB_IDS = ['overview', 'manifold', 'connectivity', 'qcviewer', 'brainview'];

function _patientModalKeyHandler(e) {
    if (!$('modalBackdrop')?.classList.contains('open')) return;
    const target = e.target;
    if (target && /input|textarea|select/i.test(target.tagName)) return;

    if (e.key === 'Escape') { closeModal(); return; }
    if (!state.currentPatient) return;

    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        e.preventDefault();
        const visits = state.currentPatient.allVisits || [];
        if (visits.length < 2) return;
        const cur = visits.indexOf(state.currentPatient.selectedVisit);
        const idx = e.key === 'ArrowRight'
            ? (cur < 0 ? 0 : (cur + 1) % visits.length)
            : (cur < 0 ? visits.length - 1 : (cur - 1 + visits.length) % visits.length);
        setSelectedVisit(visits[idx]);
        return;
    }

    if (/^[1-5]$/.test(e.key)) {
        e.preventDefault();
        const tabId = TAB_IDS[parseInt(e.key, 10) - 1];
        if (tabId) switchTab(tabId);
        return;
    }

    if ((e.key === 'r' || e.key === 'R') && state.currentPatient.activeTab === 'brainview') {
        e.preventDefault();
        resetBrainView();
    }
}

// ── CSV selector ──────────────────────────────────────────────────────────────

export function populateCSVs(csvs) {
    const sel = $('csvSelect');
    csvs.forEach(csv => {
        const o = document.createElement('option');
        o.value = csv.path;
        const parts = csv.path.replace(/\\/g, '/').split('/');
        o.textContent = parts.slice(-3).join(' / ');
        o.title = csv.path;
        sel.appendChild(o);
    });
}

// ── Folder multi-select ────────────────────────────────────────────────────────

export function setupFolderDropdown(folders) {
    const $folderDropdown = $('folderDropdown');
    const $folderSearch = $('folderSearch');
    if (!folders.length) {
        $folderDropdown.innerHTML = '<div style="padding:.5rem;color:var(--text-2);font-size:.75rem">No folders</div>';
        return;
    }

    document.addEventListener('click', e => {
        if (!e.target.closest('#folderMultiSelect')) $folderDropdown.classList.remove('open');
    });
    $folderSearch.addEventListener('focus', () => { renderDropdown(); $folderDropdown.classList.add('open'); });
    $folderSearch.addEventListener('input', () => {
        renderDropdown($folderSearch.value.toLowerCase());
        $folderDropdown.classList.add('open');
    });
}

export function renderDropdown(query = '') {
    const folders = state.discoveryData.scan_folders;
    const $folderDropdown = $('folderDropdown');
    $folderDropdown.innerHTML = '';
    let matches = 0;

    folders.forEach(f => {
        if (state.selectedScanFolders.includes(f.path)) return;
        const parts = f.path.replace(/\\/g, '/').split('/');
        const shortPath = parts.slice(-3).join(' / ');
        if (query && !f.path.toLowerCase().includes(query)) return;
        matches++;

        const bc = f.file_type === 'nii.gz' ? 'badge-nii' : f.file_type === 'npz' ? 'badge-npz' : 'badge-mixed';
        const el = document.createElement('div');
        el.className = 'folder-item';
        el.title = f.path;
        el.innerHTML = `
            <span class="folder-path">${shortPath}</span>
            <span class="badge ${bc}">${f.file_type}</span>
            <span style="font-size:.7rem;color:var(--text-2);white-space:nowrap">${f.scan_count} · ${f.subject_count} subj</span>`;
        el.addEventListener('click', () => {
            state.selectedScanFolders.push(f.path);
            $('folderSearch').value = '';
            $folderDropdown.classList.remove('open');
            renderTokens();
            checkReady();
            updateFormatBadge();
        });
        $folderDropdown.appendChild(el);
    });

    if (matches === 0) {
        $folderDropdown.innerHTML = '<div style="padding:.5rem;color:var(--text-2);font-size:.75rem">No matching folders</div>';
    }
}

export function renderTokens() {
    const folders = state.discoveryData.scan_folders;
    const container = $('selectedFoldersContainer');
    container.innerHTML = '';
    state.selectedScanFolders.forEach(path => {
        const f = folders.find(x => x.path === path);
        if (!f) return;
        const parts = f.path.replace(/\\/g, '/').split('/');
        const shortPath = parts.slice(-2).join(' / ');
        const bc = f.file_type === 'nii.gz' ? 'badge-nii' : f.file_type === 'npz' ? 'badge-npz' : 'badge-mixed';
        const token = document.createElement('div');
        token.className = 'folder-token';
        token.innerHTML = `
            <span class="badge ${bc}" style="margin-right:.4rem;font-size:.6rem">${f.file_type}</span>
            <span title="${f.path}">${shortPath}</span>
            <span class="btn-remove" title="Remove">×</span>`;
        token.querySelector('.btn-remove').addEventListener('click', () => {
            state.selectedScanFolders = state.selectedScanFolders.filter(p => p !== path);
            renderTokens();
            checkReady();
            updateFormatBadge();
        });
        container.appendChild(token);
    });
}

export function updateFormatBadge() {
    const $formatBadge = $('formatBadge');
    if (!state.selectedScanFolders.length) { $formatBadge.classList.remove('visible'); return; }
    const types = new Map();
    state.selectedScanFolders.forEach(p => {
        const f = state.discoveryData.scan_folders.find(x => x.path === p);
        if (!f) return;
        types.set(f.file_type, (types.get(f.file_type) || 0) + 1);
    });
    if (types.size > 1) {
        const summary = Array.from(types.entries()).map(([t, n]) => `${n} <code>${t}</code>`).join(' + ');
        $formatBadge.innerHTML = `📦 ${summary}`;
        $formatBadge.classList.add('visible');
        return;
    }
    const folder = state.discoveryData.scan_folders.find(f => f.path === state.selectedScanFolders[0]);
    if (folder?.format_info) {
        const fi = folder.format_info;
        let desc = fi.description || fi.type;
        if (fi.parcellation) desc += ` · ${fi.parcellation}`;
        if (fi.sample_size_mb) desc += ` · ~${fi.sample_size_mb}MB`;
        $formatBadge.innerHTML = `📦 <code>${desc}</code>`;
        $formatBadge.classList.add('visible');
    } else {
        $formatBadge.classList.remove('visible');
    }
}

export function checkReady() {
    $('btnAnalyze').disabled = !($('csvSelect').value || state.selectedScanFolders.length);
}

// ── Saved Workspaces (localStorage) ──────────────────────────────────────────

const RECENT_KEY = 'fmri_saved_workspaces';

export function getRecent() {
    try { return JSON.parse(localStorage.getItem(RECENT_KEY) || '[]'); } catch { return []; }
}

export function saveRecent(csv, folders) {
    const csvShort = csv ? csv.split('/').slice(-1)[0] : 'Unknown CSV';
    const defaultName = `${csvShort} · ${folders.length} folder${folders.length === 1 ? '' : 's'}`;
    const workspaceName = prompt('Save workspace as:', defaultName);
    if (workspaceName === null) return;
    const item = { name: workspaceName || defaultName, csv, folders, date: new Date().toISOString() };
    const list = getRecent().filter(r => !(r.csv === csv && JSON.stringify(r.folders) === JSON.stringify(folders)));
    list.unshift(item);
    localStorage.setItem(RECENT_KEY, JSON.stringify(list.slice(0, 8)));
    renderRecentSearches();
}

export function renderRecentSearches() {
    const list = getRecent();
    const container = $('recentSearches');
    const ul = $('recentList');
    if (!list.length) { container.style.display = 'none'; return; }
    container.style.display = '';
    ul.innerHTML = list.map((r, i) => {
        const d = new Date(r.date);
        const dateStr = d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const titleText = `CSV: ${r.csv}\nFolders: ${r.folders.join(', ')}`;
        return `<span class="recent-chip" title="${titleText}" onclick="applyRecent(${i})">
            <span>${r.name || 'Saved Workspace'}</span>
            <span class="chip-date">${dateStr}</span>
            <span class="chip-del" onclick="deleteRecent(event,${i})">×</span>
        </span>`;
    }).join('');
}

export function applyRecent(i) {
    const item = getRecent()[i];
    if (!item) return;
    $('csvSelect').value = item.csv;
    state.selectedScanFolders = [...item.folders];
    renderTokens();
    checkReady();
    updateFormatBadge();
}

export function deleteRecent(e, i) {
    e.stopPropagation();
    const list = getRecent();
    list.splice(i, 1);
    localStorage.setItem(RECENT_KEY, JSON.stringify(list));
    renderRecentSearches();
}
