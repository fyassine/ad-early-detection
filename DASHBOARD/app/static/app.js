/**
 * app.js — fMRI Data Dashboard v3
 * Semantic colors · Cross-filtering · Modal · Recent searches · Synced axes
 */

// ── Diagnosis semantic colors (consistent everywhere) ──
const DIAG_COLORS = {
    healthy:   '#4f98a3',
    scd:       '#6daa45',
    mci:       '#e8af34',
    converter: '#e08040',
    ad:        '#d163a7',
    relative:  '#7a7976',
};
function diagColor(label) {
    return DIAG_COLORS[String(label).toLowerCase()] || '#4f98a3';
}

const C = {
    indigo:'#4f98a3', violet:'#6daa45', sky:'#e8af34',
    green:'#6daa45', amber:'#e8af34', rose:'#d163a7',
    cyan:'#4f98a3', orange:'#e08040',
};
const BAR_COLORS = [C.indigo,C.violet,C.sky,C.green,C.amber,C.rose,C.cyan,C.orange];

// ── State ──
let discoveryData = null;
let activeCharts = {};
let patientData = [];
let scanSubjects = new Set();
let selectedScanFolders = [];
let globalMeta = null, filteredMeta = null, lastScan = null;
let activeFilter = null; // { field:'diagnosis', value:'converter' }
let sortCol = null, sortDir = 'asc';

// ── DOM ──
const $ = id => document.getElementById(id);
const $loading     = $('loadingOverlay');
const $loadingText = $('loadingText');
const $status      = $('statusText');
const $csvSelect   = $('csvSelect');
const $folderSearch = $('folderSearch');
const $folderDropdown = $('folderDropdown');
const $selectedFoldersContainer = $('selectedFoldersContainer');
const $btnAnalyze  = $('btnAnalyze');
const $formatBadge = $('formatBadge');
const $emptyState  = $('emptyState');
const $results     = $('dashboardResults');
const $summaryCards= $('summaryCards');
const $tableHead   = $('tableHead');
const $tableBody   = $('tableBody');
const $tableSearch = $('tableSearch');
const $tableCount  = $('tableCount');
const $globalCohortSelect = $('globalCohortSelect');
const $dashboardControls = $('dashboardControls');

// ── Init ──
document.addEventListener('DOMContentLoaded', init);

async function init() {
    if (typeof Chart === 'undefined') {
        $('statusText').textContent = 'Error: Chart.js failed to load — reload page';
        hideLoading(); return;
    }
    Chart.defaults.color = '#7a7976';
    Chart.defaults.borderColor = 'rgba(255,255,255,0.08)';
    Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
    Chart.defaults.font.size = 11;

    showLoading('Discovering data…');
    try {
        const r = await fetch('/api/discover');
        discoveryData = await r.json();
        populateCSVs(discoveryData.csvs);
        setupFolderDropdown(discoveryData.scan_folders);
        $status.textContent = `${discoveryData.csvs.length} CSVs · ${discoveryData.scan_folders.length} scan folders`;
    } catch(e) { $status.textContent = 'Connection error'; }
    hideLoading();

    renderRecentSearches();
    $csvSelect.addEventListener('change', checkReady);
    $btnAnalyze.addEventListener('click', analyze);
    $tableSearch.addEventListener('input', () => renderRows(filteredRows()));
    $globalCohortSelect.addEventListener('change', (e) => {
        if (e.target.value) {
            activeFilter = { field: 'diagnosis', value: e.target.value };
        } else {
            activeFilter = null;
        }
        applyFilter();
    });
    $('btnCloseModal').addEventListener('click', closeModal);
    $('modalBackdrop').addEventListener('click', e => { if(e.target===$('modalBackdrop')) closeModal(); });
    
    // Collapsible sections
    document.querySelectorAll('.section-header').forEach(hdr => {
        hdr.addEventListener('click', () => {
            const sec = hdr.closest('.dash-section');
            if(sec) sec.classList.toggle('collapsed');
        });
    });
}

// ── Config: CSV (show short name, full path as title) ──
function populateCSVs(csvs) {
    csvs.forEach(csv => {
        const o = document.createElement('option');
        o.value = csv.path;
        const parts = csv.path.replace(/\\/g,'/').split('/');
        o.textContent = parts.slice(-3).join(' / ');
        o.title = csv.path;
        $csvSelect.appendChild(o);
    });
}

function setupFolderDropdown(folders) {
    if (!folders.length) {
        $folderDropdown.innerHTML = '<div style="padding:.5rem;color:var(--text-2);font-size:.75rem">No folders</div>';
        return;
    }
    
    // Close dropdown on outside click
    document.addEventListener('click', e => {
        if (!e.target.closest('#folderMultiSelect')) $folderDropdown.classList.remove('open');
    });

    $folderSearch.addEventListener('focus', () => {
        renderDropdown();
        $folderDropdown.classList.add('open');
    });

    $folderSearch.addEventListener('input', () => {
        renderDropdown($folderSearch.value.toLowerCase());
        $folderDropdown.classList.add('open');
    });
}

function renderDropdown(query = '') {
    const folders = discoveryData.scan_folders;
    $folderDropdown.innerHTML = '';
    let matches = 0;

    // Mixed selection is now allowed: npz folders feed the matrix-based tabs
    // (Overview / Manifold / Connectivity / Brain View) and nii.gz folders feed
    // the QC viewer. Both backend walkers filter by extension, so a mixed list
    // is safe to pass to every endpoint.
    folders.forEach(f => {
        if (selectedScanFolders.includes(f.path)) return; // Hide already selected
        const parts = f.path.replace(/\\/g,'/').split('/');
        const shortPath = parts.slice(-3).join(' / ');

        if (query && !f.path.toLowerCase().includes(query)) return;
        matches++;

        const bc = f.file_type==='nii.gz'?'badge-nii':f.file_type==='npz'?'badge-npz':'badge-mixed';

        const el = document.createElement('div');
        el.className = 'folder-item';
        el.title = f.path;
        el.innerHTML = `
            <span class="folder-path">${shortPath}</span>
            <span class="badge ${bc}">${f.file_type}</span>
            <span style="font-size:.7rem;color:var(--text-2);white-space:nowrap">${f.scan_count} · ${f.subject_count} subj</span>`;

        el.addEventListener('click', () => {
            selectedScanFolders.push(f.path);
            $folderSearch.value = '';
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

function renderTokens() {
    $selectedFoldersContainer.innerHTML = '';
    const folders = discoveryData.scan_folders;
    selectedScanFolders.forEach(path => {
        const f = folders.find(x => x.path === path);
        if (!f) return;
        const parts = f.path.replace(/\\/g,'/').split('/');
        const shortPath = parts.slice(-2).join(' / ');
        const bc = f.file_type==='nii.gz'?'badge-nii':f.file_type==='npz'?'badge-npz':'badge-mixed';

        const token = document.createElement('div');
        token.className = 'folder-token';
        token.innerHTML = `
            <span class="badge ${bc}" style="margin-right:.4rem;font-size:.6rem">${f.file_type}</span>
            <span title="${f.path}">${shortPath}</span>
            <span class="btn-remove" title="Remove">×</span>
        `;
        token.querySelector('.btn-remove').addEventListener('click', () => {
            selectedScanFolders = selectedScanFolders.filter(p => p !== path);
            renderTokens();
            checkReady();
            updateFormatBadge();
        });
        $selectedFoldersContainer.appendChild(token);
    });
}

function updateFormatBadge() {
    if (!selectedScanFolders.length) { $formatBadge.classList.remove('visible'); return; }
    // Summarise mixed types: how many .npz, how many .nii.gz
    const types = new Map();
    selectedScanFolders.forEach(p => {
        const f = discoveryData.scan_folders.find(x => x.path === p);
        if (!f) return;
        types.set(f.file_type, (types.get(f.file_type) || 0) + 1);
    });
    if (types.size > 1) {
        const summary = Array.from(types.entries())
            .map(([t, n]) => `${n} <code>${t}</code>`).join(' + ');
        $formatBadge.innerHTML = `📦 ${summary}`;
        $formatBadge.classList.add('visible');
        return;
    }
    const folder = discoveryData.scan_folders.find(f=>f.path===selectedScanFolders[0]);
    if (folder?.format_info) {
        const fi = folder.format_info;
        let desc = fi.description||fi.type;
        if (fi.parcellation) desc += ` · ${fi.parcellation}`;
        if (fi.sample_size_mb) desc += ` · ~${fi.sample_size_mb}MB`;
        $formatBadge.innerHTML = `📦 <code>${desc}</code>`;
        $formatBadge.classList.add('visible');
    } else $formatBadge.classList.remove('visible');
}

function checkReady() {
    $btnAnalyze.disabled = !($csvSelect.value || selectedScanFolders.length);
}

// ── Saved Workspaces (localStorage) ──
const RECENT_KEY = 'fmri_saved_workspaces';

function getRecent() {
    try { return JSON.parse(localStorage.getItem(RECENT_KEY)||'[]'); } catch { return []; }
}
function saveRecent(csv, folders) {
    const csvShort = csv ? csv.split('/').slice(-1)[0] : 'Unknown CSV';
    const defaultName = `${csvShort} · ${folders.length} folder${folders.length===1?'':'s'}`;
    
    // Prompt for workspace name
    const workspaceName = prompt("Save workspace as:", defaultName);
    if (workspaceName === null) return; // cancelled
    
    const item = { name: workspaceName || defaultName, csv, folders, date: new Date().toISOString() };
    const list = getRecent().filter(r => !(r.csv===csv && JSON.stringify(r.folders)===JSON.stringify(folders)));
    list.unshift(item);
    localStorage.setItem(RECENT_KEY, JSON.stringify(list.slice(0,8)));
    renderRecentSearches();
}
function renderRecentSearches() {
    const list = getRecent();
    const container = $('recentSearches');
    const ul = $('recentList');
    if (!list.length) { container.style.display='none'; return; }
    container.style.display='';
    ul.innerHTML = list.map((r,i) => {
        const d = new Date(r.date);
        const dateStr = d.toLocaleDateString()+' '+d.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
        const titleText = `CSV: ${r.csv}\nFolders: ${r.folders.join(', ')}`;
        return `<span class="recent-chip" title="${titleText}" onclick="applyRecent(${i})">
            <span>${r.name || 'Saved Workspace'}</span>
            <span class="chip-date">${dateStr}</span>
            <span class="chip-del" onclick="deleteRecent(event,${i})">×</span>
        </span>`;
    }).join('');
}
window.applyRecent = function(i) {
    const item = getRecent()[i];
    if (!item) return;
    $csvSelect.value = item.csv;
    selectedScanFolders = [...item.folders];
    renderTokens();
    checkReady(); updateFormatBadge();
};
window.deleteRecent = function(e, i) {
    e.stopPropagation();
    const list = getRecent(); list.splice(i,1);
    localStorage.setItem(RECENT_KEY, JSON.stringify(list));
    renderRecentSearches();
};

// ── Analyze ──
async function analyze() {
    const csvPath = $csvSelect.value;
    activeFilter = null;
    showLoading('Analyzing…');
    // Invalidate cross-patient caches: cohort/UMAP, CN reference, atlas
    // coords are keyed by file content and stay valid; clear the JS-level
    // ones whose keys aren't watertight.
    _cohortStatsCache.clear();
    Object.keys(_cohortRefMatrices).forEach(k => delete _cohortRefMatrices[k]);
    try {
        lastScan = null; globalMeta = null; filteredMeta = null;
        if (selectedScanFolders.length) {
            $loadingText.textContent = 'Scanning folders…';
            const r = await fetch(`/api/scan?folders=${encodeURIComponent(selectedScanFolders.join(','))}`);
            lastScan = await r.json();
            scanSubjects = new Set(Object.keys(lastScan.subject_scan_counts||{}));
        }
        if (csvPath) {
            $loadingText.textContent = 'Parsing metadata…';
            const p = new URLSearchParams({csv_path:csvPath});
            if (selectedScanFolders.length) p.set('scan_folders', selectedScanFolders.join(','));
            const r = await fetch(`/api/metadata?${p}`);
            globalMeta = await r.json();
            filteredMeta = globalMeta;
        }
        if (csvPath || selectedScanFolders.length) {
            const existing = getRecent().some(r => r.csv===csvPath && JSON.stringify(r.folders)===JSON.stringify(selectedScanFolders));
            if (!existing) saveRecent(csvPath, selectedScanFolders);
        }
        render(lastScan, globalMeta, filteredMeta);

        // Pre-warm the cohort/UMAP cache in the background — first fit is
        // ~5min on the full DELCODE cohort, so kicking it off here means
        // the cache is usually ready by the time the user opens a patient.
        if (csvPath && selectedScanFolders.length) {
            fetch(`/api/cohort/warmup?csv_path=${encodeURIComponent(csvPath)}` +
                  `&scan_folders=${encodeURIComponent(selectedScanFolders.join(','))}`)
                .catch(() => {});
        }
    } catch(e) { console.error(e); alert('Analysis failed'); }
    hideLoading();
}

// ── Render all sections ──
function render(scan, global, filtered) {
    Object.values(activeCharts).forEach(c=>c.destroy());
    activeCharts = {};
    $emptyState.style.display = 'none';
    $results.classList.add('active');
    
    // Populate cohort dropdown based on global meta
    if (global?.diagnosis_distribution) {
        const labels = Object.keys(global.diagnosis_distribution);
        $globalCohortSelect.innerHTML = '<option value="">— All Patients —</option>' + 
            labels.map(l => `<option value="${l}">${l} (${global.diagnosis_distribution[l]})</option>`).join('');
        if (activeFilter && activeFilter.field === 'diagnosis') {
            $globalCohortSelect.value = activeFilter.value;
        }
        $dashboardControls.style.display = 'flex';
    } else {
        $dashboardControls.style.display = 'none';
    }

    renderSummary(scan, global);
    renderDemo(filtered);
    renderDiag(global, scan); // Diagnosis shows global context
    renderScans(scan, filtered);
    renderClinical(filtered);
    renderTable(filtered, scan);
}

// ── Summary Cards ──
function renderSummary(scan, meta) {
    const cards = [];
    if (meta) {
        const rows = meta.total_rows ? `${meta.total_rows} rows` : null;
        cards.push({icon:'👥', value:meta.unique_subjects||'—', label:'Subjects (CSV)', sub: rows});
    }
    if (scan) {
        cards.push({icon:'🧲', value:scan.total_scans, label:'Files on Disk'});
        cards.push({icon:'📁', value:scan.total_subjects, label:'Subjects w/ Scans'});
        const visitCount = scan.visit_distribution
            ? Object.values(scan.visit_distribution).reduce((a,b)=>a+b,0)
            : null;
        if (visitCount !== null) cards.push({icon:'🗓️', value:visitCount, label:'Visits on Disk'});
        if (scan.longitudinal_subjects>0) cards.push({icon:'🔄', value:scan.longitudinal_subjects, label:'Longitudinal'});
        if (scan.format_info?.description) {
            const formatLabel = scan.format_info.description.split('·')[0].trim();
            const formatSub = scan.format_info.parcellation || scan.format_info.type || null;
            cards.push({icon:'📦', value:formatLabel, label:'Format', sub: formatSub, kind:'text'});
        }
    }
    if (meta?.scan_coverage) {
        const c=meta.scan_coverage, pct=c.metadata_subjects>0?Math.round(c.matched/c.metadata_subjects*100):0;
        cards.push({icon:'🎯', value:`${pct}%`, label:'Scan Coverage'});
    }
    if (meta?.age_stats) cards.push({icon:'📅', value:`${meta.age_stats.mean}±${meta.age_stats.std}`, label:'Age (mean±SD)'});
    $summaryCards.innerHTML = cards.map(c=>{
        const valueClass = c.kind === 'text' ? 'card-value is-text' : 'card-value';
        const sub = c.sub ? `<div class="card-sub">${c.sub}</div>` : '';
        return `
        <div class="summary-card">
            <div class="card-top">
                <div class="card-icon">${c.icon}</div>
                <div class="card-label">${c.label}</div>
            </div>
            <div class="${valueClass}">${c.value}</div>
            ${sub}
        </div>`;
    }).join('');
}

// ── Demographics: Sex + Age ──
function renderDemo(meta) {
    const sec = $('sectionDemo'); const cont = $('demoCharts');
    cont.innerHTML='';
    if (!meta?.sex_distribution && !meta?.age_histogram) { sec.style.display='none'; return; }
    sec.style.display='';
    if (meta.sex_distribution) addHBar('demoCharts','sex','Sex',meta.sex_distribution, BAR_COLORS);
    if (meta.age_histogram) addVBar('demoCharts','age','Age Distribution',meta.age_histogram);
}

// ── Diagnosis: grouped bar (patients + scans) ──
function renderDiag(meta, scan) {
    const sec = $('sectionDiag'); const cont = $('diagCharts');
    cont.innerHTML='';
    if (!meta?.diagnosis_distribution) { sec.style.display='none'; return; }
    sec.style.display='';

    const labels = Object.keys(meta.diagnosis_distribution);
    const patVals = labels.map(k=>meta.diagnosis_distribution[k]);
    const visitVals = meta.diagnosis_visits ? labels.map(k=>meta.diagnosis_visits[k]||0) : null;
    const scanVals = meta.diagnosis_scans ? labels.map(k=>meta.diagnosis_scans[k]||0) : null;
    const bgColors = labels.map(l=>diagColor(l));
    const bgColorsDim = labels.map(l=>diagColor(l)+'55');
    const bgColorsMid = labels.map(l=>diagColor(l)+'99');

    const card = document.createElement('div');
    card.className = 'chart-card';
    card.innerHTML = `<h3>Diagnosis — Patients${visitVals?' &amp; Visits':''}${scanVals?' &amp; Scans':''} <span style="font-size:.65rem;color:var(--text-3);font-weight:400">(click to filter)</span></h3><div class="chart-container"><canvas id="chart-diag"></canvas></div>`;
    cont.appendChild(card);

    const datasets = [{
        label: 'Patients',
        data: patVals,
        backgroundColor: bgColorsDim,
        borderWidth: 0, borderRadius: 4, barThickness: 24,
        grouped: false, // Bullet chart effect (overlap)
    }];
    if (visitVals) datasets.push({
        label: 'Visits',
        data: visitVals,
        backgroundColor: bgColorsMid,
        borderWidth: 0, borderRadius: 4, barThickness: 14,
        grouped: false,
    });
    if (scanVals) datasets.push({
        label: 'Scans',
        data: scanVals,
        backgroundColor: bgColors,
        borderWidth: 0, borderRadius: 4, barThickness: 8,
        grouped: false,
    });

    const ctx = document.getElementById('chart-diag').getContext('2d');
    activeCharts['diag'] = new Chart(ctx, {
        type: 'bar',
        data: { labels, datasets },
        options: {
            indexAxis: 'y',
            responsive: true, maintainAspectRatio: false,
            layout: { padding: { right: 120 } }, // Make room for custom text
            plugins: {
                legend: { display: !!scanVals || !!visitVals, position: 'top', labels: { usePointStyle:true, boxWidth:10, padding:14 } },
                tooltip: tooltipStyle(),
            },
            scales: {
                x: { display: false, stacked: false },
                y: { grid:{display:false}, stacked: false, ticks:{color:'#cbd5e1', font:{size:12, weight:500}} },
            },
            onClick(e, els) {
                if (!els.length) return;
                const label = labels[els[0].index];
                if (activeFilter?.value === label) {
                    activeFilter = null;
                } else {
                    activeFilter = { field:'diagnosis', value: label };
                }
                if ($globalCohortSelect) $globalCohortSelect.value = activeFilter ? activeFilter.value : "";
                applyFilter();
            },
            animation: { duration:500 },
        },
        plugins: [{
            afterDatasetsDraw(chart) {
                const ctx2 = chart.ctx;
                const total = patVals.reduce((a,b)=>a+b,0);
                const idxVisits = visitVals ? 1 : null;
                const idxScans = scanVals ? (visitVals ? 2 : 1) : null;
                chart.data.datasets[0].data.forEach((v,i) => {
                    const m = chart.getDatasetMeta(0).data[i];
                    if (!m) return;
                    const pct = total>0?Math.round(v/total*100):0;
                    const mVisit = idxVisits !== null ? chart.getDatasetMeta(idxVisits).data[i] : null;
                    const mScan = idxScans !== null ? chart.getDatasetMeta(idxScans).data[i] : null;
                    const max_x = Math.max(
                        m.x,
                        mVisit ? mVisit.x : 0,
                        mScan ? mScan.x : 0
                    );
                    
                    ctx2.save();
                    ctx2.textBaseline='middle';
                    ctx2.font='500 11px Inter,sans-serif';
                    
                    let curX = max_x + 8;
                    
                    ctx2.fillStyle='#94a3b8'; 
                    const tPat = `${v} pat. (${pct}%)`;
                    ctx2.fillText(tPat, curX, m.y);
                    curX += ctx2.measureText(tPat).width;
                    
                    if (visitVals) {
                        ctx2.fillStyle='#64748b';
                        const sep = `  ·  `;
                        ctx2.fillText(sep, curX, m.y);
                        curX += ctx2.measureText(sep).width;
                        ctx2.fillStyle=bgColorsMid[i];
                        const tVis = `${visitVals[i]} visits`;
                        ctx2.fillText(tVis, curX, m.y);
                        curX += ctx2.measureText(tVis).width;
                    }

                    if (scanVals) {
                        ctx2.fillStyle='#64748b';
                        const sep = `  ·  `;
                        ctx2.fillText(sep, curX, m.y);
                        curX += ctx2.measureText(sep).width;
                        
                        ctx2.fillStyle=bgColors[i];
                        ctx2.fillText(`${scanVals[i]} scans`, curX, m.y);
                    }
                    
                    ctx2.restore();
                });
            }
        }],
    });
}

// ── Cross-filter ──
async function applyFilter() {
    renderDiagHighlight();
    
    // Fetch filtered metadata from backend
    if (activeFilter && activeFilter.field === 'diagnosis') {
        showLoading(`Filtering cohort to ${activeFilter.value}…`);
        try {
            const p = new URLSearchParams({
                csv_path: $csvSelect.value,
                cohort: activeFilter.value
            });
            if (selectedScanFolders.length) p.set('scan_folders', selectedScanFolders.join(','));
            const r = await fetch(`/api/metadata?${p}`);
            filteredMeta = await r.json();
        } catch(e) { console.error("Filter failed", e); }
        hideLoading();
    } else {
        filteredMeta = globalMeta;
    }
    
    // Destroy charts that will be re-rendered
    ['age','sex','scanVisits','scansPerSubj','metaVisits','mmse','cdr','apoe','split'].forEach(id => {
        if(activeCharts[id]) { activeCharts[id].destroy(); delete activeCharts[id]; }
    });
    
    renderDemo(filteredMeta);
    renderScans(lastScan, filteredMeta);
    renderClinical(filteredMeta);
    renderTable(filteredMeta, lastScan);
    
    let indicator = $('filterIndicator');
    if (!indicator) {
        indicator = document.createElement('div');
        indicator.id = 'filterIndicator';
        indicator.style.cssText = 'font-size:.75rem;color:var(--amber);margin-left:auto;cursor:pointer;';
        indicator.onclick = () => { activeFilter = null; applyFilter(); };
        document.querySelector('.table-header').appendChild(indicator);
    }
    indicator.textContent = activeFilter ? `Filtered: ${activeFilter.value} — click again to clear` : '';
    indicator.style.display = activeFilter ? '' : 'none';
}

function renderDiagHighlight() {
    const chart = activeCharts['diag'];
    if (!chart || !globalMeta?.diagnosis_distribution) return;
    const labels = chart.data.labels;
    const baseColors = labels.map(l=>diagColor(l));
    const patColors = labels.map((l,i)=>{
        const c = baseColors[i];
        if (!activeFilter) return c+'55';
        return l===activeFilter.value ? c+'55' : c+'22';
    });
    const visitColors = labels.map((l,i)=>{
        const c = baseColors[i];
        if (!activeFilter) return c+'99';
        return l===activeFilter.value ? c+'99' : c+'22';
    });
    const scanColors = labels.map((l,i)=>{
        const c = baseColors[i];
        if (!activeFilter) return c;
        return l===activeFilter.value ? c : c+'44';
    });
    chart.data.datasets.forEach(ds => {
        if (ds.label === 'Patients') ds.backgroundColor = patColors;
        if (ds.label === 'Visits') ds.backgroundColor = visitColors;
        if (ds.label === 'Scans') ds.backgroundColor = scanColors;
    });
    chart.update('none');
}

function filteredRows() {
    const q = $tableSearch.value.toLowerCase().trim();
    let rows = patientData; // patientData is already filtered by filteredMeta
    if (q) rows = rows.filter(r => Object.values(r).some(v=>v!==null&&String(v).toLowerCase().includes(q)));
    return rows;
}

// ── Scans & Visits ──
function renderScans(scan, meta) {
    const sec=$('sectionScans'); const cont=$('scanCharts');
    cont.innerHTML='';
    const has = scan||meta?.visit_distribution;
    sec.style.display=has?'':'none'; if(!has) return;
    if(scan?.visit_distribution) addVBar('scanCharts','scanVisits','Scans per Visit (on disk)',{labels:Object.keys(scan.visit_distribution),counts:Object.values(scan.visit_distribution)});
    if(scan?.scans_per_subject_distribution) addVBar('scanCharts','scansPerSubj','Scans per Subject',scan.scans_per_subject_distribution);
    if(meta?.visit_distribution) addVBar('scanCharts','metaVisits','Visits (from CSV)',meta.visit_distribution);
}

// ── Clinical ──
function renderClinical(meta) {
    const sec=$('sectionClinical'); const cont=$('clinicalCharts');
    cont.innerHTML='';
    const has=meta?.mmse_histogram||meta?.cdr_distribution||meta?.apoe_distribution||meta?.split_distribution;
    sec.style.display=has?'':'none'; if(!has) return;
    if(meta.mmse_histogram) addVBar('clinicalCharts','mmse','MMSE Distribution',meta.mmse_histogram);
    if(meta.cdr_distribution) addHBar('clinicalCharts','cdr','CDR Global',meta.cdr_distribution,BAR_COLORS);
    if(meta.apoe_distribution) addHBar('clinicalCharts','apoe','ApoE Genotype',meta.apoe_distribution,BAR_COLORS);
    if(meta.split_distribution) addHBar('clinicalCharts','split','Train / Val / Test',meta.split_distribution,BAR_COLORS);
}

// ── Chart Builders ──
function addHBar(cid, id, title, data, colors) {
    const labels=data.labels||Object.keys(data);
    const values=data.counts||Object.values(data);
    const total=values.reduce((a,b)=>a+b,0);
    const card=makeCard(cid,id,title);
    const ctx=card.querySelector('canvas').getContext('2d');
    activeCharts[id]=new Chart(ctx,{
        type:'bar',
        data:{labels,datasets:[{data:values,backgroundColor:colors?labels.map((_,i)=>colors[i%colors.length]):labels.map(l=>diagColor(l)),borderWidth:0,borderRadius:4,barThickness:22}]},
        options:{
            indexAxis:'y',responsive:true,maintainAspectRatio:false,
            layout:{padding:{right:50}},
            plugins:{legend:{display:false},tooltip:tooltipStyle()},
            scales:{x:{display:false},y:{grid:{display:false},ticks:{color:'#cbd5e1',font:{size:11,weight:500}}}},
            animation:{duration:500},
        },
        plugins:[{afterDatasetsDraw(ch){
            const ctx2=ch.ctx; const tot=total;
            ch.data.datasets[0].data.forEach((v,i)=>{
                const m=ch.getDatasetMeta(0).data[i]; if(!m) return;
                const pct=tot>0?Math.round(v/tot*100):0;
                ctx2.save(); ctx2.fillStyle='#94a3b8'; ctx2.font='500 10px Inter,sans-serif';
                ctx2.textAlign='left'; ctx2.textBaseline='middle';
                ctx2.fillText(`${v} (${pct}%)`,m.x+6,m.y); ctx2.restore();
            });
        }}],
    });
}

function addVBar(cid, id, title, data) {
    const labels=data.labels||Object.keys(data);
    const values=data.counts||Object.values(data);
    const card=makeCard(cid,id,title);
    const ctx=card.querySelector('canvas').getContext('2d');
    activeCharts[id]=new Chart(ctx,{
        type:'bar',
        data:{labels,datasets:[{data:values,backgroundColor:'rgba(129,140,248,0.55)',borderColor:'rgba(129,140,248,0.8)',borderWidth:1,borderRadius:3,hoverBackgroundColor:'rgba(167,139,250,0.7)'}]},
        options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:tooltipStyle()},scales:{x:{grid:{display:false},ticks:{maxRotation:45,font:{size:10}}},y:{beginAtZero:true,grid:{color:'rgba(255,255,255,0.03)'}}},animation:{duration:500}},
    });
}

function makeCard(cid, id, title) {
    const card=document.createElement('div'); card.className='chart-card';
    card.innerHTML=`<h3>${title}</h3><div class="chart-container"><canvas id="chart-${id}"></canvas></div>`;
    document.getElementById(cid).appendChild(card); return card;
}

function tooltipStyle() {
    return {backgroundColor:'rgba(22,22,20,0.96)',titleColor:'#d4d3d1',bodyColor:'#7a7976',borderColor:'rgba(255,255,255,0.12)',borderWidth:1,padding:10,cornerRadius:6,titleFont:{weight:600}};
}

// ── Patient Table ──
function renderTable(meta, scan) {
    if(!meta?.patient_table?.length){$('tableSection').style.display='none';return;}
    $('tableSection').style.display='';
    patientData=meta.patient_table;
    patientData.forEach(p=>{
        p.has_scan = p.subject_id&&scanSubjects.has(p.subject_id)?true:false;
        if(!Array.isArray(p.visits)) p.visits=[];
    });
    sortCol=null; sortDir='asc';
    const cols=['subject_id','has_scan','diagnosis','sex','age','n_visits','visits','mmse_total','cdr_global','apoe','split'];
    const visibleCols=cols.filter(c=>patientData.some(p=>p[c]!==undefined&&p[c]!==null));
    $tableHead.innerHTML=`<tr>${visibleCols.map(c=>`<th data-col="${c}" onclick="sortTable('${c}')">${fmtCol(c)} <span class="sort-arrow">↕</span></th>`).join('')}</tr>`;
    renderRows(filteredRows());
    $tableCount.textContent=`${patientData.length} patients`;

    // Filter indicator (deprecated in favor of dropdown, but keep simple)
    let fi=$('filterIndicator');
    if(!fi){fi=document.createElement('div');fi.id='filterIndicator';fi.style.cssText='font-size:.75rem;color:var(--amber);margin-left:auto;cursor:pointer;';fi.onclick=()=>{activeFilter=null;if($globalCohortSelect) $globalCohortSelect.value='';applyFilter();};document.querySelector('.table-header').appendChild(fi);}
    fi.style.display='none';
}

function renderRows(data) {
    if(!data.length){$tableBody.innerHTML='<tr><td colspan="100" style="text-align:center;color:var(--text-2);padding:1.5rem">No matches</td></tr>';return;}
    const cols=['subject_id','has_scan','diagnosis','sex','age','n_visits','visits','mmse_total','cdr_global','apoe','split'].filter(c=>patientData.some(p=>p[c]!==undefined&&p[c]!==null));
    $tableBody.innerHTML=data.map(row=>{
        const sid=row.subject_id||'';
        const diag=String(row.diagnosis||'').toLowerCase();
        
        let cls='';
        if (diag === 'mci') cls = 'glow-mci';
        else if (diag === 'converter') cls = 'glow-converter';

        return `<tr class="${cls}" data-sid="${sid}" onclick="openPatient('${sid}')">
            ${cols.map(c=>{
                if(c==='has_scan') {
                    if (row[c]) return `<td style="text-align:center"><span title="Has Scans on Disk" style="display:inline-block;width:10px;height:10px;border-radius:50%;border:2px solid var(--green);box-shadow: 0 0 5px rgba(52,211,153,0.5);"></span></td>`;
                    return `<td></td>`;
                }
                if(c==='diagnosis'){
                    const col=diagColor(diag);
                    return `<td><span style="display:inline-flex;align-items:center;gap:5px"><span style="width:8px;height:8px;border-radius:50%;background:${col};flex-shrink:0"></span>${fmtVal(row[c])}</span></td>`;
                }
                if(c==='visits'){
                    const tags=(row.visits||[]).map(v=>`<span class="visit-tag">${v}</span>`).join('');
                    return `<td><div class="visit-tags">${tags||'—'}</div></td>`;
                }
                if(c==='n_visits') return `<td style="font-weight:${row[c]>1?'600':'400'};color:${row[c]>1?'var(--green)':'var(--text-1)'}">${fmtVal(row[c])}</td>`;
                return `<td>${fmtVal(row[c])}</td>`;
            }).join('')}
        </tr>`;
    }).join('');
}

function fmtCol(c){
    if (c==='has_scan') return 'Disk';
    return c.replace(/_/g,' ').replace(/\b\w/g,l=>l.toUpperCase());
}
function fmtVal(v){
    if(v===null||v===undefined||v==='') return '<span style="color:var(--text-3)">—</span>';
    if(typeof v==='number') return Number.isInteger(v)?v:v.toFixed(1);
    return v;
}

window.sortTable=function(col){
    sortDir=sortCol===col?(sortDir==='asc'?'desc':'asc'):'asc'; sortCol=col;
    document.querySelectorAll('.data-table th').forEach(th=>{
        th.classList.toggle('sorted',th.dataset.col===col);
        th.querySelector('.sort-arrow').textContent=th.dataset.col===col?(sortDir==='asc'?'↑':'↓'):'↕';
    });
    const sorted=[...filteredRows()].sort((a,b)=>{
        let va=a[col],vb=b[col];
        if(va==null) return 1; if(vb==null) return -1;
        if(typeof va==='number'&&typeof vb==='number') return sortDir==='asc'?va-vb:vb-va;
        return sortDir==='asc'?String(va).localeCompare(String(vb)):String(vb).localeCompare(String(va));
    });
    renderRows(sorted);
};

// ── Patient Modal — tabbed view ──
//
// State for the currently-open patient. The modal is laid out as a tab bar
// (Overview / Manifold / Connectivity / QC Viewer / Brain View) and every
// tab reads from this single object so that ``selectedVisit`` is consistent
// across views.
let currentPatient = null;

// Module-level cache for /api/cohort/stats so opening multiple patients
// doesn't refit UMAP each time. Keyed by csv+folders.
const _cohortStatsCache = new Map();
let _activeTrajectoryController = null;

const TAB_DEFS = [
    { id: 'overview',     label: 'Overview' },
    { id: 'manifold',     label: 'Manifold' },
    { id: 'connectivity', label: 'Connectivity' },
    { id: 'qcviewer',     label: 'QC Viewer' },
    { id: 'brainview',    label: 'Brain View' },
];

window.openPatient = async function(sid) {
    if (!sid) return;
    document.querySelectorAll('.data-table tbody tr').forEach(tr =>
        tr.classList.toggle('selected', tr.dataset.sid === sid)
    );
    const patient = patientData.find(p => p.subject_id === sid) || {};
    $('modalTitle').textContent = `sub-${sid}`;
    const diagC = diagColor(String(patient.diagnosis || ''));
    $('modalSubhead').innerHTML = patient.diagnosis
        ? `<span style="color:${diagC};font-weight:600">${patient.diagnosis}</span>`
        : '';

    const fields = ['sex', 'age', 'n_visits', 'apoe'];
    const metaHtml = fields.filter(k => patient[k] != null).map(k =>
        `<div class="meta-item"><div class="meta-label">${fmtCol(k)}</div><div class="meta-value">${fmtVal(patient[k])}</div></div>`
    ).join('');

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
        `<button class="modal-tab${i === 0 ? ' active' : ''}" data-tab="${t.id}">${t.label}</button>`
    ).join('');
    const tabPanelsHtml = TAB_DEFS.map((t, i) =>
        `<div class="tab-panel${i === 0 ? ' active' : ''}" data-tab="${t.id}" id="tab-${t.id}"></div>`
    ).join('');

    $('modalBody').innerHTML = `
        <div class="patient-meta-grid">${metaHtml}</div>
        <div style="display:flex;align-items:flex-end;margin-bottom:1rem">
            <div>
                <div style="font-size:.7rem;color:var(--text-2);text-transform:uppercase;letter-spacing:.06em;margin-bottom:.4rem">Visits in CSV</div>
                <div class="visit-tags">${visitHtml}</div>
            </div>
            ${toggleHtml}
        </div>
        <div class="modal-tabs" id="modalTabBar">${tabBarHtml}</div>
        ${tabPanelsHtml}
    `;

    currentPatient = {
        sid, diagC, isConverter, includeAD: false,
        traj: null, clinical: null, manifold: null,
        cohortStats: null, allVisits: [], mergedVisits: [],
        selectedVisit: (patient.visits && patient.visits.length) ? patient.visits[patient.visits.length - 1] : null,
        activeTab: 'overview',
        tabRendered: { overview: false, manifold: false, connectivity: false, qcviewer: false, brainview: false },
        niiVue: null,
        scansList: [],
    };
    _matrixCache.clear();  // per-visit correlation matrices belong to this patient

    $('modalBackdrop').classList.add('open');

    // Tab switching
    $('modalTabBar').addEventListener('click', e => {
        const btn = e.target.closest('.modal-tab');
        if (!btn) return;
        switchTab(btn.dataset.tab);
    });

    if (isConverter) {
        $('adScanToggle').addEventListener('change', (e) => {
            currentPatient.includeAD = e.target.checked;
            // Re-fetch and re-render every tab from scratch
            currentPatient.tabRendered = { overview: false, manifold: false, connectivity: false, qcviewer: false, brainview: false };
            loadPatientData();
        });
    }

    loadPatientData();
};

function switchTab(tabId) {
    if (!currentPatient) return;
    document.querySelectorAll('#modalTabBar .modal-tab').forEach(b =>
        b.classList.toggle('active', b.dataset.tab === tabId));
    document.querySelectorAll('.tab-panel').forEach(p =>
        p.classList.toggle('active', p.dataset.tab === tabId));
    currentPatient.activeTab = tabId;
    renderActiveTab();
}

function renderActiveTab() {
    if (!currentPatient) return;
    const t = currentPatient.activeTab;
    if (t === 'overview')     renderOverviewTab();
    else if (t === 'manifold')     renderManifoldTab();
    else if (t === 'connectivity') renderConnectivityTab();
    else if (t === 'qcviewer')     renderQCViewerTab();
    else if (t === 'brainview')    renderBrainViewTab();
}

// Compute the folders to query, optionally adding sibling AD/Converter folders
// for converter patients. Mirrors the original include-AD logic.
function _resolveQueryFolders(includeAD) {
    let folders = [...selectedScanFolders];
    if (includeAD && discoveryData && discoveryData.scan_folders) {
        const baseDatasets = new Set(selectedScanFolders.map(f => f.split('/')[0]));
        const sib = discoveryData.scan_folders.filter(f => {
            if (!baseDatasets.has(f.path.split('/')[0])) return false;
            return f.path.split('/').some(seg => {
                const s = seg.toLowerCase();
                return s === 'ad' || s.startsWith('ad_') ||
                       s === 'converter' || s.startsWith('converter_');
            });
        }).map(f => f.path);
        sib.forEach(f => { if (!folders.includes(f)) folders.push(f); });
    }
    return folders;
}

async function _fetchCohortStats(csvPath, folders) {
    const key = JSON.stringify([csvPath, [...folders].sort()]);
    if (_cohortStatsCache.has(key)) return _cohortStatsCache.get(key);
    if (!csvPath || !folders.length) return null;
    try {
        const r = await fetch(
            `/api/cohort/stats?csv_path=${encodeURIComponent(csvPath)}` +
            `&scan_folders=${encodeURIComponent(folders.join(','))}`
        );
        if (!r.ok) return null;
        const json = await r.json();
        _cohortStatsCache.set(key, json);
        return json;
    } catch (e) { console.warn('cohort/stats failed', e); return null; }
}

function _setPatientLoadingText(text) {
    const panel = document.querySelector('.tab-panel.active');
    if (!panel) return;
    panel.innerHTML = `<div class="loading-text" style="padding:2rem;text-align:center">${text}</div>`;
}

async function _fetchTrajectoryStream(sid, folders, token, preferredVisit, signal) {
    if (!folders.length) return { sessions: [] };
    const query = new URLSearchParams({
        scan_folders: folders.join(','),
    });
    if (preferredVisit) query.set('prioritize_visit', preferredVisit);
    const resp = await fetch(`/api/patient/${sid}/trajectory?${query.toString()}`, { signal });
    if (!resp.ok) throw new Error(`trajectory request failed (${resp.status})`);

    let trajectory = null;
    const processLine = (line) => {
        const trimmed = line.trim();
        if (!trimmed) return;
        const msg = JSON.parse(trimmed);
        if (!currentPatient || currentPatient._loadToken !== token) return;
        if (msg.type === 'progress') {
            const visit = msg.visit || 'unknown';
            _setPatientLoadingText(`Computing biomarkers for ${visit} (${msg.current}/${msg.total})…`);
            return;
        }
        if (msg.type === 'complete') {
            trajectory = msg.data || { sessions: [] };
        }
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

// Build the merged sessions table from whatever data we currently have.
// Re-callable: harmless to invoke before manifold data arrives — that just
// means manifold_x / conversion_score columns will be undefined for now.
function _rebuildMergedVisits() {
    if (!currentPatient) return;
    const { traj, clinical, manifold } = currentPatient;
    const visitNum = v => { const m = String(v).match(/M(\d+)/i); return m ? parseInt(m[1]) : 999; };
    const allVisitSet = new Set();
    (traj?.sessions || []).forEach(s => allVisitSet.add(s.visit));
    (clinical?.visits || []).forEach(v => allVisitSet.add(v));
    const allVisits = Array.from(allVisitSet).sort((a, b) => visitNum(a) - visitNum(b));

    const mergedMap = new Map();
    (traj?.sessions || []).forEach(s => mergedMap.set(s.visit, { ...s }));
    if (clinical && clinical.visits) {
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
    currentPatient.allVisits = allVisits;
    currentPatient.mergedVisits = Array.from(mergedMap.values())
        .sort((a, b) => visitNum(a.visit) - visitNum(b.visit));
}

// Compute the Set of AD-folder visits from clinical.diagnosis. Stable across
// fast/slow loads — saved on currentPatient so the slow-path can reuse it.
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

// Apply the "Include AD scans" filter in-place to traj / clinical / manifold.
// Pass null for any input you don't want to filter (e.g. on the slow path
// when only the manifold has just arrived).
function _applyIncludeADFilter(traj, clinical, manifold, includeAD, adVisits) {
    if (!adVisits) adVisits = _computeAdVisits(clinical);
    if (clinical) clinical.adVisits = Array.from(adVisits);
    if (includeAD) return adVisits;

    if (clinical && clinical.visits) {
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

async function loadPatientData() {
    if (!currentPatient) return;
    const { sid, includeAD } = currentPatient;
    const myToken = ++currentPatient._loadToken || (currentPatient._loadToken = 1);
    const csvPath = $csvSelect.value;
    const folders = _resolveQueryFolders(includeAD);
    const preferredVisit = currentPatient.selectedVisit;
    if (_activeTrajectoryController) _activeTrajectoryController.abort();
    const trajectoryController = new AbortController();
    _activeTrajectoryController = trajectoryController;

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
        if (!currentPatient || currentPatient._loadToken !== myToken) return;  // user moved on

        const adVisits = _applyIncludeADFilter(traj, clinical, null, includeAD);

        currentPatient.traj = traj;
        currentPatient.clinical = clinical;
        currentPatient.manifold = null;       // arrives later
        currentPatient.cohortStats = null;    // arrives later
        currentPatient.scansList = scansResp?.scans || [];
        currentPatient._adVisits = adVisits;

        _rebuildMergedVisits();
        currentPatient.selectedVisit = currentPatient.mergedVisits.length
            ? currentPatient.mergedVisits[currentPatient.mergedVisits.length - 1].visit
            : null;
        currentPatient.tabRendered = {
            overview: false, manifold: false, connectivity: false, qcviewer: false, brainview: false
        };
        renderActiveTab();
    } catch (e) {
        if (e?.name === 'AbortError') return;
        console.error('fast patient load failed', e);
        const ap = document.querySelector('.tab-panel.active');
        if (ap) ap.innerHTML = '<p class="no-trajectory">Error loading patient data</p>';
        return;
    } finally {
        if (_activeTrajectoryController === trajectoryController) {
            _activeTrajectoryController = null;
        }
    }

    if (!csvPath || !folders.length) return;
    try {
        const [cohortStats, manResp] = await Promise.all([
            _fetchCohortStats(csvPath, folders),
            fetch(`/api/patient/${sid}/manifold?csv_path=${encodeURIComponent(csvPath)}` +
                  `&scan_folders=${encodeURIComponent(folders.join(','))}`)
                .then(r => r.ok ? r.json() : null).catch(() => null),
        ]);
        if (!currentPatient || currentPatient._loadToken !== myToken) return;

        _applyIncludeADFilter(null, null, manResp, includeAD, currentPatient._adVisits);

        currentPatient.cohortStats = cohortStats;
        currentPatient.manifold = manResp;
        _rebuildMergedVisits();

        // Re-render only the tabs that depend on this data.
        currentPatient.tabRendered.overview = false;
        currentPatient.tabRendered.manifold = false;
        const t = currentPatient.activeTab;
        if (t === 'overview' || t === 'manifold') renderActiveTab();
    } catch (e) {
        console.warn('manifold/cohort load failed', e);
    }
}

// When the user clicks a visit in any view, propagate the change everywhere.
function setSelectedVisit(visit) {
    if (!currentPatient || !visit) return;
    if (currentPatient.selectedVisit === visit) return;
    currentPatient.selectedVisit = visit;
    // Highlight the row in the session table (Overview is always rendered if visible)
    document.querySelectorAll('#tab-overview .session-table tbody tr').forEach(tr => {
        tr.classList.toggle('selected', tr.dataset.visit === visit);
    });
    // Re-render only the views that depend on the selected visit
    if (currentPatient.tabRendered.connectivity) renderConnectivityHeatmap();
    if (currentPatient.tabRendered.qcviewer)     loadQCViewerVolume();
    if (currentPatient.tabRendered.brainview)    renderBrainGraph();
    if (currentPatient.tabRendered.manifold)     redrawManifold();
}

function closeModal(){
    $('modalBackdrop').classList.remove('open');
    document.querySelectorAll('.data-table tbody tr.selected').forEach(tr=>tr.classList.remove('selected'));
    ['trajFC','trajMod','trajCog','trajCSF','convScore'].forEach(k=>{
        if(activeCharts[k]){activeCharts[k].destroy();delete activeCharts[k];}
    });
    // Tear down NiiVue (otherwise WebGL contexts pile up)
    if (currentPatient) {
        currentPatient.niiVue = null;
        currentPatient.brainNv = null;
    }
    currentPatient = null;
}

// Returns a Chart.js plugin that marks the MCI → AD conversion boundary.
// convVisit – label of the first post-conversion visit (e.g. "M36").
// Draws a dashed vertical line midway between the last converter visit and
// the first AD visit, with "◀ conv." on the left and "AD ▶" on the right.
function makeConvPlugin(convVisit) {
    return {
        id: 'convLine',
        afterDraw(chart) {
            const labels = chart.data.labels;
            if (!convVisit || !labels) return;
            const idx = labels.indexOf(convVisit);
            if (idx < 0) return;
            const { ctx, chartArea, scales: { x } } = chart;
            // Midpoint between last converter tick and first AD tick
            const x1 = x.getPixelForValue(labels[idx]);
            const x0 = idx > 0 ? x.getPixelForValue(labels[idx - 1]) : x1;
            const xPx = idx > 0 ? (x0 + x1) / 2 : x1;
            ctx.save();
            // Dashed vertical line
            ctx.beginPath();
            ctx.setLineDash([5, 4]);
            ctx.strokeStyle = 'rgba(248,113,113,0.65)';
            ctx.lineWidth = 1.5;
            ctx.moveTo(xPx, chartArea.top);
            ctx.lineTo(xPx, chartArea.bottom);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.font = 'bold 9px sans-serif';
            // Left label — points toward converter visits
            ctx.textAlign = 'right';
            ctx.fillStyle = 'rgba(156,163,175,0.85)';
            ctx.fillText('◄ conv.', xPx - 5, chartArea.top + 13);
            // Right label — points toward AD visits
            ctx.textAlign = 'left';
            ctx.fillStyle = 'rgba(248,113,113,0.9)';
            ctx.fillText('AD ►', xPx + 5, chartArea.top + 13);
            ctx.restore();
        }
    };
}

// ──────────────────────────────────────────────────────────────────────────────
// Overview tab
// ──────────────────────────────────────────────────────────────────────────────

function _normativeRefCohort(stats) {
    // Prefer mci (= MCI non-converters) — matches the converter baseline stage.
    // Falls back to healthy if mci has no data.
    if (!stats?.biomarker_stats) return null;
    if (stats.biomarker_stats.mci && Object.keys(stats.biomarker_stats.mci).length) return 'mci';
    if (stats.biomarker_stats.healthy && Object.keys(stats.biomarker_stats.healthy).length) return 'healthy';
    return null;
}

function _devCard(label, value, refMean, format = v => v?.toFixed(3) ?? '—') {
    if (value == null || refMean == null || !isFinite(refMean) || refMean === 0) {
        return `<div class="dev-card normal">
            <div class="dev-label">${label}</div>
            <div class="dev-value">${format(value)}</div>
            <div class="dev-sub">no reference</div>
        </div>`;
    }
    const pct = ((value - refMean) / Math.abs(refMean)) * 100;
    const cls = pct < -5 ? 'below' : pct > 5 ? 'above' : 'normal';
    const sign = pct >= 0 ? '+' : '';
    return `<div class="dev-card ${cls}">
        <div class="dev-label">${label}</div>
        <div class="dev-value">${format(value)}</div>
        <div class="dev-sub">${sign}${pct.toFixed(0)}% vs MCI-NC</div>
    </div>`;
}

function renderOverviewTab() {
    if (!currentPatient) return;
    const el = $('tab-overview');
    const { traj, clinical, manifold, cohortStats, allVisits, mergedVisits, diagC } = currentPatient;
    const hasFmri = traj?.sessions?.length > 0;
    const hasClinical = clinical?.visits?.length > 0;

    if (!hasFmri && !hasClinical) {
        el.innerHTML = '<p class="no-trajectory">No longitudinal data found for this subject</p>';
        currentPatient.tabRendered.overview = true;
        return;
    }

    const refCohort = _normativeRefCohort(cohortStats);
    const refStats = refCohort ? cohortStats.biomarker_stats[refCohort] : null;
    const refLabel = refCohort === 'mci' ? 'MCI-NC' : (refCohort || '—');

    // Deviation strip — based on the LAST fMRI visit
    let stripHtml = '';
    if (refStats && hasFmri) {
        const last = traj.sessions[traj.sessions.length - 1];
        const cards = [
            _devCard('Global FC',  last.global_fc,  refStats.global_fc?.mean),
            _devCard('DMN FC',     last.dmn_fc,     refStats.dmn_fc?.mean),
            _devCard('Modularity', last.modularity, refStats.modularity?.mean),
        ];
        const score = manifold?.trajectory?.[manifold.trajectory.length - 1]?.conversion_score;
        if (score != null) {
            const cls = score > 0.5 ? 'below' : score > 0.2 ? 'above' : 'normal';
            cards.push(`<div class="dev-card ${cls}">
                <div class="dev-label">Conversion Score</div>
                <div class="dev-value">${score.toFixed(2)}</div>
                <div class="dev-sub">0=MCI-NC · 1=AD</div>
            </div>`);
        }
        stripHtml = `<div class="deviation-strip">${cards.join('')}</div>`;
    }

    // Conversion-score sparkline row
    let convHtml = '';
    if (manifold?.trajectory?.some(t => t.conversion_score != null)) {
        const vals = manifold.trajectory.map(t =>
            t.conversion_score != null ? t.conversion_score.toFixed(2) : '—');
        const visits = manifold.trajectory.map(t => t.visit);
        const pairs = visits.map((v, i) => `<span><span class="v">${v}</span>: ${vals[i]}</span>`).join(' · ');
        convHtml = `<div class="conv-score-row">
            <div>
                <div class="conv-label">Conversion Score (fMRI-only)</div>
                <div class="conv-vals">${pairs}</div>
            </div>
            <canvas id="chart-convScore"></canvas>
        </div>`;
    }

    let html = stripHtml + convHtml;

    if (hasFmri) {
        html += `
        <div class="trajectory-chart-wrap">
            <h3 style="margin-bottom:.5rem;font-size:.75rem">Global FC &amp; DMN FC ${refLabel !== '—' ? `<span style="color:var(--text-3);font-weight:400">· band = ${refLabel} ±1σ</span>` : ''}</h3>
            <div class="chart-container"><canvas id="chart-trajFC"></canvas></div>
        </div>
        <div class="trajectory-chart-wrap">
            <h3 style="margin-bottom:.5rem;font-size:.75rem">Network Modularity Q ${refLabel !== '—' ? `<span style="color:var(--text-3);font-weight:400">· band = ${refLabel} ±1σ</span>` : ''}</h3>
            <div class="chart-container"><canvas id="chart-trajMod"></canvas></div>
        </div>`;
    }

    if (hasClinical) {
        html += `
        <div class="trajectory-chart-wrap">
            <h3 style="margin-bottom:.5rem;font-size:.75rem">Cognitive Scores</h3>
            <div class="chart-container"><canvas id="chart-trajCog"></canvas></div>
        </div>
        <div class="trajectory-chart-wrap">
            <h3 style="margin-bottom:.5rem;font-size:.75rem">CSF Biomarkers</h3>
            <div class="chart-container"><canvas id="chart-trajCSF"></canvas></div>
        </div>`;
    }

    html += `
        <div class="session-table-wrap">
            <table class="session-table">
                <thead><tr><th>Visit</th><th>Global FC</th><th>Modularity</th><th>Conv. Score</th><th>MMSE</th><th>CDR</th><th>Aβ42</th><th>tTau</th></tr></thead>
                <tbody>${mergedVisits.map(s => `<tr data-visit="${s.visit}"${s.visit === currentPatient.selectedVisit ? ' class="selected"' : ''}>
                    <td>${s.visit}</td>
                    <td>${s.global_fc?.toFixed(4) || '—'}</td>
                    <td>${s.modularity?.toFixed(4) || '—'}</td>
                    <td>${s.conversion_score != null ? s.conversion_score.toFixed(2) : '—'}</td>
                    <td>${s.mmse != null ? s.mmse : '—'}</td>
                    <td>${s.cdr != null ? s.cdr : '—'}</td>
                    <td>${s.abeta42 != null ? s.abeta42 : '—'}</td>
                    <td>${s.tau != null ? s.tau : '—'}</td>
                </tr>`).join('')}
                </tbody>
            </table>
        </div>`;

    el.innerHTML = html;

    // Wire up: clicking a session row updates selectedVisit
    el.querySelectorAll('.session-table tbody tr').forEach(tr => {
        tr.addEventListener('click', () => setSelectedVisit(tr.dataset.visit));
    });

    // ── Chart-axis setup ──────────────────────────────────────────────────────
    const visitNum = v => { const m = String(v).match(/M(\d+)/i); return m ? parseInt(m[1]) : 999; };
    const sharedY = { grid: { color: 'rgba(255,255,255,0.03)' }, grace: '15%', afterFit: a => { a.width = 45; } };
    const sharedOpts = { responsive: true, maintainAspectRatio: false, plugins: { tooltip: tooltipStyle() }, animation: { duration: 600 } };
    const sharedX = { grid: { display: false }, ticks: { font: { size: 11, weight: 500 } }, labels: allVisits };

    // Conversion-point detection (file path → fallback to last-scan boundary)
    let conversionVisit = hasFmri
        ? (traj.sessions.find(s =>
              s.file && s.file.split('/').some(seg => {
                  const sl = seg.toLowerCase();
                  return sl === 'ad' || sl.startsWith('ad_');
              })
          )?.visit ?? null)
        : null;
    const isConverter = hasClinical &&
        (clinical.diagnosis?.some(d => String(d).toLowerCase() === 'converter') ?? false);
    if (!conversionVisit && isConverter && hasFmri && hasClinical) {
        const lastFmriNum = visitNum(traj.sessions[traj.sessions.length - 1]?.visit ?? '');
        if (lastFmriNum < 999) {
            const firstAfter = clinical.visits.find(v => visitNum(v) > lastFmriNum);
            if (firstAfter) conversionVisit = firstAfter;
        }
    }
    const convPlugin = makeConvPlugin(conversionVisit);

    // Build a Chart.js dataset for a flat horizontal normative band on a metric.
    // Uses the trick of two boundary datasets with fill:'+1' between them.
    const makeBand = (metric, color) => {
        if (!refStats || !refStats[metric]) return [];
        const { mean, std } = refStats[metric];
        if (mean == null || std == null) return [];
        const upper = Array(allVisits.length).fill(mean + std);
        const lower = Array(allVisits.length).fill(mean - std);
        const meanLine = Array(allVisits.length).fill(mean);
        return [
            { label: '_band_upper_' + metric, data: upper, borderColor: 'transparent', backgroundColor: color + '22', pointRadius: 0, fill: '+1', order: 99, spanGaps: true },
            { label: '_band_lower_' + metric, data: lower, borderColor: 'transparent', backgroundColor: 'transparent', pointRadius: 0, fill: false, order: 100, spanGaps: true },
            { label: refLabel + ' mean', data: meanLine, borderColor: color + 'aa', borderDash: [4, 3], borderWidth: 1.2, pointRadius: 0, fill: false, order: 98, spanGaps: true },
        ];
    };

    // Hide normative-band entries from the legend / tooltip
    const bandLegendFilter = (item, data) => {
        const ds = data.datasets[item.datasetIndex];
        return ds && !String(ds.label || '').startsWith('_band_');
    };

    if (hasFmri) {
        const sessMap = new Map(traj.sessions.map(s => [s.visit, s]));
        const gFC = allVisits.map(v => sessMap.get(v)?.global_fc ?? null);
        const dFC = allVisits.map(v => sessMap.get(v)?.dmn_fc ?? null);
        const mod = allVisits.map(v => sessMap.get(v)?.modularity ?? null);
        const accentColor = diagC || C.indigo;

        const fcDatasets = [
            ...makeBand('global_fc', '#6daa45'),  // band first → drawn under lines
            { label: 'Global FC', data: gFC, borderColor: C.indigo, backgroundColor: 'rgba(129,140,248,0.1)', borderWidth: 2.5, pointRadius: 5, pointBackgroundColor: C.indigo, pointBorderColor: C.indigo, tension: 0.3, fill: false, spanGaps: true },
            { label: 'DMN FC', data: dFC, borderColor: accentColor, backgroundColor: accentColor + '18', borderWidth: 2.5, pointRadius: 5, pointBackgroundColor: accentColor, pointBorderColor: accentColor, tension: 0.3, fill: false, spanGaps: true },
        ];
        activeCharts['trajFC'] = new Chart($('chart-trajFC').getContext('2d'), {
            type: 'line',
            data: { labels: allVisits, datasets: fcDatasets },
            options: { ...sharedOpts, plugins: { ...sharedOpts.plugins, legend: { position: 'top', labels: { usePointStyle: true, boxWidth: 10, padding: 12, font: { size: 11 }, filter: bandLegendFilter } }, tooltip: { ...tooltipStyle(), filter: c => !String(c.dataset?.label || '').startsWith('_band_') } }, scales: { x: sharedX, y: sharedY } },
            plugins: [convPlugin],
        });

        const modDatasets = [
            ...makeBand('modularity', '#6daa45'),
            { label: 'Modularity Q', data: mod, borderColor: C.amber, backgroundColor: 'rgba(251,191,36,0.08)', borderWidth: 2.5, pointRadius: 5, pointBackgroundColor: C.amber, pointBorderColor: C.amber, tension: 0.3, fill: false, spanGaps: true },
        ];
        activeCharts['trajMod'] = new Chart($('chart-trajMod').getContext('2d'), {
            type: 'line',
            data: { labels: allVisits, datasets: modDatasets },
            options: { ...sharedOpts, plugins: { ...sharedOpts.plugins, legend: { position: 'top', labels: { usePointStyle: true, boxWidth: 10, padding: 12, font: { size: 11 }, filter: bandLegendFilter } }, tooltip: { ...tooltipStyle(), filter: c => !String(c.dataset?.label || '').startsWith('_band_') } }, scales: { x: sharedX, y: sharedY } },
            plugins: [convPlugin],
        });
    }

    if (hasClinical) {
        const clinIdxMap = new Map(clinical.visits.map((v, i) => [v, i]));
        const mapClin = arr => allVisits.map(v => {
            const i = clinIdxMap.get(v);
            return i !== undefined ? (arr[i] ?? null) : null;
        });
        const c_mmse  = mapClin(clinical.cognitive.mmse);
        const c_cdr   = mapClin(clinical.cognitive.cdr);
        const c_pacc5 = mapClin(clinical.cognitive.pacc5);

        activeCharts['trajCog'] = new Chart($('chart-trajCog').getContext('2d'), {
            type: 'line',
            data: { labels: allVisits, datasets: [
                { label: 'MMSE (0-30)', data: c_mmse, borderColor: C.green, backgroundColor: C.green + '18', borderWidth: 2.5, pointRadius: 5, pointBackgroundColor: C.green, pointBorderColor: C.green, tension: 0.3, fill: true, spanGaps: true, yAxisID: 'y' },
                { label: 'CDR', data: c_cdr, borderColor: C.rose, backgroundColor: 'transparent', borderWidth: 2.5, pointRadius: 5, pointBackgroundColor: C.rose, pointBorderColor: C.rose, tension: 0.3, fill: false, spanGaps: true, yAxisID: 'y1' },
                { label: 'PACC5', data: c_pacc5, borderColor: C.cyan, backgroundColor: 'transparent', borderWidth: 2.5, pointRadius: 5, pointBackgroundColor: C.cyan, pointBorderColor: C.cyan, tension: 0.3, fill: false, spanGaps: true, yAxisID: 'y2' },
            ]},
            options: { ...sharedOpts, plugins: { ...sharedOpts.plugins, legend: { position: 'top', labels: { usePointStyle: true, boxWidth: 10, padding: 12, font: { size: 11 } } } },
                scales: {
                    x: sharedX,
                    y:  { ...sharedY, position: 'left', title: { display: true, text: 'MMSE', font: { size: 9 } }, min: 0, max: 30 },
                    y1: { grid: { display: false }, position: 'right', title: { display: true, text: 'CDR', font: { size: 9 } }, min: 0, max: 3 },
                    y2: { grid: { display: false }, position: 'right', title: { display: true, text: 'PACC5', font: { size: 9 } } }
                }
            },
            plugins: [convPlugin],
        });

        const c_abeta = mapClin(clinical.csf.abeta42);
        const c_tau   = mapClin(clinical.csf.tau);
        const c_ptau  = mapClin(clinical.csf.ptau);
        activeCharts['trajCSF'] = new Chart($('chart-trajCSF').getContext('2d'), {
            type: 'line',
            data: { labels: allVisits, datasets: [
                { label: 'Aβ42', data: c_abeta, borderColor: C.indigo, backgroundColor: 'transparent', borderWidth: 2.5, pointRadius: 5, pointBackgroundColor: C.indigo, pointBorderColor: C.indigo, tension: 0.3, fill: false, spanGaps: true, yAxisID: 'y' },
                { label: 't-Tau', data: c_tau, borderColor: C.amber, backgroundColor: 'transparent', borderWidth: 2.5, pointRadius: 5, pointBackgroundColor: C.amber, pointBorderColor: C.amber, tension: 0.3, fill: false, spanGaps: true, yAxisID: 'y1' },
                { label: 'p-Tau', data: c_ptau, borderColor: C.violet, backgroundColor: 'transparent', borderWidth: 2.5, pointRadius: 5, pointBackgroundColor: C.violet, pointBorderColor: C.violet, tension: 0.3, fill: false, spanGaps: true, yAxisID: 'y1' },
            ]},
            options: { ...sharedOpts, plugins: { ...sharedOpts.plugins, legend: { position: 'top', labels: { usePointStyle: true, boxWidth: 10, padding: 12, font: { size: 11 } } } },
                scales: {
                    x: sharedX,
                    y:  { ...sharedY, position: 'left', title: { display: true, text: 'Aβ42 (pg/ml)', font: { size: 9 } }, grace: '15%' },
                    y1: { grid: { display: false }, position: 'right', title: { display: true, text: 'Tau (pg/ml)', font: { size: 9 } }, grace: '15%' }
                }
            },
            plugins: [convPlugin],
        });
    }

    // Conversion-score sparkline
    if (manifold?.trajectory?.some(t => t.conversion_score != null)) {
        const visits = manifold.trajectory.map(t => t.visit);
        const data   = manifold.trajectory.map(t => t.conversion_score);
        activeCharts['convScore'] = new Chart($('chart-convScore').getContext('2d'), {
            type: 'line',
            data: { labels: visits, datasets: [{
                data, borderColor: C.orange, backgroundColor: 'rgba(224,128,64,0.18)',
                borderWidth: 2, pointRadius: 3, pointBackgroundColor: C.orange,
                tension: 0.25, fill: true, spanGaps: true,
            }]},
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: tooltipStyle() }, scales: { x: { display: false }, y: { display: false, suggestedMin: 0, suggestedMax: 1 } } },
        });
    }

    currentPatient.tabRendered.overview = true;
}

// ──────────────────────────────────────────────────────────────────────────────
// Manifold tab — 2D scatter on a Canvas (no extra library)
// ──────────────────────────────────────────────────────────────────────────────

const MANIFOLD_COLORS = {
    healthy:   { fill: 'rgba(79,152,163,0.42)',  stroke: '#4f98a3' },
    scd:       { fill: 'rgba(232,175,52,0.42)',  stroke: '#e8af34' },
    mci:       { fill: 'rgba(109,170,69,0.42)',  stroke: '#6daa45' },
    converter: { fill: 'rgba(224,128,64,0.42)',  stroke: '#e08040' },
    ad:        { fill: 'rgba(209,99,167,0.42)',  stroke: '#d163a7' },
};

function renderManifoldTab() {
    if (!currentPatient) return;
    const el = $('tab-manifold');
    const { cohortStats, manifold } = currentPatient;

    // The cohort/UMAP fit is the heaviest backend operation (~5 min the very
    // first time). Show progress instead of an error while it's still loading.
    if (cohortStats === null) {
        el.innerHTML = `<div class="manifold-wrap">
            <div class="loading-text" style="padding:2rem;text-align:center">
                Fitting baseline UMAP… this only happens once per dataset.<br>
                <span style="font-size:.7rem;color:var(--text-3)">First fit on a full cohort can take a few minutes; subsequent opens are instant.</span>
            </div>
        </div>`;
        return;  // don't mark tabRendered — re-render once data arrives
    }
    if (!cohortStats?.manifold?.points?.length) {
        el.innerHTML = `<div class="manifold-wrap">
            <p class="no-trajectory">Manifold could not be computed — make sure the cohort has enough baseline scans across CN/SCD/MCI/AD groups.</p>
        </div>`;
        currentPatient.tabRendered.manifold = true;
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
    currentPatient.tabRendered.manifold = true;
}

function drawManifold() {
    if (!currentPatient) return;
    const canvas = $('manifoldCanvas');
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width  = Math.floor(rect.width * dpr);
    canvas.height = Math.floor(rect.height * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const W = rect.width, H = rect.height;
    ctx.clearRect(0, 0, W, H);

    const { cohortStats, manifold, selectedVisit } = currentPatient;
    const points = cohortStats.manifold.points || [];
    const centroids = cohortStats.manifold.centroids || {};
    const axis = cohortStats.manifold.conversion_axis || {};

    // Compute world bounds across all points + centroids + patient trajectory
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

    // Background grid (very subtle)
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

    // Cohort points (translucent)
    points.forEach(p => {
        if (p.x == null || p.y == null) return;
        const col = MANIFOLD_COLORS[p.cohort];
        if (!col) return;
        ctx.fillStyle = col.fill;
        ctx.beginPath();
        ctx.arc(sx(p.x), sy(p.y), 4, 0, Math.PI * 2);
        ctx.fill();
    });

    // Conversion axis (dashed line from origin to target)
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

    // Centroid markers
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
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillText(cohort.toUpperCase(), cx, cy + 12);
        ctx.textAlign = 'start';
        ctx.textBaseline = 'alphabetic';
    });

    // Patient trajectory
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
        // Arrow on the final segment
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
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`${i + 1}`, cx, cy);
        ctx.textAlign = 'start';
        ctx.textBaseline = 'alphabetic';
        // Visit label below dot
        ctx.fillStyle = 'rgba(255,255,255,0.75)';
        ctx.font = '10px Inter, sans-serif';
        ctx.fillText(t.visit, cx + 10, cy + 3);
    });

    // Hit-test for visit clicks
    canvas.onclick = e => {
        const r = canvas.getBoundingClientRect();
        const mx = e.clientX - r.left, my = e.clientY - r.top;
        for (const t of validTraj) {
            const cx = sx(t.x), cy = sy(t.y);
            if (Math.hypot(mx - cx, my - cy) <= 9) {
                setSelectedVisit(t.visit);
                return;
            }
        }
    };
}

function redrawManifold() {
    if (currentPatient?.tabRendered.manifold) drawManifold();
}

// ──────────────────────────────────────────────────────────────────────────────
// Connectivity tab — 200×200 heatmap on Canvas
// ──────────────────────────────────────────────────────────────────────────────

let _matrixCache = new Map();  // visit → matrix payload

function _visitPills() {
    if (!currentPatient) return '';
    return currentPatient.allVisits.map(v =>
        `<button class="visit-pill${v === currentPatient.selectedVisit ? ' active' : ''}" data-visit="${v}">${v}</button>`
    ).join('');
}

function renderConnectivityTab() {
    if (!currentPatient) return;
    const el = $('tab-connectivity');
    el.innerHTML = `
        <div class="heatmap-wrap">
            <div class="visit-selector">
                <label>Visit:</label>
                <div id="connVisitPills">${_visitPills()}</div>
                <label style="margin-left:1rem">
                    <input type="checkbox" id="heatmapGroup" style="margin-right:.3rem">
                    Group by Schaefer network
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

    el.querySelectorAll('.visit-pill').forEach(p => {
        p.addEventListener('click', () => setSelectedVisit(p.dataset.visit));
    });
    $('heatmapGroup').addEventListener('change', renderConnectivityHeatmap);
    currentPatient.tabRendered.connectivity = true;
    renderConnectivityHeatmap();
}

async function renderConnectivityHeatmap() {
    if (!currentPatient) return;
    const visit = currentPatient.selectedVisit;
    if (!visit) return;
    document.querySelectorAll('#connVisitPills .visit-pill').forEach(p =>
        p.classList.toggle('active', p.dataset.visit === visit));

    const canvas = $('heatmapCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.fillStyle = '#161614';
    ctx.fillRect(0, 0, W, H);

    let payload = _matrixCache.get(visit);
    if (!payload) {
        ctx.fillStyle = 'rgba(122,121,118,0.7)';
        ctx.font = '12px Inter';
        ctx.textAlign = 'center';
        ctx.fillText('Loading…', W / 2, H / 2);
        try {
            const folders = _resolveQueryFolders(currentPatient.includeAD);
            const r = await fetch(`/api/patient/${currentPatient.sid}/matrix?scan_folders=${encodeURIComponent(folders.join(','))}&visit=${encodeURIComponent(visit)}`);
            if (!r.ok) throw new Error('fetch failed');
            payload = await r.json();
            _matrixCache.set(visit, payload);
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
        // DMN indices first, then the rest, so the DMN block is visible at top-left.
        const dmn = new Set(payload.dmn_indices || []);
        order = [...order.filter(i => dmn.has(i)), ...order.filter(i => !dmn.has(i))];
    }

    // Resize canvas to match cell × cell so each pixel maps cleanly
    const cell = Math.max(1, Math.floor(540 / n));
    canvas.width  = cell * n;
    canvas.height = cell * n;

    const img = ctx.createImageData(canvas.width, canvas.height);
    const cmap = (v) => {  // RdBu_r-ish, clamped to ±0.5
        const t = Math.max(-1, Math.min(1, v / 0.5));
        if (t >= 0) {
            return [
                Math.round(22 + (214 - 22) * t),
                Math.round(22 + (90 - 22) * t),
                Math.round(20 + (59 - 20) * t),
            ];
        } else {
            const a = -t;
            return [
                Math.round(22 + (59 - 22) * a),
                Math.round(22 + (108 - 22) * a),
                Math.round(20 + (214 - 20) * a),
            ];
        }
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
                    img.data[px]     = r;
                    img.data[px + 1] = g;
                    img.data[px + 2] = b;
                    img.data[px + 3] = 255;
                }
            }
        }
    }
    ctx.putImageData(img, 0, 0);

    // Network divider for grouped view
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

    // Hover tooltip
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
        tip.style.top  = (e.clientY - r.top + 12) + 'px';
        tip.style.display = 'block';
    };
    canvas.onmouseleave = () => { tip.style.display = 'none'; };
}

// ──────────────────────────────────────────────────────────────────────────────
// QC Viewer tab — embedded NiiVue
// ──────────────────────────────────────────────────────────────────────────────

function renderQCViewerTab() {
    if (!currentPatient) return;
    const el = $('tab-qcviewer');
    el.innerHTML = `
        <div class="qc-wrap">
            <div class="visit-selector">
                <label>Visit:</label>
                <div id="qcVisitPills">${_visitPills()}</div>
            </div>
            <canvas class="qc-canvas" id="qcCanvas"></canvas>
            <div class="qc-status" id="qcStatus"></div>
        </div>`;

    el.querySelectorAll('.visit-pill').forEach(p => {
        p.addEventListener('click', () => setSelectedVisit(p.dataset.visit));
    });

    if (typeof niivue === 'undefined' || !niivue?.Niivue) {
        $('qcStatus').textContent = 'NiiVue library failed to load (CDN blocked?). Check your network.';
        currentPatient.tabRendered.qcviewer = true;
        return;
    }

    try {
        const nv = new niivue.Niivue({ backColor: [0, 0, 0, 1], show3Dcrosshair: true });
        nv.attachTo('qcCanvas');
        currentPatient.niiVue = nv;
    } catch (e) {
        $('qcStatus').textContent = 'NiiVue init failed: ' + e.message;
    }

    currentPatient.tabRendered.qcviewer = true;
    loadQCViewerVolume();
}

async function loadQCViewerVolume() {
    if (!currentPatient || !currentPatient.niiVue) return;
    const status = $('qcStatus');
    document.querySelectorAll('#qcVisitPills .visit-pill').forEach(p =>
        p.classList.toggle('active', p.dataset.visit === currentPatient.selectedVisit));

    const visit = currentPatient.selectedVisit;
    const hasNifti = currentPatient.scansList.some(s =>
        String(s.visit).toUpperCase() === String(visit).toUpperCase());
    if (!hasNifti) {
        status.textContent = `No .nii.gz on disk for ${visit}.` +
            (currentPatient.scansList.length === 0
                ? ' Selected scan folders contain only parcellated .npz files — switch to a NIfTI folder for QC.'
                : '');
        return;
    }
    status.textContent = `Loading ${visit}…`;
    const folders = _resolveQueryFolders(currentPatient.includeAD);
    const url = `/api/patient/${currentPatient.sid}/scan?scan_folders=${encodeURIComponent(folders.join(','))}&visit=${encodeURIComponent(visit)}&ext=.nii.gz`;
    try {
        await currentPatient.niiVue.loadVolumes([{ url, name: `${currentPatient.sid}_${visit}.nii.gz` }]);
        status.textContent = `Showing ${visit} · scroll/drag to navigate slices`;
    } catch (e) {
        status.textContent = 'Failed to load volume: ' + (e?.message || e);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Brain View tab — glass brain with strongest edges
// ──────────────────────────────────────────────────────────────────────────────

let _atlasCoords = null;  // cached across patients
async function _getAtlasCoords() {
    if (_atlasCoords) return _atlasCoords;
    try {
        const r = await fetch('/api/atlas/schaefer/coords?n_parcels=200');
        if (!r.ok) return null;
        _atlasCoords = await r.json();
        return _atlasCoords;
    } catch { return null; }
}

const NETWORK_COLORS = {
    Default:    '#e08040',
    Cont:       '#6daa45',
    SalVentAttn:'#e8af34',
    DorsAttn:   '#4f98a3',
    Limbic:     '#d163a7',
    SomMot:     '#7a7976',
    Vis:        '#9f7fbf',
};

// Schaefer 7-network → integer index used for NiiVue node coloring.
const NETWORK_INDEX = {
    Vis: 0, SomMot: 1, DorsAttn: 2, SalVentAttn: 3, Limbic: 4, Cont: 5, Default: 6,
};

// Cohort reference matrix cache — fetched once per session, indexed by cohort.
const _cohortRefMatrices = {};
async function _getCohortReferenceMatrix(cohort = 'healthy') {
    if (_cohortRefMatrices[cohort]) return _cohortRefMatrices[cohort];
    const csv = $csvSelect.value;
    if (!csv || !selectedScanFolders.length) return null;
    try {
        const r = await fetch(
            `/api/cohort/reference?cohort=${encodeURIComponent(cohort)}` +
            `&csv_path=${encodeURIComponent(csv)}` +
            `&scan_folders=${encodeURIComponent(selectedScanFolders.join(','))}`
        );
        if (!r.ok) return null;
        const json = await r.json();
        _cohortRefMatrices[cohort] = json.matrix;
        return _cohortRefMatrices[cohort];
    } catch { return null; }
}

async function _ensureMatrix(visit) {
    let payload = _matrixCache.get(visit);
    if (payload) return payload;
    const folders = _resolveQueryFolders(currentPatient.includeAD);
    try {
        const r = await fetch(
            `/api/patient/${currentPatient.sid}/matrix` +
            `?scan_folders=${encodeURIComponent(folders.join(','))}` +
            `&visit=${encodeURIComponent(visit)}`
        );
        if (!r.ok) return null;
        payload = await r.json();
        _matrixCache.set(visit, payload);
        return payload;
    } catch { return null; }
}

function renderBrainViewTab() {
    if (!currentPatient) return;
    const el = $('tab-brainview');
    el.innerHTML = `
        <div class="brain-wrap">
            <div class="visit-selector">
                <label>Visit:</label>
                <div id="brainVisitPills">${_visitPills()}</div>
                <label style="margin-left:1rem">Mode:</label>
                <select id="brainMode" class="config-input" style="max-width:240px;font-size:.78rem">
                    <option value="raw">Raw — strongest edges</option>
                    <option value="vs-cn">Δ vs CN baseline (deviation)</option>
                    <option value="delta-baseline">Δ since first visit (progression)</option>
                </select>
            </div>
            <div class="brain-controls">
                <label>Top edges:</label>
                <input type="range" id="brainTopPct" min="0.5" max="10" value="2" step="0.5">
                <span id="brainTopPctVal" style="font-size:.78rem;color:var(--text-1);min-width:2.5em">2%</span>
                <span style="font-size:.7rem;color:var(--text-3);margin-left:1rem">drag to rotate · scroll to zoom</span>
            </div>
            <div id="brainNetworkFilters" class="network-filters"></div>
            <canvas id="brainNvCanvas" style="width:100%;height:520px;background:#000;border-radius:8px;display:block;cursor:grab"></canvas>
            <div id="brainStatus" style="font-size:.72rem;color:var(--text-2);margin-top:.5rem;line-height:1.5"></div>
        </div>`;

    el.querySelectorAll('.visit-pill').forEach(p => {
        p.addEventListener('click', () => setSelectedVisit(p.dataset.visit));
    });
    $('brainTopPct').addEventListener('input', e => {
        $('brainTopPctVal').textContent = e.target.value + '%';
        renderBrainGraph();
    });
    $('brainMode').addEventListener('change', renderBrainGraph);

    // Initialize a dedicated NiiVue instance for the connectome.
    if (typeof niivue !== 'undefined' && niivue?.Niivue) {
        try {
            const nv = new niivue.Niivue({
                backColor: [0.06, 0.06, 0.055, 1],
                show3Dcrosshair: false,
                isOrientCube: true,
            });
            nv.attachTo('brainNvCanvas');
            currentPatient.brainNv = nv;
        } catch (e) {
            $('brainStatus').textContent = 'NiiVue init failed: ' + e.message;
        }
    } else {
        $('brainStatus').textContent = 'NiiVue library not loaded.';
    }

    currentPatient.tabRendered.brainview = true;
    renderBrainGraph();
}

async function renderBrainGraph() {
    if (!currentPatient) return;
    const visit = currentPatient.selectedVisit;
    document.querySelectorAll('#brainVisitPills .visit-pill').forEach(p =>
        p.classList.toggle('active', p.dataset.visit === visit));

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

    // Network filter checkboxes
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
        Array.from(filtersEl.querySelectorAll('input[type=checkbox]:checked'))
             .map(i => i.dataset.net)
    );

    const m = payload.matrix;
    const n = m.length;
    if (n !== coords.rois.length) {
        if (status) status.innerHTML =
            `Matrix is ${n}×${n} but coords are ${coords.rois.length}×${coords.rois.length}. ` +
            `Brain View requires the Schaefer parcellation to match the .npz size.`;
        return;
    }

    // ── Compute the display matrix based on the current mode ────────────────
    const mode = $('brainMode').value;
    let displayMatrix = m;
    let modeLabel = `raw correlation @ ${visit}`;

    if (mode === 'vs-cn') {
        const cn = await _getCohortReferenceMatrix('healthy');
        if (!cn) {
            status.textContent = 'CN reference unavailable — fitting cohort statistics first…';
            return;
        }
        if (cn.length !== n) {
            status.textContent = `CN reference is ${cn.length}×${cn.length}, current matrix is ${n}×${n}.`;
            return;
        }
        displayMatrix = m.map((row, i) => row.map((v, j) => v - cn[i][j]));
        modeLabel = `Δ vs CN baseline mean @ ${visit}`;
    } else if (mode === 'delta-baseline') {
        const firstVisit = currentPatient.allVisits.find(v =>
            (currentPatient.traj?.sessions || []).some(s => s.visit === v));
        if (!firstVisit) {
            status.textContent = 'No fMRI baseline visit available.';
            return;
        }
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

    // ── Pick top |weight| edges respecting network filters ──────────────────
    const topPct = parseFloat($('brainTopPct').value) / 100;
    const totalPairs = n * (n - 1) / 2;
    const keepCount = Math.max(5, Math.floor(totalPairs * topPct));
    const edges = [];
    for (let i = 0; i < n; i++) {
        if (!enabledNetworks.has(coords.rois[i].network)) continue;
        for (let j = i + 1; j < n; j++) {
            if (!enabledNetworks.has(coords.rois[j].network)) continue;
            const w = displayMatrix[i][j];
            if (w === 0) continue;
            edges.push({ i, j, w });
        }
    }
    edges.sort((a, b) => Math.abs(b.w) - Math.abs(a.w));
    const top = edges.slice(0, keepCount);

    const minKept = top.length > 0 ? Math.abs(top[top.length - 1].w).toFixed(3) : '—';
    const maxKept = top.length > 0 ? Math.abs(top[0].w).toFixed(3) : '—';
    const nPos = top.filter(e => e.w > 0).length;
    const nNeg = top.length - nPos;
    if (status) status.innerHTML =
        `<strong>${modeLabel}</strong><br>` +
        `${top.length} edges shown (top ${(topPct * 100).toFixed(1)}%)` +
        ` · |w| ∈ [${minKept}, ${maxKept}]` +
        ` · <span style="color:#d65a3b">${nPos} positive</span> · <span style="color:#3b6cd6">${nNeg} negative</span>`;

    // ── Build connectome JSON for NiiVue ─────────────────────────────────────
    // Edges format is the full symmetric n×n matrix flattened, but we zero
    // everything outside the top-N so NiiVue draws only the strongest ones.
    const sparseEdges = new Float32Array(n * n);
    top.forEach(e => {
        sparseEdges[e.i * n + e.j] = e.w;
        sparseEdges[e.j * n + e.i] = e.w;
    });
    // Find which nodes touch a kept edge so we can downsize the rest
    const activeNodes = new Set();
    top.forEach(e => { activeNodes.add(e.i); activeNodes.add(e.j); });

    const edgeMaxAbs = Math.max(0.05, Math.abs(top[0]?.w ?? 0.05));
    const connectome = {
        name: `sub-${currentPatient.sid} ${visit} ${mode}`,
        nodeColormap: 'warm',
        nodeColormapNegative: 'winter',
        nodeMinColor: 0,
        nodeMaxColor: 6,           // 7 networks → indices 0..6
        nodeScale: 2.5,
        edgeColormap: 'warm',
        edgeColormapNegative: 'winter',
        edgeMin: edgeMaxAbs * 0.05,
        edgeMax: edgeMaxAbs,
        edgeScale: 1.0,
        nodes: {
            names: coords.rois.map(r => r.label || `ROI${r.index}`),
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

    const nv = currentPatient.brainNv;
    if (!nv) {
        status.innerHTML += '<br>NiiVue instance unavailable — cannot render 3D view.';
        return;
    }
    try {
        if (typeof nv.loadConnectomeFromJSON === 'function') {
            await nv.loadConnectomeFromJSON(connectome);
        } else if (typeof nv.loadConnectome === 'function') {
            await nv.loadConnectome(connectome);
        } else {
            throw new Error('NiiVue version too old — needs loadConnectome(FromJSON)');
        }
    } catch (e) {
        console.warn('connectome load failed', e);
        status.innerHTML += '<br>Connectome load failed: ' + (e?.message || e);
    }
}

// ── Loading ──
function showLoading(t){$loadingText.textContent=t;$loading.classList.add('active');}
function hideLoading(){$loading.classList.remove('active');}
