/**
 * app.js — fMRI Data Dashboard v3
 * Semantic colors · Cross-filtering · Modal · Recent searches · Synced axes
 */

// ── Diagnosis semantic colors (consistent everywhere) ──
const DIAG_COLORS = {
    healthy:   '#34d399',
    scd:       '#38bdf8',
    mci:       '#a78bfa',
    converter: '#fbbf24',
    ad:        '#fb7185',
    relative:  '#94a3b8',
};
function diagColor(label) {
    return DIAG_COLORS[String(label).toLowerCase()] || '#818cf8';
}

const C = {
    indigo:'#818cf8', violet:'#a78bfa', sky:'#38bdf8',
    green:'#34d399', amber:'#fbbf24', rose:'#fb7185',
    cyan:'#22d3ee', orange:'#fb923c',
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
    Chart.defaults.color = '#64748b';
    Chart.defaults.borderColor = 'rgba(255,255,255,0.04)';
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
    
    // Determine allowed type if folders are selected
    let allowedType = null;
    if (selectedScanFolders.length > 0) {
        const firstSelected = folders.find(f => f.path === selectedScanFolders[0]);
        if (firstSelected) allowedType = firstSelected.file_type;
    }

    folders.forEach(f => {
        if (selectedScanFolders.includes(f.path)) return; // Hide already selected
        const parts = f.path.replace(/\\/g,'/').split('/');
        const shortPath = parts.slice(-3).join(' / ');
        
        if (query && !f.path.toLowerCase().includes(query)) return;
        matches++;

        const isDisabled = allowedType && f.file_type !== allowedType;
        const bc = f.file_type==='nii.gz'?'badge-nii':f.file_type==='npz'?'badge-npz':'badge-mixed';
        
        const el = document.createElement('div');
        el.className = `folder-item ${isDisabled ? 'disabled' : ''}`;
        el.title = isDisabled ? `Requires ${allowedType}` : f.path;
        el.innerHTML = `
            <span class="folder-path">${shortPath}</span>
            <span class="badge ${bc}">${f.file_type}</span>
            <span style="font-size:.7rem;color:var(--text-2);white-space:nowrap">${f.scan_count} · ${f.subject_count} subj</span>`;
        
        if (!isDisabled) {
            el.addEventListener('click', () => {
                selectedScanFolders.push(f.path);
                $folderSearch.value = '';
                $folderDropdown.classList.remove('open');
                renderTokens();
                checkReady();
                updateFormatBadge();
            });
        }
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
        
        const token = document.createElement('div');
        token.className = 'folder-token';
        token.innerHTML = `
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
    if (meta) cards.push({icon:'👥', value:meta.unique_subjects||'—', label:'Subjects (CSV)'});
    if (scan) {
        cards.push({icon:'🧲', value:scan.total_scans, label:'Files on Disk'});
        cards.push({icon:'📁', value:scan.total_subjects, label:'Subjects w/ Scans'});
        if (scan.longitudinal_subjects>0) cards.push({icon:'🔄', value:scan.longitudinal_subjects, label:'Longitudinal'});
        if (scan.format_info?.description) cards.push({icon:'📦', value:scan.format_info.description.split('·')[0].trim(), label:'Format'});
    }
    if (meta?.scan_coverage) {
        const c=meta.scan_coverage, pct=c.metadata_subjects>0?Math.round(c.matched/c.metadata_subjects*100):0;
        cards.push({icon:'🎯', value:`${pct}%`, label:'Scan Coverage'});
    }
    if (meta?.age_stats) cards.push({icon:'📅', value:`${meta.age_stats.mean}±${meta.age_stats.std}`, label:'Age (mean±SD)'});
    $summaryCards.innerHTML = cards.map(c=>`
        <div class="summary-card">
            <div class="card-icon">${c.icon}</div>
            <div class="card-value">${c.value}</div>
            <div class="card-label">${c.label}</div>
        </div>`).join('');
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
    const scanVals = meta.diagnosis_scans ? labels.map(k=>meta.diagnosis_scans[k]||0) : null;
    const bgColors = labels.map(l=>diagColor(l));
    const bgColorsDim = labels.map(l=>diagColor(l)+'55');

    const card = document.createElement('div');
    card.className = 'chart-card';
    card.innerHTML = `<h3>Diagnosis — Patients${scanVals?' &amp; Scans':''} <span style="font-size:.65rem;color:var(--text-3);font-weight:400">(click to filter)</span></h3><div class="chart-container"><canvas id="chart-diag"></canvas></div>`;
    cont.appendChild(card);

    const datasets = [{
        label: 'Patients',
        data: patVals,
        backgroundColor: bgColorsDim,
        borderWidth: 0, borderRadius: 4, barThickness: 24,
        grouped: false, // Bullet chart effect (overlap)
    }];
    if (scanVals) datasets.push({
        label: 'Scans',
        data: scanVals,
        backgroundColor: bgColors,
        borderWidth: 0, borderRadius: 4, barThickness: 8,
        grouped: false, // Bullet chart effect (overlap)
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
                legend: { display: !!scanVals, position: 'top', labels: { usePointStyle:true, boxWidth:10, padding:14 } },
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
                chart.data.datasets[0].data.forEach((v,i) => {
                    const m = chart.getDatasetMeta(0).data[i];
                    if (!m) return;
                    const pct = total>0?Math.round(v/total*100):0;
                    
                    const mScan = scanVals ? chart.getDatasetMeta(1).data[i] : null;
                    const max_x = mScan && mScan.x > m.x ? mScan.x : m.x;
                    
                    ctx2.save();
                    ctx2.textBaseline='middle';
                    ctx2.font='500 11px Inter,sans-serif';
                    
                    let curX = max_x + 8;
                    
                    // Draw Patients text
                    ctx2.fillStyle='#94a3b8'; 
                    const tPat = `${v} pat. (${pct}%)`;
                    ctx2.fillText(tPat, curX, m.y);
                    curX += ctx2.measureText(tPat).width;
                    
                    // Draw Scans text
                    if (scanVals) {
                        ctx2.fillStyle='#64748b';
                        ctx2.fillText(`  ·  `, curX, m.y);
                        curX += ctx2.measureText(`  ·  `).width;
                        
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
    if (!activeFilter) {
        chart.data.datasets[0].backgroundColor = baseColors;
    } else {
        chart.data.datasets[0].backgroundColor = labels.map((l,i)=>
            l===activeFilter.value ? baseColors[i] : baseColors[i]+'44');
    }
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
    return {backgroundColor:'rgba(15,23,42,0.95)',titleColor:'#f8fafc',bodyColor:'#94a3b8',borderColor:'rgba(255,255,255,0.08)',borderWidth:1,padding:10,cornerRadius:6,titleFont:{weight:600}};
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

// ── Patient Modal ──
window.openPatient=async function(sid){
    if(!sid) return;
    document.querySelectorAll('.data-table tbody tr').forEach(tr=>tr.classList.toggle('selected',tr.dataset.sid===sid));
    const patient=patientData.find(p=>p.subject_id===sid)||{};
    $('modalTitle').textContent=`sub-${sid}`;
    const diagC=diagColor(String(patient.diagnosis||''));
    $('modalSubhead').innerHTML=patient.diagnosis?`<span style="color:${diagC};font-weight:600">${patient.diagnosis}</span>`:'';

    const fields=['sex','age','n_visits','apoe'];
    const metaHtml=fields.filter(k=>patient[k]!=null).map(k=>
        `<div class="meta-item"><div class="meta-label">${fmtCol(k)}</div><div class="meta-value">${fmtVal(patient[k])}</div></div>`
    ).join('');

    // Visit tags
    const visitHtml=(patient.visits||[]).map(v=>`<span class="visit-tag">${v}</span>`).join('') || '—';
    const isConverter = String(patient.diagnosis||'').toLowerCase() === 'converter';

    let toggleHtml = '';
    if (isConverter) {
        toggleHtml = `<div style="margin-left:auto;display:flex;align-items:center;gap:.5rem;">
            <label style="font-size:.75rem;color:var(--text-1);cursor:pointer;display:flex;align-items:center;gap:.3rem;">
                <input type="checkbox" id="adScanToggle" style="accent-color:var(--rose)"> Include AD scans
            </label>
        </div>`;
    }

    $('modalBody').innerHTML=`
        <div class="patient-meta-grid">${metaHtml}</div>
        <div style="display:flex;align-items:flex-end;margin-bottom:1rem">
            <div>
                <div style="font-size:.7rem;color:var(--text-2);text-transform:uppercase;letter-spacing:.06em;margin-bottom:.4rem">Visits in CSV</div>
                <div class="visit-tags">${visitHtml}</div>
            </div>
            ${toggleHtml}
        </div>
        <div class="trajectory-section">
            <h3>📈 Longitudinal Trajectories</h3>
            <div id="trajectoryContent"><div class="loading-text" style="padding:1rem;text-align:center">Loading biomarkers…</div></div>
        </div>`;

    $('modalBackdrop').classList.add('open');

    if (isConverter) {
        $('adScanToggle').addEventListener('change', (e) => {
            loadPatientTrajectory(sid, diagC, e.target.checked);
        });
    }

    loadPatientTrajectory(sid, diagC, false);
};

async function loadPatientTrajectory(sid, diagC, includeAD) {
    $('trajectoryContent').innerHTML='<div class="loading-text" style="padding:1rem;text-align:center">Loading biomarkers…</div>';
    try{
        let traj = { sessions: [] };
        
        let foldersToQuery = [...selectedScanFolders];
        if (includeAD && discoveryData && discoveryData.scan_folders) {
            // Find AD and Converter folders from the same base dataset.
            // Converters keep the same patient ID across folders — their post-conversion
            // scans land in the AD folder (e.g. AD_postprocessed_v0) with the same subject ID,
            // so we must include both AD and Converter sibling folders in the query.
            const baseDatasets = new Set(selectedScanFolders.map(f => f.split('/')[0]));
            const adFolders = discoveryData.scan_folders.filter(f => {
                if (!baseDatasets.has(f.path.split('/')[0])) return false;
                // Match any path segment that equals or starts with "ad" or "converter"
                return f.path.split('/').some(seg => {
                    const s = seg.toLowerCase();
                    return s === 'ad' || s.startsWith('ad_') ||
                           s === 'converter' || s.startsWith('converter_');
                });
            }).map(f => f.path);
            adFolders.forEach(f => {
                if(!foldersToQuery.includes(f)) foldersToQuery.push(f);
            });
        }

        if(foldersToQuery.length){
            const r1 = await fetch(`/api/patient/${sid}/trajectory?scan_folders=${encodeURIComponent(foldersToQuery.join(','))}`);
            traj = await r1.json();
        }
        
        let clinical = {};
        const currentCsv = $csvSelect.value;
        if(currentCsv){
            const r2 = await fetch(`/api/patient/${sid}/clinical?csv_path=${encodeURIComponent(currentCsv)}`);
            if(r2.ok) clinical = await r2.json();
        }

        let adVisits = new Set();
        if (clinical && clinical.diagnosis && clinical.visits) {
            clinical.diagnosis.forEach((diag, i) => {
                // Only true AD diagnoses drive the filter — converters are in the MCI folder
                // and must show when the toggle is OFF; the toggle's job is adding the AD folder
                const isAd = String(diag).toLowerCase() === 'ad' || String(diag) === '5';
                if (isAd) adVisits.add(clinical.visits[i]);
            });
        }
        clinical.adVisits = Array.from(adVisits);

        if (!includeAD) {
            if (clinical && clinical.visits) {
                const allowedIndices = clinical.visits.map((v, i) => adVisits.has(v) ? -1 : i).filter(i => i !== -1);
                clinical.visits = allowedIndices.map(i => clinical.visits[i]);
                // Guard required: clinical.diagnosis can be absent if the CSV lacks a diagnosis column
                if (clinical.diagnosis) {
                    clinical.diagnosis = allowedIndices.map(i => clinical.diagnosis[i]);
                }
                if (clinical.cognitive) {
                    clinical.cognitive.mmse = allowedIndices.map(i => clinical.cognitive.mmse[i]);
                    clinical.cognitive.cdr = allowedIndices.map(i => clinical.cognitive.cdr[i]);
                    clinical.cognitive.pacc5 = allowedIndices.map(i => clinical.cognitive.pacc5[i]);
                }
                if (clinical.csf) {
                    clinical.csf.abeta42 = allowedIndices.map(i => clinical.csf.abeta42[i]);
                    clinical.csf.tau = allowedIndices.map(i => clinical.csf.tau[i]);
                    clinical.csf.ptau = allowedIndices.map(i => clinical.csf.ptau[i]);
                }
            }
            if (traj && traj.sessions) {
                traj.sessions = traj.sessions.filter(s => !adVisits.has(s.visit));
            }
        }
        
        renderTrajectory(traj, clinical, diagC);
    }catch(e){
        console.error(e);
        $('trajectoryContent').innerHTML='<p class="no-trajectory">Error loading trajectory</p>';
    }
}

function closeModal(){
    $('modalBackdrop').classList.remove('open');
    document.querySelectorAll('.data-table tbody tr.selected').forEach(tr=>tr.classList.remove('selected'));
    ['trajFC','trajMod','trajCog','trajCSF'].forEach(k=>{if(activeCharts[k]){activeCharts[k].destroy();delete activeCharts[k];}});
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

function renderTrajectory(traj, clinical, diagColor) {
    const el=$('trajectoryContent');
    const hasFmri = traj.sessions?.length > 0;
    const hasClinical = clinical?.visits?.length > 0;
    
    if(!hasFmri && !hasClinical){
        el.innerHTML='<p class="no-trajectory">No longitudinal data found for this subject</p>';
        return;
    }
    
    // Merge visits for the table
    // We'll map by visit string
    const mergedMap = new Map();
    if(hasFmri){
        traj.sessions.forEach(s => mergedMap.set(s.visit, { ...s }));
    }
    if(hasClinical){
        clinical.visits.forEach((v, i) => {
            if(!mergedMap.has(v)) mergedMap.set(v, { visit: v });
            const s = mergedMap.get(v);
            s.mmse = clinical.cognitive?.mmse[i];
            s.cdr = clinical.cognitive?.cdr[i];
            s.pacc5 = clinical.cognitive?.pacc5[i];
            s.abeta42 = clinical.csf?.abeta42[i];
            s.tau = clinical.csf?.tau[i];
            s.ptau = clinical.csf?.ptau[i];
        });
    }
    
    // Sort merged visits chronologically (M0 < M12 < M24 …).
    // parseInt("0") === 0 which is falsy, so we use a regex capture group
    // to safely get the numeric part — M0 → 0, unknown → 999.
    const visitNum = v => { const m = String(v).match(/M(\d+)/i); return m ? parseInt(m[1]) : 999; };
    const mergedVisits = Array.from(mergedMap.values()).sort((a,b) => visitNum(a.visit) - visitNum(b.visit));

    let html = '';
    
    if(hasFmri) {
        html += `
        <div class="trajectory-chart-wrap">
            <h3 style="margin-bottom:.5rem;font-size:.75rem">Global FC &amp; DMN FC</h3>
            <div class="chart-container"><canvas id="chart-trajFC"></canvas></div>
        </div>
        <div class="trajectory-chart-wrap">
            <h3 style="margin-bottom:.5rem;font-size:.75rem">Network Modularity Q</h3>
            <div class="chart-container"><canvas id="chart-trajMod"></canvas></div>
        </div>`;
    }
    
    if(hasClinical) {
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
                <thead><tr><th>Visit</th><th>Global FC</th><th>Modularity</th><th>MMSE</th><th>CDR</th><th>Aβ42</th><th>tTau</th></tr></thead>
                <tbody>${mergedVisits.map(s=>`<tr>
                    <td>${s.visit}</td>
                    <td>${s.global_fc?.toFixed(4)||'—'}</td>
                    <td>${s.modularity?.toFixed(4)||'—'}</td>
                    <td>${s.mmse!=null?s.mmse:'—'}</td>
                    <td>${s.cdr!=null?s.cdr:'—'}</td>
                    <td>${s.abeta42!=null?s.abeta42:'—'}</td>
                    <td>${s.tau!=null?s.tau:'—'}</td>
                    </tr>`).join('')}
                </tbody>
            </table>
        </div>`;
        
    el.innerHTML = html;


    const sharedY={grid:{color:'rgba(255,255,255,0.03)'}, grace: '15%', afterFit: function(axis) { axis.width = 45; }};
    const sharedOpts={responsive:true,maintainAspectRatio:false,plugins:{tooltip:tooltipStyle()},animation:{duration:600}};

    // ── Unified visit axis ────────────────────────────────────────────────────────
    // All four charts share the same x-axis covering every visit that exists in
    // either the fMRI trajectory or the clinical CSV.  fMRI data is null (gap) for
    // clinical-only visits; clinical data is null for fMRI-only visits.
    const allVisitSet = new Set();
    if (hasFmri) traj.sessions.forEach(s => allVisitSet.add(s.visit));
    if (hasClinical) clinical.visits.forEach(v => allVisitSet.add(v));
    const allVisits = Array.from(allVisitSet).sort((a,b) => visitNum(a) - visitNum(b));

    const sharedX = {grid:{display:false}, ticks:{font:{size:11,weight:500}}, labels:allVisits};

    // ── Conversion-point detection ────────────────────────────────────────────────
    // Priority 1: file-path — first session whose file lives in the AD folder.
    //   The converter keeps the same subject ID in both MCI and AD folders; AD-folder
    //   sessions only appear when "Include AD Scans" adds those folders to the query.
    let conversionVisit = hasFmri
        ? (traj.sessions.find(s =>
              s.file && s.file.split('/').some(seg => {
                  const sl = seg.toLowerCase();
                  return sl === 'ad' || sl.startsWith('ad_');
              })
          )?.visit ?? null)
        : null;

    // Priority 2: last-scan / first-clinical-only boundary (converter, mixed folder).
    //   When all scans live in the same mixed folder, file-path detection finds nothing.
    //   For converters the pattern is: fMRI scans stop at the last MCI visit, then
    //   follow-up continues as clinical-only rows. The first clinical visit that falls
    //   *after* the last fMRI visit is therefore the first post-conversion timepoint.
    //   e.g. last fMRI = M48, first clinical-only = M60 → line between M48 and M60.
    const isConverter = hasClinical &&
        (clinical.diagnosis?.some(d => String(d).toLowerCase() === 'converter') ?? false);
    if (!conversionVisit && isConverter && hasFmri && hasClinical) {
        const lastFmriNum = visitNum(traj.sessions[traj.sessions.length - 1]?.visit ?? '');
        if (lastFmriNum < 999) {
            const firstAfter = clinical.visits.find(v => visitNum(v) > lastFmriNum);
            if (firstAfter) conversionVisit = firstAfter;
        }
    }
    // ─────────────────────────────────────────────────────────────────────────────

    const convPlugin = makeConvPlugin(conversionVisit);

    if(hasFmri){
        // Map fMRI data onto the unified axis — null for visits without a scan
        const sessMap = new Map(traj.sessions.map(s => [s.visit, s]));
        const gFC  = allVisits.map(v => sessMap.get(v)?.global_fc   ?? null);
        const dFC  = allVisits.map(v => sessMap.get(v)?.dmn_fc      ?? null);
        const mod  = allVisits.map(v => sessMap.get(v)?.modularity   ?? null);
        const accentColor = diagColor||C.indigo;

        activeCharts['trajFC']=new Chart($('chart-trajFC').getContext('2d'),{
            type:'line',
            data:{labels:allVisits,datasets:[
                {label:'Global FC',data:gFC,borderColor:C.indigo,backgroundColor:'rgba(129,140,248,0.1)',borderWidth:2.5,pointRadius:5,pointBackgroundColor:C.indigo,pointBorderColor:C.indigo,tension:0.3,fill:true,spanGaps:true},
                {label:'DMN FC',data:dFC,borderColor:accentColor,backgroundColor:accentColor+'18',borderWidth:2.5,pointRadius:5,pointBackgroundColor:accentColor,pointBorderColor:accentColor,tension:0.3,fill:true,spanGaps:true},
            ]},
            options:{...sharedOpts,plugins:{...sharedOpts.plugins,legend:{position:'top',labels:{usePointStyle:true,boxWidth:10,padding:12,font:{size:11}}}},scales:{x:sharedX,y:sharedY}},
            plugins:[convPlugin],
        });

        activeCharts['trajMod']=new Chart($('chart-trajMod').getContext('2d'),{
            type:'line',
            data:{labels:allVisits,datasets:[
                {label:'Modularity Q',data:mod,borderColor:C.amber,backgroundColor:'rgba(251,191,36,0.08)',borderWidth:2.5,pointRadius:5,pointBackgroundColor:C.amber,pointBorderColor:C.amber,tension:0.3,fill:true,spanGaps:true},
            ]},
            options:{...sharedOpts,plugins:{...sharedOpts.plugins,legend:{position:'top',labels:{usePointStyle:true,boxWidth:10,padding:12,font:{size:11}}}},scales:{x:sharedX,y:sharedY}},
            plugins:[convPlugin],
        });
    }

    if(hasClinical){
        // Map clinical data onto the unified axis — null for visits without clinical data
        const clinIdxMap = new Map(clinical.visits.map((v,i) => [v,i]));
        const mapClin = arr => allVisits.map(v => {
            const i = clinIdxMap.get(v);
            return i !== undefined ? (arr[i] ?? null) : null;
        });
        const c_mmse  = mapClin(clinical.cognitive.mmse);
        const c_cdr   = mapClin(clinical.cognitive.cdr);
        const c_pacc5 = mapClin(clinical.cognitive.pacc5);

        activeCharts['trajCog']=new Chart($('chart-trajCog').getContext('2d'),{
            type:'line',
            data:{labels:allVisits,datasets:[
                {label:'MMSE (0-30)',data:c_mmse,borderColor:C.green,backgroundColor:C.green+'18',borderWidth:2.5,pointRadius:5,pointBackgroundColor:C.green,pointBorderColor:C.green,tension:0.3,fill:true,spanGaps:true,yAxisID:'y'},
                {label:'CDR',data:c_cdr,borderColor:C.rose,backgroundColor:'transparent',borderWidth:2.5,pointRadius:5,pointBackgroundColor:C.rose,pointBorderColor:C.rose,tension:0.3,fill:false,spanGaps:true,yAxisID:'y1'},
                {label:'PACC5',data:c_pacc5,borderColor:C.cyan,backgroundColor:'transparent',borderWidth:2.5,pointRadius:5,pointBackgroundColor:C.cyan,pointBorderColor:C.cyan,tension:0.3,fill:false,spanGaps:true,yAxisID:'y2'},
            ]},
            options:{...sharedOpts,plugins:{...sharedOpts.plugins,legend:{position:'top',labels:{usePointStyle:true,boxWidth:10,padding:12,font:{size:11}}}},
                scales:{
                    x:sharedX,
                    y:{...sharedY, position:'left', title:{display:true, text:'MMSE', font:{size:9}}, min:0, max:30},
                    y1:{grid:{display:false}, position:'right', title:{display:true, text:'CDR', font:{size:9}}, min:0, max:3},
                    y2:{grid:{display:false}, position:'right', title:{display:true, text:'PACC5', font:{size:9}}}
                }
            },
            plugins:[convPlugin],
        });

        const c_abeta = mapClin(clinical.csf.abeta42);
        const c_tau   = mapClin(clinical.csf.tau);
        const c_ptau  = mapClin(clinical.csf.ptau);
        activeCharts['trajCSF']=new Chart($('chart-trajCSF').getContext('2d'),{
            type:'line',
            data:{labels:allVisits,datasets:[
                {label:'Aβ42',data:c_abeta,borderColor:C.indigo,backgroundColor:'transparent',borderWidth:2.5,pointRadius:5,pointBackgroundColor:C.indigo,pointBorderColor:C.indigo,tension:0.3,fill:false,spanGaps:true,yAxisID:'y'},
                {label:'t-Tau',data:c_tau,borderColor:C.amber,backgroundColor:'transparent',borderWidth:2.5,pointRadius:5,pointBackgroundColor:C.amber,pointBorderColor:C.amber,tension:0.3,fill:false,spanGaps:true,yAxisID:'y1'},
                {label:'p-Tau',data:c_ptau,borderColor:C.violet,backgroundColor:'transparent',borderWidth:2.5,pointRadius:5,pointBackgroundColor:C.violet,pointBorderColor:C.violet,tension:0.3,fill:false,spanGaps:true,yAxisID:'y1'},
            ]},
            options:{...sharedOpts,plugins:{...sharedOpts.plugins,legend:{position:'top',labels:{usePointStyle:true,boxWidth:10,padding:12,font:{size:11}}}},
                scales:{
                    x:sharedX,
                    y:{...sharedY, position:'left', title:{display:true, text:'Aβ42 (pg/ml)', font:{size:9}}, grace: '15%'},
                    y1:{grid:{display:false}, position:'right', title:{display:true, text:'Tau (pg/ml)', font:{size:9}}, grace: '15%'}
                }
            },
            plugins:[convPlugin],
        });
    }
}

// ── Loading ──
function showLoading(t){$loadingText.textContent=t;$loading.classList.add('active');}
function hideLoading(){$loading.classList.remove('active');}
