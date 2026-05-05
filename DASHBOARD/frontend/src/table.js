import { $, diagColor, fmtCol, fmtVal } from './utils.js';
import { state } from './state.js';
import { filteredRows, applyFilter } from './dashboard.js';
import { openPatient } from './modal.js';

const TABLE_COLS = ['subject_id', 'has_scan', 'diagnosis', 'sex', 'age', 'n_visits', 'visits', 'mmse_total', 'cdr_global', 'apoe', 'split'];

export function renderTable(meta, scan) {
    if (!meta?.patient_table?.length) { $('tableSection').style.display = 'none'; return; }
    $('tableSection').style.display = '';
    state.patientData = meta.patient_table;
    state.patientData.forEach(p => {
        p.has_scan = !!(p.subject_id && state.scanSubjects.has(p.subject_id));
        if (!Array.isArray(p.visits)) p.visits = [];
    });
    state.sortCol = null; state.sortDir = 'asc';
    const visibleCols = TABLE_COLS.filter(c => state.patientData.some(p => p[c] !== undefined && p[c] !== null));
    $('tableHead').innerHTML = `<tr>${visibleCols.map(c => `<th data-col="${c}" onclick="sortTable('${c}')">${fmtCol(c)} <span class="sort-arrow">↕</span></th>`).join('')}</tr>`;
    renderRows(filteredRows());
    $('tableCount').textContent = `${state.patientData.length} patients`;

    let fi = $('filterIndicator');
    if (!fi) {
        fi = document.createElement('div');
        fi.id = 'filterIndicator';
        fi.style.cssText = 'font-size:.75rem;color:var(--amber);margin-left:auto;cursor:pointer;';
        fi.onclick = () => {
            state.activeFilter = null;
            const sel = $('globalCohortSelect');
            if (sel) sel.value = '';
            applyFilter();
        };
        document.querySelector('.table-header').appendChild(fi);
    }
    fi.style.display = 'none';
}

export function renderRows(data) {
    const $tableBody = $('tableBody');
    if (!data.length) {
        $tableBody.innerHTML = '<tr><td colspan="100" style="text-align:center;color:var(--text-2);padding:1.5rem">No matches</td></tr>';
        return;
    }
    const cols = TABLE_COLS.filter(c => state.patientData.some(p => p[c] !== undefined && p[c] !== null));
    $tableBody.innerHTML = data.map(row => {
        const sid = row.subject_id || '';
        const diag = String(row.diagnosis || '').toLowerCase();
        let cls = '';
        if (diag === 'mci') cls = 'glow-mci';
        else if (diag === 'converter') cls = 'glow-converter';

        return `<tr class="${cls}" data-sid="${sid}" onclick="openPatient('${sid}')">
            ${cols.map(c => {
                if (c === 'has_scan') {
                    if (row[c]) return `<td style="text-align:center"><span title="Has Scans on Disk" style="display:inline-block;width:10px;height:10px;border-radius:50%;border:2px solid var(--green);box-shadow: 0 0 5px rgba(52,211,153,0.5);"></span></td>`;
                    return `<td></td>`;
                }
                if (c === 'diagnosis') {
                    const col = diagColor(diag);
                    return `<td><span style="display:inline-flex;align-items:center;gap:5px"><span style="width:8px;height:8px;border-radius:50%;background:${col};flex-shrink:0"></span>${fmtVal(row[c])}</span></td>`;
                }
                if (c === 'visits') {
                    const tags = (row.visits || []).map(v => `<span class="visit-tag">${v}</span>`).join('');
                    return `<td><div class="visit-tags">${tags || '—'}</div></td>`;
                }
                if (c === 'n_visits') return `<td style="font-weight:${row[c] > 1 ? '600' : '400'};color:${row[c] > 1 ? 'var(--green)' : 'var(--text-1)'}">${fmtVal(row[c])}</td>`;
                return `<td>${fmtVal(row[c])}</td>`;
            }).join('')}
        </tr>`;
    }).join('');
}

export function sortTable(col) {
    state.sortDir = state.sortCol === col ? (state.sortDir === 'asc' ? 'desc' : 'asc') : 'asc';
    state.sortCol = col;
    document.querySelectorAll('.data-table th').forEach(th => {
        th.classList.toggle('sorted', th.dataset.col === col);
        th.querySelector('.sort-arrow').textContent = th.dataset.col === col ? (state.sortDir === 'asc' ? '↑' : '↓') : '↕';
    });
    const sorted = [...filteredRows()].sort((a, b) => {
        let va = a[col], vb = b[col];
        if (va == null) return 1;
        if (vb == null) return -1;
        if (typeof va === 'number' && typeof vb === 'number') return state.sortDir === 'asc' ? va - vb : vb - va;
        return state.sortDir === 'asc' ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
    });
    renderRows(sorted);
}
