// Patient view — full-screen single-subject dashboard. Phase 0 ships
// only the empty-state and a sid-resolver hook. Subsequent phases
// re-use the existing modal tab modules (overview, staging, manifold,
// connectivity, brainview, qcviewer) plus the new ones (risk, networks,
// dfc, topology) mounted inline instead of inside the modal.
import { $ } from '../utils.js';
import { state } from '../state.js';
import { openPatient } from '../modal.js';

export function renderPatientView(sid) {
    const empty = $('patientEmpty');
    const content = $('patientContent');
    if (!empty || !content) return;

    if (!sid) {
        empty.style.display = '';
        content.style.display = 'none';
        content.innerHTML = '';
        return;
    }

    // Phase 0: defer to the existing modal as the patient renderer. The
    // modal already orchestrates all six tabs and the visit selector.
    // Phase 4 will replace this with an inline full-screen mount.
    empty.style.display = 'none';
    content.style.display = '';
    content.innerHTML = `
        <div class="placeholder-card">
            <div class="placeholder-title">Patient ${escapeHtml(sid)}</div>
            <div class="placeholder-body">
                Opening detailed view in a modal. Phase 4 will inline the
                six existing tabs (Overview, Staging, Manifold, Connectivity,
                Brain View, QC Viewer) plus the new GELSTM Risk, Per-network,
                Dynamic FC and Graph Metrics tabs directly into this panel.
            </div>
        </div>
    `;

    if (state.patientData && state.patientData.length) {
        const row = state.patientData.find(r => String(r.subject_id) === String(sid));
        if (row) openPatient(row.subject_id);
    }
}

export function resetPatientView() {
    const content = $('patientContent');
    const empty = $('patientEmpty');
    if (content) content.innerHTML = '';
    if (content) content.style.display = 'none';
    if (empty) empty.style.display = '';
}

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
}
