// Patient view — full-screen single-subject view.
// When the user clicks the 👤 button in the modal, the entire modal node is
// physically moved into #patientContent (see modal.js:embedModalInPatientView).
// This view therefore renders nothing of its own when a modal is already
// embedded — it just shows a placeholder otherwise.
import { $ } from '../utils.js';

export function renderPatientView(sid) {
    const empty   = $('patientEmpty');
    const content = $('patientContent');
    if (!empty || !content) return;

    if (!sid) {
        empty.style.display   = '';
        content.style.display = 'none';
        content.innerHTML     = '';
        return;
    }

    // If the modal node is already embedded here (via 👤), do nothing — it renders itself.
    if (content.querySelector('#patientModal')) {
        empty.style.display   = 'none';
        content.style.display = '';
        return;
    }

    // Direct navigation (e.g. ?sid=… in URL with no modal open yet): show a hint.
    empty.style.display   = 'none';
    content.style.display = '';
    content.innerHTML = `
        <div class="placeholder-card" style="margin:2rem auto;max-width:520px">
            <div class="placeholder-title">👤 ${escapeHtml(sid)}</div>
            <div class="placeholder-body">Open the Cohort view, click this patient in the directory, then use the 👤 button in the modal to bring the full view here.</div>
        </div>`;
}

export function resetPatientView() {
    const content = $('patientContent');
    const empty   = $('patientEmpty');
    if (content) { content.innerHTML = ''; content.style.display = 'none'; }
    if (empty)   empty.style.display = '';
}

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c =>
        ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
