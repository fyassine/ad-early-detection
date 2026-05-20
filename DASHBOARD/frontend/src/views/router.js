// Top-level view router. Owns the three-tier navigation
// (Population / Cohort / Patient) and delegates rendering to the
// matching view module.
import { state } from '../state.js';
import { $ } from '../utils.js';
import { renderPopulation, resetPopulation } from './population.js';
import { mountCohort } from './cohort.js';
import { renderPatientView, resetPatientView } from './patient.js';
import { restoreModalToBackdrop } from '../modal.js';

const VALID_VIEWS = new Set(['population', 'cohort', 'patient']);
const DEFAULT_VIEW = 'cohort';

export const router = {
    activeView: DEFAULT_VIEW,
};

export function initRouter() {
    document.querySelectorAll('.top-tab').forEach(btn => {
        btn.addEventListener('click', () => setView(btn.dataset.view, { pushHash: true }));
    });

    const goCohort = $('btnGoCohort');
    if (goCohort) goCohort.addEventListener('click', () => setView('cohort', { pushHash: true }));

    window.addEventListener('hashchange', () => setView(readHash() || DEFAULT_VIEW, { pushHash: false }));

    const initial = readHash() || DEFAULT_VIEW;
    setView(initial, { pushHash: false });
}

export function setView(view, opts = {}) {
    if (!VALID_VIEWS.has(view)) view = DEFAULT_VIEW;
    // If we're leaving the Patient tab and the modal was embedded there, put it back.
    if (router.activeView === 'patient' && view !== 'patient') {
        restoreModalToBackdrop();
    }
    router.activeView = view;

    document.querySelectorAll('.top-tab').forEach(btn => {
        const active = btn.dataset.view === view;
        btn.classList.toggle('active', active);
        btn.setAttribute('aria-selected', active ? 'true' : 'false');
    });

    document.querySelectorAll('.view-panel').forEach(panel => {
        const id = panel.id.replace(/^view/, '').toLowerCase();
        const active = id === view;
        panel.classList.toggle('active', active);
        panel.style.display = active ? '' : 'none';
    });

    if (opts.pushHash !== false) {
        const params = new URLSearchParams(window.location.hash.replace(/^#/, ''));
        params.set('view', view);
        history.replaceState(null, '', '#' + params.toString());
    }

    if (view === 'population') renderPopulation();
    if (view === 'cohort') mountCohort();
    if (view === 'patient') renderPatientView(readSidFromHash());
}

function readHash() {
    const params = new URLSearchParams(window.location.hash.replace(/^#/, ''));
    return params.get('view');
}

export function readSidFromHash() {
    const params = new URLSearchParams(window.location.hash.replace(/^#/, ''));
    return params.get('sid');
}

export function openPatientView(sid) {
    const params = new URLSearchParams(window.location.hash.replace(/^#/, ''));
    params.set('view', 'patient');
    if (sid) params.set('sid', sid);
    history.replaceState(null, '', '#' + params.toString());
    setView('patient', { pushHash: false });
}

// Re-render the currently active view (called after analyze()).
export function refreshActiveView() {
    if (router.activeView === 'population') renderPopulation();
    if (router.activeView === 'cohort') mountCohort();
    if (router.activeView === 'patient') renderPatientView(readSidFromHash());
}

// Used by config.js when the user clears the workspace.
export function resetAllViews() {
    resetPopulation();
    resetPatientView();
}
