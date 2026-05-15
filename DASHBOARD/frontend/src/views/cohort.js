// Cohort view — wraps the existing dashboard.js so the Cohort tab is
// functionally identical to the pre-tiered dashboard. Phase 3 adds new
// sections (network panel, dynamic FC, graph topology, risk distribution)
// on top of this base.
import { state } from '../state.js';
import { render } from '../dashboard.js';

let _mounted = false;

export function mountCohort() {
    if (!_mounted) _mounted = true;
    // If data is already loaded re-render so charts honour the active view.
    if (state.globalMeta || state.lastScan) {
        render(state.lastScan, state.globalMeta, state.filteredMeta || state.globalMeta);
    }
}
