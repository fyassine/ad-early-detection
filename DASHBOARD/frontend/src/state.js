// Single mutable state object shared across all modules.
export const state = {
    // Discovery / config
    discoveryData: null,
    selectedScanFolders: [],

    // Dashboard-level data
    activeCharts: {},
    patientData: [],
    scanSubjects: new Set(),
    globalMeta: null,
    filteredMeta: null,
    lastScan: null,
    activeFilter: null,   // { field: 'diagnosis', value: 'converter' }
    sortCol: null,
    sortDir: 'asc',

    // Patient modal
    currentPatient: null,
    _activeTrajectoryController: null,
    _cohortStatsCache: new Map(),    // keyed by JSON([ csvPath, sortedFolders ])
    _matrixCache: new Map(),         // keyed by visit code (cleared on each openPatient)
    _atlasCoords: null,              // cached across patients
    _cohortRefMatrices: {},          // keyed by cohort name
    _qcPrefetched: new Set(),        // keyed by `${sid}|${visit}|mean`
};
