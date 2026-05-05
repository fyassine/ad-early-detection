// DOM helper
export const $ = id => document.getElementById(id);

// Diagnosis semantic colors
export const DIAG_COLORS = {
    healthy:   '#4f98a3',
    scd:       '#6daa45',
    mci:       '#e8af34',
    converter: '#e08040',
    ad:        '#d163a7',
    relative:  '#7a7976',
};

export function diagColor(label) {
    return DIAG_COLORS[String(label).toLowerCase()] || '#4f98a3';
}

export const C = {
    indigo: '#4f98a3', violet: '#6daa45', sky: '#e8af34',
    green: '#6daa45', amber: '#e8af34', rose: '#d163a7',
    cyan: '#4f98a3', orange: '#e08040',
};
export const BAR_COLORS = [C.indigo, C.violet, C.sky, C.green, C.amber, C.rose, C.cyan, C.orange];

export function fmtCol(c) {
    if (c === 'has_scan') return 'Disk';
    return c.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

export function fmtVal(v) {
    if (v === null || v === undefined || v === '') return '<span style="color:var(--text-3)">—</span>';
    if (typeof v === 'number') return Number.isInteger(v) ? v : v.toFixed(1);
    return v;
}

export function tooltipStyle() {
    return {
        backgroundColor: 'rgba(22,22,20,0.96)',
        titleColor: '#d4d3d1', bodyColor: '#7a7976',
        borderColor: 'rgba(255,255,255,0.12)', borderWidth: 1,
        padding: 10, cornerRadius: 6, titleFont: { weight: 600 },
    };
}

export function showLoading(text) {
    $('loadingText').textContent = text;
    $('loadingOverlay').classList.add('active');
}

export function hideLoading() {
    $('loadingOverlay').classList.remove('active');
}
