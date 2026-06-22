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

let _toastTimer = null;
export function showError(msg) {
    let el = $('errorToast');
    if (!el) {
        el = document.createElement('div');
        el.id = 'errorToast';
        el.style.cssText = [
            'position:fixed', 'right:20px', 'bottom:20px', 'z-index:10000',
            'max-width:380px', 'padding:12px 16px',
            'background:rgba(22,22,20,0.96)', 'color:#e2e0dd',
            'border:1px solid rgba(224,128,64,0.5)', 'border-radius:8px',
            'font-size:.85rem', 'line-height:1.4',
            'box-shadow:0 6px 24px rgba(0,0,0,0.4)',
            'opacity:0', 'transition:opacity .2s ease-out',
            'pointer-events:none',
        ].join(';');
        document.body.appendChild(el);
    }
    el.textContent = msg;
    requestAnimationFrame(() => { el.style.opacity = '1'; });
    if (_toastTimer) clearTimeout(_toastTimer);
    _toastTimer = setTimeout(() => { el.style.opacity = '0'; }, 5000);
}
