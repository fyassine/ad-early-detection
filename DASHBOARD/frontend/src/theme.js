import { Chart } from 'chart.js';

const KEY = 'nt-theme';

export function initTheme() {
    const saved = localStorage.getItem(KEY) ?? 'dark';
    _apply(saved);
}

export function toggleTheme() {
    const current = document.documentElement.dataset.theme ?? 'dark';
    const next = current === 'light' ? 'dark' : 'light';
    localStorage.setItem(KEY, next);
    _apply(next);
}

function _apply(theme) {
    document.documentElement.dataset.theme = theme;
    const btn = document.getElementById('themeToggle');
    if (btn) btn.setAttribute('aria-label', theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode');
    if (theme === 'light') {
        Chart.defaults.color = '#6b6963';
        Chart.defaults.borderColor = 'rgba(0,0,0,0.09)';
    } else {
        Chart.defaults.color = '#7a7976';
        Chart.defaults.borderColor = 'rgba(255,255,255,0.08)';
    }
}
