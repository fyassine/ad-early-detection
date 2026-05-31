import { Chart, registerables } from 'chart.js';

import './styles/base.css';
import './styles/layout.css';
import './styles/components.css';
import './styles/modals.css';
import './styles/brain.css';
import './styles/staging.css';
import './styles/theme-light.css';

import { init, applyRecent, deleteRecent } from './config.js';
import { openPatient } from './modal.js';
import { sortTable } from './table.js';
import { initRouter, openPatientView } from './views/router.js';
import { initTheme, toggleTheme } from './theme.js';

Chart.register(...registerables);
Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
Chart.defaults.font.size = 11;

// Expose handlers used by inline onclick attributes in dynamic HTML.
window.openPatient = openPatient;
window.openPatientView = openPatientView;
window.sortTable = sortTable;
window.applyRecent = applyRecent;
window.deleteRecent = deleteRecent;
window.toggleTheme = toggleTheme;

document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    init();
    initRouter();
});
