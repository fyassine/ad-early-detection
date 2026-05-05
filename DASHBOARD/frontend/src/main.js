import { Chart, registerables } from 'chart.js';

import './styles/base.css';
import './styles/layout.css';
import './styles/components.css';
import './styles/modals.css';
import './styles/brain.css';

import { init, applyRecent, deleteRecent } from './config.js';
import { openPatient } from './modal.js';
import { sortTable } from './table.js';

Chart.register(...registerables);
Chart.defaults.color = '#7a7976';
Chart.defaults.borderColor = 'rgba(255,255,255,0.08)';
Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
Chart.defaults.font.size = 11;

// Expose handlers used by inline onclick attributes in dynamic HTML.
window.openPatient = openPatient;
window.sortTable = sortTable;
window.applyRecent = applyRecent;
window.deleteRecent = deleteRecent;

document.addEventListener('DOMContentLoaded', init);
