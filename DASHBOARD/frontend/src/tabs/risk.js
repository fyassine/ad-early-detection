// Patient Risk tab — GELSTM ensemble conversion-risk prediction.
//
// Fetches /api/patient/{sid}/risk and renders:
//   - circular gauge with mean probability
//   - CI band (2.5 – 97.5%)
//   - per-fold dots (one per ensemble fold)
//   - meta: model version, visits used, note (eg. "model not deployed")
// When `available: false` the panel renders a calm "model not yet deployed"
// placeholder so the rest of the patient modal stays usable.
import { $ } from '../utils.js';
import { state } from '../state.js';

export async function renderRiskTab() {
    if (!state.currentPatient) return;
    const host = $('tab-risk');
    if (!host) return;

    host.innerHTML = `
        <div class="risk-wrap">
            <div class="loading-text" style="padding:2rem;text-align:center">Loading GELSTM prediction…</div>
        </div>`;

    const { sid } = state.currentPatient;
    const csv = $('csvSelect').value;
    const folders = state.selectedScanFolders.join(',');
    if (!csv || !folders) {
        host.innerHTML = riskPlaceholder('CSV + scan folders are required to fetch a GELSTM prediction.');
        return;
    }

    let pred = null;
    try {
        const r = await fetch(`/api/patient/${sid}/risk?csv_path=${encodeURIComponent(csv)}&scan_folders=${encodeURIComponent(folders)}`);
        if (!r.ok) throw new Error(`risk request failed (${r.status})`);
        pred = await r.json();
    } catch (e) {
        host.innerHTML = riskPlaceholder(`Failed to fetch GELSTM prediction: ${escapeHtml(e.message)}`);
        return;
    }

    if (!pred || pred.available === false) {
        host.innerHTML = riskPlaceholder(pred?.note || 'GELSTM ensemble not yet deployed.');
        state.currentPatient.tabRendered.risk = true;
        return;
    }

    if (pred.prob == null) {
        host.innerHTML = riskPlaceholder(pred.note || 'No prediction returned for this subject.');
        state.currentPatient.tabRendered.risk = true;
        return;
    }

    const prob = Number(pred.prob);
    const ciLo = pred.ci_lo != null ? Number(pred.ci_lo) : null;
    const ciHi = pred.ci_hi != null ? Number(pred.ci_hi) : null;
    const folds = Array.isArray(pred.fold_probs) ? pred.fold_probs : [];
    const visits = Array.isArray(pred.visits_used) ? pred.visits_used : [];

    const riskBand = prob >= 0.7 ? 'high' : prob >= 0.4 ? 'medium' : 'low';
    const ciHtml = (ciLo != null && ciHi != null)
        ? `${(ciLo * 100).toFixed(1)}% – ${(ciHi * 100).toFixed(1)}%`
        : '–';
    const foldDots = folds.length
        ? folds.map((p, i) => `
            <span class="fold-dot" style="--p:${(p * 100).toFixed(1)}%"
                  title="Fold ${i + 1}: ${(p * 100).toFixed(1)}%">${(p * 100).toFixed(0)}</span>
        `).join('')
        : '<span class="muted">no fold probabilities</span>';

    host.innerHTML = `
        <div class="risk-wrap">
            <div class="risk-gauge-card risk-band-${riskBand}">
                <div class="risk-gauge">
                    <svg viewBox="0 0 120 120" width="220" height="220">
                        <circle cx="60" cy="60" r="50" stroke="rgba(255,255,255,.08)" stroke-width="10" fill="none"/>
                        <circle class="risk-gauge-fill"
                                cx="60" cy="60" r="50"
                                stroke="${gaugeColor(prob)}" stroke-width="10" fill="none"
                                stroke-linecap="round"
                                stroke-dasharray="${(prob * 314.16).toFixed(1)} 314.16"
                                transform="rotate(-90 60 60)"/>
                        <text x="60" y="56" text-anchor="middle"
                              font-size="22" font-weight="700"
                              fill="var(--text-0)">${(prob * 100).toFixed(1)}%</text>
                        <text x="60" y="76" text-anchor="middle"
                              font-size="9" letter-spacing="0.1em"
                              fill="var(--text-2)">P(CONVERTER)</text>
                    </svg>
                </div>
                <div class="risk-summary">
                    <div class="risk-summary-row">
                        <div class="risk-summary-label">Ensemble probability</div>
                        <div class="risk-summary-value risk-summary-prob">${(prob * 100).toFixed(1)}%</div>
                    </div>
                    <div class="risk-summary-row">
                        <div class="risk-summary-label">95% confidence band</div>
                        <div class="risk-summary-value">${ciHtml}</div>
                    </div>
                    <div class="risk-summary-row">
                        <div class="risk-summary-label">Visits used</div>
                        <div class="risk-summary-value">${visits.length}${visits.length ? ` <span class="muted">(${visits.map(escapeHtml).join(', ')})</span>` : ''}</div>
                    </div>
                    <div class="risk-summary-row">
                        <div class="risk-summary-label">Model version</div>
                        <div class="risk-summary-value mono">${escapeHtml(pred.model_version || '–')}</div>
                    </div>
                </div>
            </div>

            <div class="risk-folds-card">
                <h3>Per-fold probabilities (${folds.length} folds)</h3>
                <div class="fold-dots">${foldDots}</div>
                <div class="placeholder-body" style="margin-top:.5rem">
                    Each fold is an independently-trained GELSTM ensemble
                    member. The reported probability is the mean across
                    folds; the 95% CI band is the percentile bootstrap of
                    fold predictions.
                </div>
            </div>

            ${pred.note ? `<div class="risk-note">${escapeHtml(pred.note)}</div>` : ''}
        </div>
    `;
    state.currentPatient.tabRendered.risk = true;
}

function riskPlaceholder(message) {
    return `
        <div class="risk-wrap">
            <div class="placeholder-card">
                <div class="placeholder-title">GELSTM risk unavailable</div>
                <div class="placeholder-body">${escapeHtml(message || '')}</div>
            </div>
        </div>
    `;
}

function gaugeColor(prob) {
    if (prob >= 0.7) return '#d05c5c';
    if (prob >= 0.4) return '#e8af34';
    return '#6daa45';
}

function escapeHtml(s) {
    return String(s ?? '').replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
}
