# METHODS.md Phrase Migration Table

Applied to every sentence drawn from `DOCS/flipped/METHODS.md` or
`DOCS/random/temporal-first-ablation.md` before it enters a chapter.
`METHODS.md` is left untouched as the research record (OUTLINE §6).

---

## Group 1 — Terminology replacements

| Source phrase | Thesis replacement | Notes |
|---|---|---|
| pre-registered | pre-specified | 31 occurrences in METHODS.md; time-stamped evidence is commit `d5e8353` (2026-08-23) |
| pre-registration | defined prior to running the experiment | |
| registered | pre-specified | when describing design decisions fixed before any run |

---

## Group 2 — Banned tone (must not appear in any chapter)

| Banned phrase | Why banned | If needed, write instead |
|---|---|---|
| "a crossover, not a defeat" | contrastive/narrative | describe the observed pattern numerically |
| "S3 is void, not rejected" | outcome phrasing belongs in Ch.6 | in Ch.4: state the structural condition only |
| "S5 is kept, not rejected" | outcome phrasing | in Ch.4: "S5 is pre-specified as kept regardless of AUC" |
| "the checkpoint learned something real" | editorialising | describe the MSE comparison in Ch.6 |
| "Winning config" / "the winner" | outcome phrasing | in Ch.4: omit; in Ch.6: "S1, selected by the Tier-2 rule" |
| "clean win" | editorialising | "under the evaluated configuration, X showed lower held-out performance" |
| em dash | style rule | use comma, semicolon, or recast |
| "X, not Y" framing | contrastive filler | restate as positive claim |

---

## Group 3 — Methods/results boundary: sentences belonging in Ch.6, not Ch.4

| Source | Banned construction | Belongs in |
|---|---|---|
| METHODS.md §1.3 | persistence MSE values and "learned something real" | §6.3 |
| METHODS.md §1.7 | fork verdicts, void/kept labels | §6.4 |
| METHODS.md §1.8 | the three ad-hoc Tier-4 AUC rows | §6.4/§6.5 |
| reconstruction-value-ablation.md §Results | arm AUC numbers | §6.3 |
