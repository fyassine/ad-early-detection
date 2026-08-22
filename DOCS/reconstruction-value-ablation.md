# How valuable is reconstruction to our classification task?

**Status:** initial single-seed (seed 42) results below — all four arms run 2026-08-17 on
commit `20b5957`. Not yet repeated across seeds; see caveats before drawing conclusions.

## The question

The GAAE is pretrained with a *reconstruction* objective (rebuild the node features and the
adjacency of each visit's functional-connectivity graph). Its encoder is then frozen and
reused as a feature extractor by the downstream converter-vs-stable-MCI classifiers
(GELSTM / GEGRU, and offline for GEC / GEP).

Nothing in that pipeline has ever tested whether the reconstruction step earns its place.
Three distinct things could be doing the work:

1. the **reconstruction pretraining** (what the encoder learned from unlabelled graphs),
2. the **encoder architecture** (a 3-layer GATv2 with FiLM conditioning is a strong prior
   even at random initialisation — random graph convolutions are a decent feature map),
3. the **encoder existing at all** (versus feeding the classifier the raw pooled ROI rows).

The four arms below separate them.

## The four arms

One knob, `encoder_init`, on `GELSTMTrainConfig` (and thus on any `configs/*.json` or
`experiments.yaml` `hyperparams:` block). Implemented in
[`CLASSIFIER/configs/encoder.py`](../CLASSIFIER/configs/encoder.py).

| Arm | Encoder built? | GAAE weights loaded? | Encoder trained? | Isolates |
|---|---|---|---|---|
| `pretrained_frozen` | yes | yes | no | **Reference.** Exactly today's model. |
| `pretrained_finetuned` | yes | yes | yes | Value of *adapting* the reconstruction features to the conversion task. |
| `random` | yes | **no** | yes | Value of the **reconstruction pretraining itself** — architecture held constant. |
| `none` | **no** | no | — | Value of the **graph encoder itself** — the mean-pooled raw node features go straight to the LSTM. |

Read the arms as a chain of differences:

```
none  ──(+ graph encoder)──►  random  ──(+ reconstruction pretraining)──►  pretrained_frozen
                                                                    └──(+ task adaptation)──►  pretrained_finetuned
```

### What changes under the hood

* `none` builds **no encoder module at all** (`model.encoder is None`, zero parameters whose
  name starts with `encoder`). `encode_visit` pools the raw node features instead, so the
  per-visit embedding width becomes `in_features` (200 for Schaefer-200) rather than
  `gaae_latent` (64). The recurrent core's input width is **derived** from that
  (`lstm_input_dim = embed_dim + 1 if use_time_delta`), never hardcoded — the LSTM and the
  classifier head resize themselves.
* `random` builds the identical encoder architecture, skips the checkpoint entirely
  (the adapter prints `GAAE checkpoint NOT loaded`), and trains it end-to-end with the
  classifier.
* **Gradient plumbing (important).** `encode_batch_sequences` historically embedded every
  visit inside `torch.no_grad()` + forced eval mode, so the encoder was a pure feature
  extractor *no matter what `requires_grad` said* — the pre-existing `freeze_encoder: false`
  path could not actually finetune anything. A new `EvalConfig.encoder_grad` flag (default
  `False`) opens that context, and the adapter sets it **only** for the training pass of
  the `pretrained_finetuned` / `random` arms. Evaluation always embeds under no-grad + eval
  mode. `CLASSIFIER/tests/test_encoder_arms.py::test_encoder_grad_false_starves_a_trainable_encoder`
  pins that this flag is what makes the `random` arm meaningful.

### Back-compat contract

`encoder_init` defaults to `None`, meaning *"not set — derive the arm from the legacy
`freeze_encoder` flag"*. Every existing config (`freeze_encoder: true`) therefore resolves
to `pretrained_frozen` and runs exactly as before; `encoder_grad` defaults to `False`, so
the training path is byte-identical to today's. Setting `encoder_init` and `freeze_encoder`
to contradictory values raises `ValueError` rather than silently picking one.

### Scope: GELSTM only

The four arms are wired for the **GELSTM/GEGRU** adapter, the only downstream model whose
encoder lives *inside* the trained network and can therefore be trained end-to-end. The
`GEC` and `GEP` adapters precompute embeddings once with a frozen encoder and then train an
MLP on the resulting vectors; `pretrained_finetuned` and an end-to-end `random` arm have no
meaning there without restructuring those adapters. If the GELSTM result says the encoder
is not earning its place, the same question for GEC/GEP is worth a follow-up — it is not
answered here.

## Running it

From `CLASSIFIER/`, with the project-root `.venv` active:

```bash
source ../.venv/bin/activate

# Preview the merged parameters for one arm without executing anything:
python run_experiment.py --id recon-ablation-gelstm-pretrained-frozen --dry-run

# The four arms, in order (each is a full 5-fold CV run):
python run_experiment.py --id recon-ablation-gelstm-pretrained-frozen
python run_experiment.py --id recon-ablation-gelstm-pretrained-finetuned
python run_experiment.py --id recon-ablation-gelstm-random
python run_experiment.py --id recon-ablation-gelstm-none

# Or queue all four detached, then watch:
python run_experiment.py --mode longitudinal --background   # runs every longitudinal entry
python run_experiment.py --status
python run_experiment.py --collect                          # rebuild outputs/RESULTS.csv
```

Registry: [`CLASSIFIER/experiments/ablation.yaml`](../CLASSIFIER/experiments/ablation.yaml).
Shared hyperparameters: [`CLASSIFIER/configs/gelstm_recon_ablation_delcode.json`](../CLASSIFIER/configs/gelstm_recon_ablation_delcode.json)
— a copy of the headline GELSTM config with `freeze_encoder` dropped, so the arm comes from
one place only.

All four arms use `seed: 42`, the same `n_folds: 5`, the same `downstream` split CSVs and
the same notebook, so the folds and the held-out test subjects are identical across arms —
the comparison is apples-to-apples subject-for-subject.

`checkpoint_path` is present on all four entries because the shared notebook requires one;
the `random` and `none` arms never read it. Verify that in each run's `run.log`: those two
must print `[encoder_init=…] GAAE checkpoint NOT loaded`.

### Before comparing

1. Confirm all four runs are `done` in `--status` and share the same `git_commit`.
2. Check `resolved_config.json` in each run dir: `encoder_init` should be the arm, and every
   other key identical across the four.
3. Read metrics from `outputs/RESULTS.csv` (`cv.val_auc_mean` ± `cv.val_auc_std`, and the
   `metric.test_*` columns). The test threshold is the OOF-derived one — never re-tune on test.

## Results (4 seeds, updated 2026-08-22)

Recomputed directly from `CLASSIFIER/outputs/**/run_summary.json` rather than transcribed.
Test AUC is the mean over per-seed means (sample SD, `ddof=1`, over the 4 seeds); CV AUC
pools every completed run.

| Arm | Seeds | Runs | Test AUC | Test F1 | CV AUC | Trainable params |
|---|---|---|---|---|---|---|
| `none` | 4 | 5 | **0.8313 ± 0.0529** | **0.7028** | 0.8932 ± 0.0073 | 31,169 |
| `pretrained_frozen` | 4 | 4 | 0.7812 ± 0.0122 | 0.6048 | **0.9211 ± 0.0087** | 520,905 |
| `random` | 4 | 4 | 0.7312 ± 0.1113 | 0.6080 | 0.7676 ± 0.0514 | 965,897 |
| `pretrained_finetuned` | 4 | 4 | 0.7107 ± 0.1946 | 0.5842 | 0.8507 ± 0.0195 | 965,897 |

Per-seed test AUC:

| Arm | 42 | 43 | 44 | 45 |
|---|---|---|---|---|
| `none` | 0.7607, 0.7607 | 0.8643 | 0.8214 | 0.8786 |
| `pretrained_frozen` | 0.7821 | 0.7643 | 0.7929 | 0.7857 |
| `random` | 0.5714 | 0.8250 | 0.7464 | 0.7821 |
| `pretrained_finetuned` | 0.7821 | **0.4214** | 0.7964 | 0.8429 |

### Reference floors (both required by the doc's own caveats)

| Floor | CV AUC | Test AUC |
|---|---|---|
| `sanity-metadata-baseline` (age, sex, visit timing; GradientBoosting) | 0.6157 ± 0.0653 | 0.4929 |

Every arm clears the metadata floor on CV AUC, so all four are saying something about
imaging rather than about demographics. Note the metadata floor's *test* AUC is at chance
(0.4929) while its CV AUC is 0.6157 — with 34 test subjects that gap is within noise, and it
is a good reminder of how little a single test point estimate carries here.

### Reproducibility note (new)

`recon-ablation-gelstm-none` was run **twice at seed 42 and produced identical results**
(test AUC 0.7607 both times). The `none` arm builds no encoder and therefore executes no
scatter backward, which is exactly the operation class that makes BrainTokenGT
non-reproducible (see `DOCS/draft_paper/SOTA_POSITIONING.md`). The other three arms have one
run per seed, so **their within-seed variance is unmeasured** — and for
`pretrained_finetuned` it is known to be non-zero: a re-run at seed 43 diverged from the
original on folds 1 and 4. A repeat sweep is in progress; until it lands, treat that arm's
±0.1946 as conflating seed and run noise.

### Read against the pre-registered table

The observed pattern matches **two** rows of the table below, not one:

* **Row 2 — `none` ≈ every other arm.** `none` has the *highest* test AUC and F1 of the four
  and a CV AUC (0.8932) within fold-to-fold noise of `pretrained_frozen` (0.9211), using
  ~17× fewer trainable parameters. The graph encoder is not earning its place.
* **Row 5 — `pretrained_finetuned` < `pretrained_frozen`.** Freezing beats fine-tuning, and
  fine-tuning is far noisier (SD 0.1946 vs 0.0122). **But see the confound below before
  treating this as a finding.**

`random` is the worst arm on CV AUC (0.7676 ± 0.0514, the largest spread of the four), so
pretraining *does* beat random initialisation when the encoder is trained end-to-end. Since
`none` also beats `random`, this does not support a clean "pretraining transfers real
structure" story (row 3): a model with *no* graph encoder outperforms a trained-from-scratch
one.

**Net read:** the reconstruction pretraining's measurable benefit is **optimisation
stability relative to random initialisation, not peak performance** — and the graph encoder
itself is not adding value over mean-pooled raw connectivity features at this cohort size.

### ⚠️ The fine-tuning conclusion is withdrawn, not caveated

`configs/gelstm_recon_ablation_delcode.json` sets a single `learning_rate: 0.001` and
`adapters/gelstm.py` builds one optimizer param group. The unfrozen pretrained GATv2 encoder
is therefore fine-tuned at the **same LR as the freshly initialised head** — the textbook way
to destroy pretrained features, and precisely the treatment BrainTokenGT's own stabilisation
applied to *its* newly unfrozen parameters (`give_lr_scale=0.1`) and this arm did not.

**The arm measures naive fine-tuning, not fine-tuning.** Any statement about whether
fine-tuning helps needs an `encoder_lr_scale` arm before it can be made.

## Pre-registered interpretation

Written **before** any numbers exist, so the conclusion cannot be fitted to the result.
Read the first row that matches; "≈" means the difference is inside the CV fold-to-fold
spread (roughly: |ΔAUC| smaller than the larger of the two arms' CV std).

| Observation | What it means | What follows |
|---|---|---|
| `random` ≈ `pretrained_frozen` ≈ `pretrained_finetuned` | Reconstruction pretraining adds **nothing** beyond the architecture prior. The GAAE is functioning as a fixed random-ish graph feature map. | Drop the pretraining stage from the thesis pipeline, or keep it only as a compute-saving initialisation. Report it as a negative result — it is a real finding about this cohort size, not a failure. |
| `none` ≈ every other arm | The **graph encoder itself** is not earning its place; the pooled raw ROI features carry the same signal. | The story is about the longitudinal head (LSTM + Δt), not the graph model. Re-frame accordingly and compare against a plain per-visit feature baseline. |
| `pretrained_*` > `random` (both clearly above `none`) | Reconstruction pretraining contributes **real** transferable structure beyond architecture. This is the result the current pipeline assumes. | Keep the GAAE stage; quantify the gain (ΔAUC) as the headline justification for it, and consider more/longer pretraining. |
| `pretrained_finetuned` > `pretrained_frozen` | The pretrained features are useful but **mis-adapted** to the conversion task; freezing costs performance. | Switch the headline model to the finetuned arm (watch for overfitting — check the CV/test gap, converter *n* is small). |
| `pretrained_finetuned` < `pretrained_frozen` | Finetuning **overfits** the small converter cohort; freezing is a regulariser doing real work. | Keep freezing; say so explicitly, since "we froze it" then becomes a justified choice rather than an inherited default. |
| `random` > `pretrained_*` | Reconstruction pretraining is actively **harmful** — it optimises for rebuilding edges, which may wash out the between-subject variance the classifier needs. | Investigate the reconstruction objective (see `DOCS/reconstruction-fidelity-pearson-r.md`); consider a contrastive or supervised pretraining objective instead. |
| All four arms ≈ each other **and** ≈ the metadata-only floor | Nothing in the imaging pipeline is contributing signal at this sample size. | This is a power/cohort problem, not an encoder problem. Escalate to sample size and label definition before any further architecture work. |
| Fold-to-fold std is large enough that every arm overlaps every other | The comparison is **underpowered** at 5 folds. | Do not conclude anything from a single seed. Repeat all four arms over ≥3 seeds (e.g. 42/43/44) and compare the pooled distributions before writing any of the above. |

### Caveats to state alongside whatever you find

* **One seed.** The arms share seed 42 for comparability, which controls the split but not
  the training noise. Any difference smaller than the fold-to-fold spread needs repeat seeds
  before it is reportable.
* **`random` and `none` train more parameters** than the frozen reference (the encoder joins
  the optimiser, or the LSTM widens to a 200-d input). If they lose, part of that may be an
  optimisation/overfitting effect rather than a statement about pretraining. Check the
  trainable-parameter counts printed at model build and the train/val curves before
  concluding.
* **Feature standardisation.** `standardize_features: true` fits a per-fold `StandardScaler`
  on the *initial* encoder's pooled embeddings and then holds those statistics fixed. For the
  arms whose encoder keeps training, that is a fixed affine on a moving representation —
  harmless, but note it. Set `standardize_features: false` if you want that variable removed.
* **Converter class imbalance** is handled by `use_class_cost_weights: true` in all four arms;
  do not change it in one arm only.
