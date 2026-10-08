# Sequential model and column selection

Implemented training-only phases 0–3. This specification supersedes the draft
`ranker-and-feature-selection-plan.md`. Existing grouped-evaluation runs retain
their original contracts and results. These experiments do not change the public
`amiga` API or its default model.

## Scientific scope and phases

Use the same 104 fronts, 87 topology groups, five outer folds and three inner
folds as the [grouped protocol](grouped-validation.md). All conditions belonging
to the same topology stay together. Each phase runs separately inside each
outer training complement, for both BIO-INSIGHT and MO-GENECI. Outer held-out
outcomes never enter fitting, feature selection, candidate selection or these
plots. The existing benchmark has already been studied: this is internal
validation on reused benchmarks, not a new independent external dataset.

| Phase | Question | Fixed settings | Output |
| --- | --- | --- | --- |
| 0 | Is execution feasible and reproducible? | Inputs, topology partitions, resource limits and decision rules | Checksummed contract and training-only timing pilot |
| 1 | Which relevance labels suit each ranker? | Reference parameters, all predictors | One label mode per family and outer fold |
| 2 | Which parameters suit each family/formulation? | Phase-1 label for ranking, all predictors | One parameter configuration per family/formulation and outer fold |
| 3 | Which columns and family should the procedure use? | Phase-2 parameters and selected ranking label | An internally selected family and feature fraction per formulation and outer fold |

LightGBM, XGBoost and CatBoost all reach phase 3. The four formulations are
ranking, direct AUPR regression, within-front min–max AUPR regression, and
classification of the top 20% candidates. Regression and classification have
fixed label definitions and therefore no phase-1 relevance search. They receive
the same parameter grid and feature budgets as ranking within each family.

Use training seed 1101 throughout selection. The five final seeds 1201–1205
belong to the later outer evaluation and are **not** five repetitions of each
parameter search. Phase 4 must refit each selected procedure using only its
outer training complement, relearn its feature identities there, and then
predict held-out fronts. Phases 0–3 do not launch that evaluation, learning
curves, deployment refits or biological evidence analysis.

## Labels, fitting and candidate selection

Phase 1 compares dense ranks, average ranks, continuous relevance, quantiles
with 5/10/15 bins, reversed labels and shuffled labels. As in the original
screening rule, all eight modes are eligible, including the two controls; a
control winner is retained and must be reported. Labels are constructed within
training fronts. Linear gain is used for LightGBM/XGBoost ranking; continuous
relevance is mapped to integers by `floor(255 * relevance)` for those libraries.
CatBoost receives the original relevance values. This discretization is a
library compatibility choice, not a tuned resolution.

Each candidate is evaluated on all three inner validation folds. Average
conditions within each topology, then average topologies with equal weight.
Select by mean Regret@5, then mean Regret@1 (rounded to 12 decimal places for
numerical ties), then stable candidate ID. In phase 3, fewer retained columns
precedes the stable ID tie-break. Query IDs remain individual fronts; topology
IDs are used for partitioning and weighting, never as predictors. Training
weights, flat-front handling, classification ties and evaluation score ties
follow the grouped protocol. Validation fronts are retained even if flat.

Reference fits use 2,000 boosting iterations; phases 2 and 3 use 3,000. No
validation set or validation-driven early stopping is supplied to a fit.
LightGBM may finish earlier when it has no admissible split; actual iterations
are recorded and the candidate remains eligible. LightGBM uses
`subsample_freq=1`, explicitly activating the stated 0.8 row subsample. The
legacy grid did not set this frequency. These adapter choices are changes to
the fitting specification, even when grid values match the original study.

## Parameter grids and compute gate

The preferred grid retains the original family-specific search values:

| Family | Search values | Configurations |
| --- | --- | ---: |
| LightGBM | leaves 31/63/127; minimum leaf samples 30/50/100; learning rate .03/.05/.10 | 27 |
| XGBoost | depth 4/6/8; subsample .8/1; minimum child weight 1/5/10; learning rate .03/.05 | 36 |
| CatBoost | depth 4/6/8; L2 3/5/7/10; learning rate .03/.05/.10 | 36 |

LightGBM and XGBoost column subsampling is .8. The complete-grid study uses
all 27/36/36 configurations for every formulation within the corresponding
family. The run never reduces a grid automatically. Different family counts
reflect the original parameter spaces, not equal wall-clock cost or exhaustive
optimization over all possible hyperparameters.

A completed training-only pilot established compatibility of all families,
formulations and feature budgets. Its source snapshot and artifacts remain
available for provenance. Runtime projections from that pilot assumed only two
workers and high-complexity configurations; they do not determine the grid used
by the complete study.

A separate hardware calibration compares identical training-only tasks under
8 threads per worker, 4, 2 and 1, filling the machine's available CPU capacity.
On a 64-CPU machine these are 8×8, 16×4, 32×2 and 64×1. The task mix covers both
cases, all three families and four formulations, with four repetitions: 96 fits
per layout. Calibration uses 1,000 boosting iterations and up to eight SHAP rows
per training front; it computes no validation metrics. The fastest measured
layout is selected before starting the complete run. The scientific fits retain
2,000 iterations in phase 1, 3,000 in phases 2–3 and 32 SHAP rows per front.

Worker CPU sets are disjoint and set before numerical libraries are imported.
OpenMP and estimator thread limits follow the chosen layout; BLAS pools use one
thread. The product of workers and threads cannot exceed the CPU affinity or
cgroup quota available to the process. The environment, layout and CPU sets are
recorded and must match on resume. High utilization is checked in the actual
selection run; the purpose of calibration is completed work per unit time.

| Phase | Model fits | Jobs |
| --- | ---: | ---: |
| 1: labels | 720 | 240 |
| 2: complete parameter grids | 11,880 | 3,960 |
| 3: recursive column selection | 1,440 | 120 |
| Total | 14,040 | 4,320 |

Selection has a cumulative six-day (144-hour) limit and a two-hour per-job
limit. A phase-3 job contains 12 fits. A technical failure stops execution;
there are no automatic retries or outcome-based exclusions. Completed
artifacts, versions and source hashes are checked before resuming. A source
fix requires a new contract/run. The original two-worker technical pilot has
its own completed contract and is not resumed with changed sources.

## Recursive column selection

For each inner training set, fit with all columns. Compute native TreeSHAP
contributions on at most 32 uniformly sampled candidates per training front
(sampling seed 1501, independent of quality). Center each column's contributions
within its sampled front, take mean absolute centered contribution, and aggregate
with equal topology weight. Centering emphasizes differences that can change
candidate order within a front; uncentered mean absolute SHAP is also saved.
Contributions sum to raw model output (log-odds for classifiers), not probability.

Keep the highest scoring 75% of original columns, refit with the same parameters,
and repeat to 50% and 25%. Counts round up. Ties use column name. All three
families and all four formulations use this recursive procedure. The 100% model
remains eligible. Inner validation scores choose the fraction and, after all
families finish, the family. Validation outcomes do not determine the elimination
order. A selected fraction is a procedure: feature identities may differ across
training splits and must be relearned before outer evaluation.

This is training-based predictive importance and selection, not causal feature
importance. Correlated variables may substitute for each other. Stability is
descriptive inclusion frequency across 15 overlapping inner training fits per
family/formulation; those fits are not 15 independent statistical replicates.

## Commands, status and artifacts

```bash
.venv/bin/python -u -m scripts.experiments.amiga_exp.sequential_selection.throughput \
  --output experiments/sequential-selection/throughput/throughput-001

scripts/experiments/amiga-exp sequential freeze \
  --profile original --budget-hours 144 \
  --output experiments/sequential-selection/full-contract-001.json
```

After calibration, launch the following command using the measured worker and
thread counts. Run it in a detached process with stdout/stderr redirected to a
log. All three output directories must be new:

```bash
.venv/bin/python -u -m scripts.experiments.amiga_exp.sequential_selection.pipeline \
  --contract experiments/sequential-selection/full-contract-001.json \
  --output experiments/sequential-selection/launches/full-001 \
  --run-output experiments/sequential-selection/runs/full-001 \
  --summary-output experiments/sequential-selection/summaries/full-001 \
  --jobs <measured_workers> --threads <measured_threads>
```

The launch `state.json` records running selection, summarizing, complete,
interrupted or failed. The selection run has its own live `state.json`, with completed jobs
by phase, active jobs, PID, elapsed time and any failure. Monitor it with:

```bash
scripts/experiments/amiga-exp sequential status \
  --run experiments/sequential-selection/runs/full-001
```

During calibration, inspect its `state.json` and `report.json` instead. Job-specific details
are under `jobs/<job-id>/attempt-001/{worker.log,result.json}`. A failed attempt
is preserved; an explicit technical retry requires `run --resume --retry-failed`
with the original contract. Source fixes require a new contract/run. A completed
run can regenerate summaries in a new directory using `sequential summarize`.

Summary artifacts include all candidate scores, selected procedures per outer
fold, selected inner masks, full column frequencies, input/output hashes, and
PNG/PDF/CSV figures. Phase 1 adapts the original family-by-label heatmap; phase 2
adapts the original mean/dispersion scatter and reuses its palette helper. Both
use current topology-weighted inner results and mark the retained choice within
each family. Phase 3 adds performance-by-column-count curves and a full column
inclusion heatmap. There are no significance claims in selection figures.

**Inner selection scores are optimistic development diagnostics, not final
performance estimates.** Figures identify their case, outer training complement
and inner-validation scope. Do not choose a final family by pooling outer test
results or describe an inner-selection win as evidence of general superiority.
