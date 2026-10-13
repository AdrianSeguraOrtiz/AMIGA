# Phase 4: held-out topology evaluation

This phase evaluates the procedures selected by the complete
[sequential phases 1–3](sequential-selection.md). It uses the same five outer
partitions, with whole topologies kept together. The benchmark has already been
studied: these are outer out-of-fold estimates on reused benchmarks, not a new
independent external dataset. The completed selection's sources and artifacts
remain immutable. Phase 4 has its own contract, package and outputs.

## Frozen selections and final fits

Before starting, verify all completed selection jobs and summary artifact hashes,
recompute the phase-3 winners from inner predictions, and freeze the 30 selected
procedures: two cases, five outer folds and three formulations. Each procedure
specifies its family, relevance label, parameters and retained feature fraction.
No parameter search is repeated and no selection uses outer test scores.

Refit each procedure on its complete outer training complement with seeds
1201, 1202, 1203, 1204 and 1205: **150 final model fits**. All fits request 3,000
boosting iterations, using the same adapters and weight policies as selection.
There is no evaluation set or validation-based early stopping. As in selection,
LightGBM may exhaust admissible splits; record actual iterations. Ranking label
construction retains seed 1101 (relevant if shuffled labels were selected).

For a reduced feature fraction, relearn its column identities once using only
that outer training complement and the selected family's parameters. Retain
selection seed 1101 and SHAP sampling seed 1501, at most 32 candidates per front,
and the same centered, topology-weighted native SHAP criterion. Traverse
100% → 75% → 50% → 25% only as far as the selected fraction. Fit the parent at
each elimination step; the final selected representation is fitted by the five
final-seed jobs. Inner-fold masks are never reused as outer masks.

The learned mask is shared across the five final seeds. Thus seed variation
measures final estimator randomness conditional on the chosen procedure and
mask, rather than rerunning selection five times. A 100% selection needs no
mask fit. Additional mask fits depend on the fractions selected by inner
validation; the job plan records their counts before execution.

Training labels are transformed after restricting to training fronts. Held-out
predictions are computed from predictors only and saved before loading held-out
AUPR for evaluation. Save per-candidate scores, per-front metrics, selected
columns, fitting reports and hashes. These are evaluation fits, not deployment
models for a biological application.

## Comparators and summaries

Evaluate ranking, direct AUPR regression and top-20% classification on identical held-out candidates. Each formulation
uses its independently selected family/parameters/fraction. Also evaluate all
objective-only selectors from the experimental design, including trade-off-worthiness
(knee), and the exact uniform-random reference. Retain the AUPR oracle as a
nondeployable ceiling. The method ID `ranking` now denotes the selected ranking
procedure; it must not be labeled as a universally fixed CatBoost model.

Regret@5 remains primary and Regret@1 secondary. Export Regret, BestAUPR and Hit
at k=1,3,5,10 with the existing exact score-tie expectation. Average **metrics**
over all five seeds within each front, then conditions within topology, then
equally over the 87 topologies. Do not average scores into an ensemble or select
a favorable seed. Export front-macro sensitivity results and fold/seed diagnostics.

Paired summaries use 10,000 topology bootstrap resamples (seed 1401) for
conditional descriptive intervals. The publication comparison uses mean ranks,
tie-corrected Friedman and Holm control comparisons for Regret@5, with separate
supervised and objective-selector panels; see [reporting](reporting.md).
Overlapping training sets and reused benchmarks limit inferential independence.
Summarization requires every job. Recompute metrics from all saved predictions,
verify exact candidate/fold/seed coverage and learned mask dependencies, and
export tables plus PNG/PDF figures. Pipeline completion means both execution
and these audited summaries succeeded.

## Execution and monitoring

A 64-CPU machine can use 16 workers × 4 threads. Workers have
disjoint affinity sets and single-thread BLAS pools. Validate available capacity
and record the environment. Stop on a technical error or timeout; no automatic
retries, skipped results, or outcome-based stopping. Limits remain two hours
per job and six cumulative days. Resume verifies the contract, source/data hashes,
environment, resources and every completed artifact. Explicit retries preserve
previous attempts; a source fix requires a new contract/run.

From the repository root:

```bash
scripts/experiments/amiga-exp benchmark comparison freeze \
  --selection-run experiments/sequential-selection/runs/full-001 \
  --selection-summary experiments/sequential-selection/summaries/full-001 \
  --output experiments/outer-evaluation/contract-001.json

.venv/bin/python -u -m scripts.experiments.amiga_exp.outer_evaluation.pipeline \
  --contract experiments/outer-evaluation/contract-001.json \
  --output experiments/outer-evaluation/launches/evaluation-001 \
  --run-output experiments/outer-evaluation/runs/evaluation-001 \
  --summary-output experiments/outer-evaluation/summaries/evaluation-001 \
  --jobs 16 --threads 4
```

Launch the pipeline detached with stdout/stderr redirected to a log. All three
output directories must be new. Monitor the pipeline and its live child state:

```bash
scripts/experiments/amiga-exp benchmark comparison status \
  --run experiments/outer-evaluation/launches/evaluation-001

tail -f experiments/outer-evaluation/runs/evaluation-001/supervisor.log
```

The launch state progresses through `running_evaluation`, `summarizing`, and
`complete`; `failed` or `interrupted` includes an error. The child state contains
the live completed/total job counts, active jobs, elapsed time and heartbeat.
Each job has a `worker.log` and a checksummed completion `result.json`.

`benchmark comparison run --dry-run` writes the contract/plan without fitting. To resume an
existing execution use `benchmark comparison run --contract ... --output ... --jobs 16 --threads 4
--resume`, adding `--retry-failed` only after inspecting a technical failure.
Manual resumption runs the jobs; then use `benchmark comparison summarize --run ... --output ...`
with a new summary destination. The pipeline is the automatic execution-plus-summary
entry point. Learning curves, deployment refits and
biological evidence analysis are separate blocks and are not launched here.

## Classification thresholds

After the main comparison, run `benchmark comparison top5` and then
`benchmark comparison top10`. Their [top-5%](top5-classification.md) and
[top-10%](top10-classification.md) guides describe the same full-grid,
training-only selection followed by held-out fits. Reports combine the saved
predictions of all five supervised formulations without refitting AMIGA.
