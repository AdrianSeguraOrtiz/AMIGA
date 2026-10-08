# Experimental workflow

This is the entry point for the current BIO-INSIGHT and MO-GENECI benchmark
workflow. Its [scientific design](experiments/design.md) is separate from the
public `amiga` API. Run repository experiments through
`scripts/experiments/amiga-exp`; install dependencies with
`poetry install --with experiments`.

The current source release is **amiga-exp 0.2.0**, tagged `amiga-exp-v0.2.0`.
Start with the [installation and input-data guide](experiments/reproducibility.md)
and [release notes](experiments/releases.md). The PyPI package retains its own
independent version. Check this workflow with `scripts/experiments/amiga-exp --version`.

## Current phases

Both cases use 104 fronts, 87 topology groups, five outer folds and three inner
folds. All conditions of a topology remain together. Configuration selection
uses seed 1101; final fits use seeds 1201–1205.

| Phase | Purpose | Implementation and specification |
| --- | --- | --- |
| 0 | Validate inputs, freeze partitions and calibrate CPU resources | [Grouped data policies](experiments/grouped-validation.md) and [sequential execution](experiments/sequential-selection.md) |
| 1 | Choose relevance labels separately for each ranker | `sequential`: LightGBM, XGBoost and CatBoost |
| 2 | Tune parameters using the complete family-specific grids | `sequential`: inner validation only |
| 3 | Select columns and model family | `sequential`: recursive training TreeSHAP and inner validation |
| 4 | Evaluate the selected procedures on held-out topologies | [`outer`](experiments/outer-evaluation.md): five final seeds and objective-only comparators |
| Classification thresholds | Extend the supervised comparison | [`top5-classification`](experiments/top5-classification.md) and [`top10-classification`](experiments/top10-classification.md) |

AMIGA is the procedure that selects a family, relevance labels, parameters and
feature fraction inside each outer training complement. It is not a universally
fixed CatBoost configuration. The main supervised presentation contains AMIGA,
direct AUPR regression and top-5%, top-10% and top-20% classification.

The detailed specifications describe the immutable runs that produced their
artifacts. Some complete runs contain additional methods. Reporting a selected
method set does not modify those contracts, fit counts or original results.
The 5% and 10% thresholds are exploratory sensitivity analyses.

## Completed runs and monitoring

Inspect completed pipelines without launching training:

```bash
scripts/experiments/amiga-exp sequential status \
  --run experiments/sequential-selection/runs/full-001

scripts/experiments/amiga-exp outer status \
  --run experiments/outer-evaluation/launches/evaluation-001

scripts/experiments/amiga-exp top5-classification status \
  --run experiments/top5-classification/launches/full-001

scripts/experiments/amiga-exp top10-classification status \
  --run experiments/top10-classification/launches/full-001
```

For a new execution, use the freeze and pipeline commands in the corresponding
specification, with new output directories. The complete selection uses the
original parameter grids. Technical failures require inspection and explicit
resumption; source or policy changes require a new contract.

The measured parallel layout is 16 workers with four disjoint CPU threads each
on the 64-CPU machine, with single-thread BLAS pools. Runtime limits, package
versions, effective parameters and input/source hashes belong to each run.

## Results and figures

| Output | Location relative to the repository |
| --- | --- |
| Selected procedures, candidate scores and column stability | `experiments/sequential-selection/summaries/full-001/` |
| Phase-1 label figures | `plots/<case>/outer-<fold>/label_screening.{csv,png,pdf}` within the selection summary |
| Phase-2 tuning figures | `plots/<case>/outer-<fold>/hyperparameters.{csv,png,pdf}` within the selection summary |
| Phase-3 column curves | `plots/<case>/outer-<fold>/feature_curves.{csv,png,pdf}` within the selection summary |
| Column inclusion figures | `plots/<case>/column_stability.{png,pdf}` within the selection summary |
| Original phase-4 evaluation | `experiments/outer-evaluation/summaries/evaluation-001/` |
| Top-5% selection and evaluation | `experiments/top5-classification/full-001/{selection-summary,outer-summary}/` |
| Combined results including all three classification thresholds | `experiments/top10-classification/full-001/outer-summary/` |

The combined `topology_metrics.csv` contains seed-averaged and condition-averaged
metrics for all 87 topologies in each case. Use it for final mean-rank tables.
`front_metrics.csv` provides the 104-front sensitivity summary;
`metrics_long.csv` preserves seed-specific rows. Source manifests identify which
completed runs supplied the reused comparator predictions.

Regret@5 remains primary. Regret@1, Hit@1 and Hit@5 provide secondary context.
The selected five-method rank presentation can be regenerated from saved metrics
without fitting models using [`report supervised`](experiments/reporting.md).
The [design](experiments/design.md) describes its exploratory Friedman and Holm analysis.

Selection figures describe inner-validation decisions; final comparison figures
describe outer predictions. Compact figures must preserve this distinction and
represent all five outer folds. The top-5% and top-10% searches have complete
candidate tables; their tuning and column-selection figures still require
presentation work. Existing complete comparison figures may contain additional
methods and can be rendered again for the selected presentation.

## Remaining evaluation blocks

Phases 0–4 and the additional classifier evaluations are complete. The following
blocks need separate treatment before being attributed to the current procedure:

- Learning curves: existing runs use an earlier fixed configuration. Evaluating
  label scarcity for the current procedure requires restricting training and
  configuration selection to the available labels at each size.
- Leave-family-out evaluation: earlier results describe their recorded fixed
  configuration. Updating it is a separate experiment if that claim is retained.
- TCGA-BRCA application: select and fit deployment procedures using the 104
  benchmark fronts, score the existing real front, and recalculate source support
  for the resulting recommendations.

The existing `real-world-validate <case_dir>` command regenerates the earlier
Top1 source-support table from an already ranked front. It does not select or
train the current deployment model. The input front and evidence resources can
be reused, with their versions and provenance checked.

## Compatibility and generated files

`grouped`, `run-phase`, `run-all`, `summarize-paper`, `plot-phase` and `plot-all`
remain available for their recorded workflows. They do not replace the current
sequential and outer pipelines.

Generated contracts, raw results and figures belong under the Git-ignored
`experiments/` directory. Frozen dependencies stay at their recorded paths.
Protocols, source code and tests are versioned; private working archives stay
under the Git-ignored `.local-work/` directory.
