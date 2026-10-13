# Study 1: labelled benchmark evaluation, phases 1–5

The [design](design.md) defines grouped partitions, selection and statistics.
The same fronts and final seeds underpin phases 4 and 5. Phase numbering describes
the scientific questions; the execution engine combines dependent tasks.

## Phases 1–3: select the AMIGA procedure

`amiga-exp benchmark selection` executes the complete parameter grids. Use its `freeze`, `run`,
`status` and `summarize` commands according to the
[frozen selection specification](sequential-selection.md).

1. Screen relevance labels separately for LightGBM, XGBoost and CatBoost.
2. Tune parameters using three grouped inner folds.
3. Select a column budget and family using recursive training TreeSHAP and
   inner validation. Reduced masks are subsequently learned on the complete
   outer training complement, before held-out evaluation.

No phase uses outer-test results to choose a configuration.

## Phase 4: compare selectors

`amiga-exp benchmark comparison` freezes, runs, monitors and summarizes phase 4.
The [evaluation specification](outer-evaluation.md) describes fitting and metrics.
`benchmark comparison top5` and `benchmark comparison top10` preserve the
classifier extensions and their fixed dependency order.

Generate phase-4 statistical reports and the phase 1–4 figures from saved results:

```bash
scripts/experiments/amiga-exp benchmark statistics \
  --summary experiments/top10-classification/full-001/outer-summary \
  --output experiments/reports/supervised-new
scripts/experiments/amiga-exp benchmark figures \
  --output experiments/reports/figures-new
```

Outputs must be new directories. The [figure guide](figures.md) describes the
eight main figures and 26 supplements. Phase 4 has separate supervised and
objective-selector panels, plotting Regret@5 ranks, means and Friedman/Holm.
Hit@1 and Hit@5 remain available for textual reporting.

## Phase 5: fixed-configuration learning curves

Reuse each outer fold's exact phase-4 family, relevance labels, hyperparameters
and predictor columns. Refit at 10, 20 and 40 labelled topology groups, with three
recorded nested subsets and final seeds 1201–1205. Keep the original test folds.
The full-size point reuses the saved phase-4 predictions, with 69–70 training
topologies. The phase has 90 independent jobs and 450 fits; it includes AMIGA only.

The recipe and columns were selected with the original full training complement.
This analysis measures data-size sensitivity conditional on that choice; it does
not repeat selection with scarce labels. Phase 4 supplies a full-size reference,
not a test-driven choice of recipe.

Average seeds within subset/front, subsets within front, then conditions within
topology, with equal topology weights. Bands are conditional 95% intervals from
10,000 topology bootstrap resamples, seed 1401. The figures show Regret@5 and
Hit@5; the CSVs retain all twelve benchmark metrics.

Inspect the completed phase or regenerate its report without model fitting:

```bash
scripts/experiments/amiga-exp benchmark learning status
scripts/experiments/amiga-exp benchmark learning report
scripts/experiments/amiga-exp benchmark learning report \
  --output experiments/reports/phase5-new
```

The `--bundle` option defaults to `experiments/supplementary/fixed-001` and accepts
another complete joint execution. With no `--output`, `report` only reads results.
With `--output`, it verifies and copies the saved metrics and renders new figures;
it neither refits models nor recomputes metrics or bootstrap intervals. Rendering
does not require the original fitting environment or machine-specific run path.

Phase-5 fitting and the single Study-2 deployment fit share
[`run`](supplementary.md). That command has 451 fits in total; it is not
an isolated phase-5 command. Saved outputs can be inspected without fitting models.

## Results

After `organize`, use `experiments/studies/benchmark/phases-1-4/` and
`experiments/studies/benchmark/learning/`. These links point to the verified original
reports without moving or renaming frozen inputs and outputs.
