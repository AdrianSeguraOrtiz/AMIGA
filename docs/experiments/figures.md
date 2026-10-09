# Current phase figures

`amiga-exp report figures` renders the four phase designs from completed
selection and evaluation results. It fits no models and does not change saved
predictions, selections, contracts or summaries. This command is a source-tree
addition after the `amiga-exp-v0.2.0` release; that tag is unchanged.

## Generate

From the repository root, with the experiment dependencies installed:

```bash
scripts/experiments/amiga-exp report figures \
  --selection-summary experiments/sequential-selection/summaries/full-001 \
  --selection-run experiments/sequential-selection/runs/full-001 \
  --top5-summary experiments/top5-classification/full-001/selection-summary \
  --top5-run experiments/top5-classification/full-001/selection-run \
  --top10-summary experiments/top10-classification/full-001/selection-summary \
  --top10-run experiments/top10-classification/full-001/selection-run \
  --outer-summary experiments/top10-classification/full-001/outer-summary \
  --layout separate \
  --feature-count 20 \
  --output experiments/reports/figures-002
```

All input paths shown are defaults. The shorter equivalent is:

```bash
scripts/experiments/amiga-exp report figures --output experiments/reports/figures-002
```

The destination must be new and outside the input directories. Use a different
destination for subsequent renders. The command verifies complete summary
manifests and all their artifact hashes. Training SHAP files are additionally
bound to the exact result hashes recorded by the selection summaries. Missing
folds, labels, grid configurations, feature budgets, predictor inclusion rates
or paired final metrics stop generation. Input hashes are checked again before
publishing the output directory.

The reporting adapter reuses the original plotting palette, density clouds and
hyperparameter marginal helpers. It leaves the earlier plotting module and
all frozen fitting dependencies intact, preserving the identities of completed
runs. No old inference is copied onto the new selection diagnostics.

## Main figures

Each case produces four vector PDFs and PNG previews. Prefixes are `bio` and
`mogeneci`.

| Filename after the prefix | Content |
| --- | --- |
| `_phase01_model_screening` | Three ranker families × eight relevance labels, mean inner Regret@5 and original selection frequency out of five outer training complements |
| `_phase02_hyperparameter_tuning` | All 99 family-specific configurations: mean inner Regret@5 versus mean within-complement SD; descriptive candidate-rank coloring, original selected configurations and parameter marginals |
| `_phase03_feature_selection` | Recursive feature-budget curves, original budget-selection counts and the highest/lowest relative training SHAP columns with their inclusion frequencies |
| `_phase04_decision_baselines` | Side-by-side supervised and objective-selector panels; Regret@5 mean ranks, raw means, Friedman and control-based Holm p-values |

### Selection diagnostics: phases 1–3

Each cell or configuration averages its diagnostics equally over the five outer
training complements. These complements overlap: their averages describe the
selection process, not independent replication or held-out performance. Labels
selected in phase 1 can differ across complements; phase 2 retains that fact in
the accompanying `label_modes` column. Configuration IDs are aggregated only
when their family and parameter values agree.

The heatmap counts and tuning rings record the original within-family choices.
They do not select a new global configuration from pooled diagnostics. The
phase-2 color is the average of descriptive configuration ranks computed within
each complement and formulation; it carries no significance coding. Model
selection continues to use the recorded inner Regret@5 rule.

For phase 3, lighter curves show each complement and thick curves show their
mean. Individual curves use a 1.5-point stroke and 45% opacity. The small table
records how many of the five complements selected each budget within each
family; its rows each sum to five. Centered mean absolute
SHAP is computed on training data at the full feature budget, normalized to sum
to one within each inner fit, then averaged across 15 fits per family. All-zero
fits contribute zeros. Inclusion frequencies describe the columns retained by
the originally selected budget in those same 15 fits. These summaries describe
predictive attribution and selection stability, not causal effects.

The main matrix shows the ten highest and ten lowest columns ordered by average
relative training SHAP across the three families, separated by a dashed line.
The same rows appear in the SHAP and inclusion matrices. Relative importance
retains its full-model denominator; the displayed subset is not renormalized.
`--feature-count` changes the total displayed count, split between the two
extremes (the higher-importance group gets the extra column for odd counts).
`--top-features` remains an alias. Neither option changes fitting or selection.
Low SHAP does not establish that removal is harmless; the budget curves assess
the performance of the recursively reduced sets. Complete column matrices and phase-2
and phase-3 figures for the four supervised alternatives are supplementary.
Phase 3 uses the filename `feature_selection`, replacing the earlier
block-ablation design.

Use one full-text-width figure per case for phases 3 and 4. Phase 4 places its
two comparison blocks side by side and plots only Regret@5. Hit@1 and Hit@5
remain in the saved CSVs for a concise textual account.

### Final comparisons: phase 4

`--layout separate` is the default. Panel A contains AMIGA, direct AUPR
regression and top-5%, top-10% and top-20% classification. Panel B contains
AMIGA, the objective-only selectors and the uniform-random reference. The
objective set is case-specific: the complete benchmark produces 16 methods in
BIO-INSIGHT and 13 in MO-GENECI, including AMIGA. The oracle and normalized
regression remain in their original source summaries and are omitted from this
presentation. `--layout joint` is available for an explicitly different analysis
that pools deployable methods into one comparison family.

Ranks are computed separately within each panel and each topology. Use the 87
equally weighted topology means after averaging seeds within fronts and
conditions within topologies. Average tied ranks after rounding values to 12
decimals. Minimize Regret; its smaller ranks are better. Raw mean Regret@5,
displayed as `μ`, is computed before rounding. Each panel is ordered by its own
Regret@5 mean rank. Rank
scales from different panels must not be compared directly.

Only Regret@5 receives hypothesis tests: tie-corrected SciPy Friedman followed
by two-sided fixed-control rank contrasts and Holm adjustment over that panel's
`k-1` alternatives. AMIGA remains the control even when another method has a
better rank. The standard error is `sqrt(k * (k + 1) / (6 * 87))`. Orange
indicates both omnibus and adjusted p-values below 0.05; blue indicates no
rejection. The [statistical reporting specification](reporting.md) gives the
same five-method computation for panel A. The CSV also preserves descriptive
Hit@1 and Hit@5 ranks, maximizing Hit when assigning ranks, together with raw
hit rates on the [0,1] scale. Multiply those raw rates by 100 for percentages in
text. These secondary metrics are not plotted or tested. These
analyses remain exploratory on reused benchmarks; a first mean rank does not
establish superiority.

Uniform random uses exact expected metrics under tied-score random selection.
Its expected Hit probability is positive on every front, whereas a
deterministic selector often has zero Hit on individual fronts. Consequently,
random can have a favorable Hit rank despite a much smaller mean Hit rate.
Interpret saved Hit ranks together with the raw rates; ranks alone do not
measure effect size.

## Outputs and provenance

The complete two-case render produces 34 PDFs and 34 PNGs: eight main figures,
16 tuning/feature figures for the other formulations, and ten complete column
matrices. Supplementary files are under `supplementary/<case>/<arm>/`.

CSV files preserve `selection_candidates`, `screening`, `tuning`,
`feature_curves`, `relative_training_shap`, `feature_stability`,
`comparison_ranks` and `friedman_omnibus`. The final two tables support the
figure directly; they need not be duplicated as manuscript tables.

`manifest.json` records all input hashes, code hashes, the reused style source,
library versions, averaging and tie policies, comparison scope and every
output hash. Results and figures belong under the Git-ignored `experiments/`
directory. Existing learning-curve and real-world figures are separate blocks
and are not regenerated by this command.
