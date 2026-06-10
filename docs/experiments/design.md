# AMIGA Experimental Design

This document defines the experimental protocol we want to implement for the
AMIGA paper. It is intentionally written before rebuilding the automation
scripts, so the scientific design drives the pipeline instead of the other way
around.

The current experimental scripts are considered prototypes. They are useful for
understanding the intended workflow, but the new implementation does not need
to preserve their structure or interfaces.

## 1. Main Scientific Goal

AMIGA is designed as a decision-support system for consensus GRN optimization.
Its primary purpose is not to reproduce the full ordering of a Pareto front, but
to help users identify a small set of highly promising candidate networks.

The central claim to evaluate is:

> AMIGA improves the identification of top-quality consensus networks within
> Pareto fronts produced by evolutionary consensus algorithms.

This framing has two consequences:

- top-of-ranking quality must drive model selection and evaluation;
- global ranking agreement is useful as diagnostic context, but should not be
  the main success criterion.

## 2. Case Studies

The paper will treat `MO-GENECI` and `BIO-INSIGHT` as two independent case
studies.

They should not be collapsed into a single benchmark for the main results.
Instead, the same protocol should be run independently for each case:

- Case study 1: `MO-GENECI`
- Case study 2: `BIO-INSIGHT`

The comparison across both cases should be used to discuss consistency and
robustness of AMIGA's behavior, not to hide differences between the two
upstream algorithms.

Cross-case transfer will not be included in the main experimental design.
`MO-GENECI` and `BIO-INSIGHT` optimize different objective sets, so their
tabular datasets can contain different columns. Training on one case and
testing on the other would require an additional feature-alignment design that
does not match the main paper question.

## 3. Experimental Unit

The natural experimental unit is the Pareto front, represented by `front_id`.

All training, validation, model selection, and statistical comparison must avoid
mixing individuals from the same front across train and validation/test sets.

Therefore:

- all splits must be group-aware by `front_id`;
- all ranking metrics must be computed within each front first;
- aggregate performance should be computed across fronts, not across individual
  rows globally.

## 4. Metric Hierarchy

### 4.1 Primary Metric

The primary metric should measure AMIGA's ability to place high-quality
networks near the top of the ranking.

Final decision:

```text
Primary metric: Regret@5
```

Implementation source:

```text
amiga.selection.learn2rank.compute_ranking_metrics_by_front
amiga.selection.learn2rank.compute_ranking_metrics
```

Definitions are computed inside each front first. Let:

```text
y_i = true AUPR of candidate i
s_i = predicted AMIGA score of candidate i
TopK_s = k candidates with highest predicted score s_i
Best(front) = max_i y_i
Best@k(front) = E[max_{i in TopK_s} y_i]
```

Then:

```text
BestAUPR@k = E[Best@k(front)]
Regret@k = Best(front) - E[Best@k(front)]
Hit@k = P(any candidate with y_i = Best(front) appears in TopK_s)
```

Aggregated metrics are the arithmetic mean across fronts, not across rows.
Predicted-score ties are evaluated fairly: if the cutoff falls inside a tied
score block, every ordering of the tied candidates is treated uniformly. Metrics
must never depend on the row order of the input CSV. Exported ranked CSVs may use
a deterministic identifier-based tie-breaker only for display/reproducibility.

Rationale:

- `Regret@k` directly measures the expected quality loss when inspecting only
  the top `k` predicted individuals.
- `k=5` is more stable than `k=1` and matches a realistic decision-support
  scenario where a user may inspect a small shortlist rather than a single
  candidate.
- The exact value of `k` is fixed as part of this design before rebuilding the
  final experimental pipeline.

### 4.2 Secondary Top-k Metrics

Secondary metrics should help interpret the primary result:

```text
Regret@1
Regret@3
Hit@5
BestAUPR@5
NDCG@5
```

Possible additions:

```text
Hit@1
Hit@3
BestAUPR@1
BestAUPR@3
NDCG@3
NDCG@10
```

These metrics should not override the primary metric unless the protocol is
explicitly revised before final evaluation.

### 4.3 Diagnostic Metrics

Global rank-correlation metrics should be reported as diagnostics:

```text
Spearman
KendallTau
```

They are useful for understanding whether AMIGA captures the broad shape of the
front, but they should not be treated as primary evidence of success. A model
that ranks the lower tail poorly can still be valuable if it consistently
identifies the best networks at the top.

## 5. Model-Selection Protocol

The old workflow selected winners by combining or ranking several metrics. That
approach is only defensible if the aggregation rule is fixed and justified in
advance.

For the new workflow, model selection should be simpler and aligned with the
scientific goal.

Final decision:

1. Select the best configuration by paired average rank on `Regret@5` across
   validation fronts.
2. Report Friedman and Holm post-hoc evidence for that ranking, but do not use
   `p < 0.05` as a gate for advancing a configuration.
3. Use mean `Regret@5` and secondary metrics only as predefined tie-breakers.
4. Record the full metric and statistical tables for transparency.

Possible tie-breaker order:

```text
1. lower paired average rank for Regret@5
2. lower mean Regret@5
3. higher Hit@5
4. lower Regret@1
5. higher BestAUPR@5
6. simpler or more stable model family, if still tied
```

Hyperparameter tuning must follow the same rule. It should optimize the primary
ranking statistic first and use the tie-breaker order only when configurations
are tied on paired average rank.

No weighted average of heterogeneous metrics should be used to select the final
configuration unless the paper explicitly introduces and justifies that score
before seeing the final results.

## 6. Control Label Modes

The label modes `reversed` and `shuffled` are useful sanity checks and compete
in phase 01 exactly like the other label modes.

Their purpose is to demonstrate that performance degrades when the training
signal is inverted or destroyed. If one of them ranks unexpectedly well, that
is not filtered out by the pipeline; it must be interpreted and discussed as an
experimental finding.

The paper can describe them as control label modes, but the implementation does
not mark them as non-selectable candidates.

## 7. Splitting Strategy

Within each case study, we will use a held-out development/test design by
front.

Final split contract:

```text
Split unit: front_id, with front_name as the durable external key
Development/test ratio: 80/20 by front
Current concrete split size: 83 development fronts and 21 held-out test fronts
Seed: 42
Shared assignment across cases: yes, when front_name is available in both cases
Target column used for assignment: no
```

The same `front_name` should receive the same development/test assignment in
`MO-GENECI` and `BIO-INSIGHT` whenever both case studies contain the same front.
This keeps the two case studies independent while making their final results
easier to compare.

The split must be created before model screening, hyperparameter tuning,
ablation, baseline comparison, or final test evaluation.

### 7.1 Development/Test Split By Front

For each case study:

1. split fronts into development and held-out test sets;
2. use only development fronts for model/label/hyperparameter selection;
3. freeze the selected AMIGA configuration;
4. evaluate once on held-out test fronts.

Within the development set, model and hyperparameter selection can still use
GroupKFold by `front_id`. The held-out test fronts must remain untouched until
the final evaluation.

This design is easier to explain in the paper than nested cross-validation and
provides a clean final estimate for each case study.

### 7.2 Front Stratification

Front stratification means that the development/test split should preserve, as
far as possible, the diversity of fronts in each case study.

For example, the split should avoid placing all large networks, all DREAM
networks, or all fronts from a specific source family in only one side of the
split.

The split is deterministic and uses only metadata available before model
training. It must not use `AUPR` or any other target-derived value to decide
which fronts go into test.

The selected strategy is:

1. Use `family` as the quota level.
2. Compute the global number of test fronts as `round(n_fronts * 0.20)`.
3. Allocate family-level test quotas proportionally to the number of fronts.
4. For every family with more than one front, require at least one development
   front and at least one test front when the global quota allows it.
5. Assign singleton families to development, because they cannot be split while
   preserving representation on both sides.
6. Adjust quota rounding with a largest-remainder rule until the global test
   count is reached.
7. Within each family, choose the test fronts with a deterministic
   metadata-diversity rule over network size, organism or benchmark source when
   available, and experimental condition family when available.

The within-family selector favors test fronts that add new values of `size`,
`condition`, and `source`; remaining ties are resolved by a stable hash of
`front_name` and the fixed seed.

If reliable metadata are not available for a variable, the pipeline must not
invent it. Missing metadata are treated as `unknown`, and the split audit table
must make this visible.

The split strategy must be identical in spirit for `MO-GENECI` and
`BIO-INSIGHT`, even if the available metadata differ.

The frozen split artifacts are:

```text
docs/experiments/splits/MO-GENECI_split_manifest.json
docs/experiments/splits/MO-GENECI_split_assignments.csv
docs/experiments/splits/BIO-INSIGHT_split_manifest.json
docs/experiments/splits/BIO-INSIGHT_split_assignments.csv
```

The implemented pipeline reads these files instead of creating new random
splits implicitly.

## 8. Data And Feature Contract

Every case study must have a versioned JSON config describing its data contract
before any model screening is run.

Current configs:

```text
docs/experiments/cases/MO-GENECI.json
docs/experiments/cases/BIO-INSIGHT.json
```

The configs define:

- the training CSV;
- the frozen split manifest;
- the target column;
- control columns;
- optimization objective columns;
- objective directions;
- feature blocks;
- ablation feature sets;
- explicitly excluded feature columns.

Final target/control decision:

```text
Target column: AUPR
Group column: front_id
Item column: item_id
```

Objective columns:

```text
MO-GENECI: quality, degreedistribution, motifs
BIO-INSIGHT: quality, degreedistribution, motifs,
             reducenonessentialsinteractions, dynamicity, metricdistribution
```

All evolutionary-algorithm objectives are minimization objectives. This is
important for single-objective and objective-aggregation baselines.

Feature blocks:

```text
objectives
technique_weights
expression
network
```

The following expression columns are constant zero columns in both current
case-study datasets and are excluded explicitly:

```text
expr_cond_missing_frac_mean
expr_gene_missing_frac_mean
expr_prop_missing
```

The validated contract artifacts are:

```text
docs/experiments/contracts/MO-GENECI_case_manifest.json
docs/experiments/contracts/MO-GENECI_data_manifest.json
docs/experiments/contracts/MO-GENECI_feature_columns.json
docs/experiments/contracts/MO-GENECI_validation_report.json
docs/experiments/contracts/BIO-INSIGHT_case_manifest.json
docs/experiments/contracts/BIO-INSIGHT_data_manifest.json
docs/experiments/contracts/BIO-INSIGHT_feature_columns.json
docs/experiments/contracts/BIO-INSIGHT_validation_report.json
```

The implemented pipeline reads these case configs and feature-column manifests
instead of re-inferring columns implicitly.

## 9. Decision Baselines

AMIGA should be compared against meaningful non-learned post-Pareto decision
rules. These baselines do not learn from AUPR labels; they select or rank
solutions using only the objective values available on the held-out Pareto
fronts. This makes phase 4 a direct test of whether AMIGA adds value beyond
standard objective-based decision rules.

Mandatory baseline battery:

- one single-objective ranking baseline per optimization objective, to test
  whether any single objective is enough to identify high-AUPR networks;
- mean rank over objectives, as a scale-robust rank-aggregation rule;
- normalized equal-weight objective aggregation, as a transparent weighted-sum
  / WSM rule over per-front objective goodness;
- ideal-L2 distance, as a smooth proximity-to-ideal compromise rule;
- TOPSIS, as a standard ideal-vs-anti-ideal relative-closeness rule;
- VIKOR with `v=0.5`, as a standard compromise-ranking rule balancing group
  utility and worst-objective regret;
- augmented Tchebycheff, as an achievement-scalarizing worst-objective rule
  with a small sum term for deterministic tie-breaking;
- simple Tchebycheff, kept as a transparent worst-objective control and to
  clarify what the augmented variant adds.

The objective direction for every objective must be declared explicitly in the
case-study configuration. If all objectives are minimization objectives, this
can be recorded once. If not, each objective needs its own direction.

Potential additional baselines:

- simple linear model or random forest over the same features;
- objective-only learning-to-rank;
- technique-weight-only heuristic.

These additional baselines are optional and should not replace the non-learned
post-Pareto baseline battery above.

Baseline definitions must be frozen before final evaluation.

## 10. Ablation Protocol

Ablation should answer:

> Which information blocks contribute to AMIGA's top-k selection ability?

Candidate feature blocks:

- technique weights;
- objective optimization levels;
- expression-derived features;
- consensus-network features.

Possible finer sub-blocks:

- expression global/statistical features;
- expression correlation/PCA/time-series features;
- network global features;
- network strength features;
- network community/path/reciprocity/entropy features.

Important rule:

The ablation study should use a frozen modeling protocol. It should not freely
select a different best model for every ablation variant unless that is stated
as a different experimental question.

The previous observation that an expression-only variant appeared to outperform
richer feature sets should be treated as a warning sign until audited.

Specific checks for the new ablation implementation:

- save the exact columns used by each variant;
- save the exact columns dropped by each variant;
- save feature counts per block;
- save the split assignment by `front_id`;
- verify no control columns or target-derived columns remain as features;
- verify expression-only features cannot accidentally encode leakage from the
  target or evaluation process.

## 11. Statistical Testing

Statistical tests should be paired by front.

The recommended unit for comparison is the per-front value of the primary
metric, e.g. `Regret@5`.

Candidate tests:

- Wilcoxon signed-rank test for AMIGA vs one baseline;
- Friedman test with Holm correction for multiple methods;
- confidence intervals over fronts via bootstrap, if useful for reporting.

The paper should avoid relying only on aggregate means without paired
uncertainty or significance analysis.

## 11.1. Phase-01 Reference Hyperparameters

Phase 01 is a screening stage, not a hyperparameter optimization stage. Each
model family therefore uses one fixed reference configuration while comparing
label modes. The values are centered on the phase-02 grid where possible, with
moderate shrinkage and subsampling for tree-based rankers.

```text
LGBMRanker:
  num_leaves        = 63
  min_child_samples = 50
  learning_rate     = 0.05
  fixed             = n_estimators=2000, subsample=0.8, colsample_bytree=0.8

XGBRanker:
  max_depth         = 6
  min_child_weight  = 5
  learning_rate     = 0.05
  fixed             = n_estimators=2000, subsample=0.8, colsample_bytree=0.8

CatBoostRanker:
  depth             = 6
  l2_leaf_reg       = 5
  learning_rate     = 0.05
  fixed             = iterations=2000, allow_writing_files=False
```

These reference settings are recorded in each phase-01 `run_manifest.json`.
Phase 02 then performs the actual hyperparameter comparison on the shortlisted
label formulation for each model family.

## 11.2. Hyperparameter Tuning Grid

The default phase-02 grid is pre-specified before held-out evaluation and is
applied independently to the best label mode selected for each model family in
phase 01. It is intentionally compact but covers the main tree-ranker controls:
model capacity, regularization and shrinkage.

```text
LGBMRanker:
  num_leaves        = {31, 63, 127}
  min_child_samples = {30, 50, 100}
  learning_rate     = {0.03, 0.05, 0.10}
  fixed             = n_estimators=3000, subsample=0.8, colsample_bytree=0.8

XGBRanker:
  max_depth         = {4, 6, 8}
  subsample         = {0.8, 1.0}
  min_child_weight  = {1, 5, 10}
  learning_rate     = {0.03, 0.05}
  fixed             = n_estimators=3000, colsample_bytree=0.8

CatBoostRanker:
  depth             = {4, 6, 8}
  l2_leaf_reg       = {3, 5, 7, 10}
  learning_rate     = {0.03, 0.05, 0.10}
  fixed             = iterations=3000, allow_writing_files=False
```

With one shortlisted label mode per model family, phase 02 evaluates 99
candidate configurations per case study: 27 LGBM, 36 XGB and 36 CatBoost
settings. The large estimator/iteration budget is paired with validation-fold
early stopping where the underlying learner supports it, so the budget acts as
an upper bound rather than a fixed number of effective boosting rounds.

## 12. Reproducibility Contract

Every experimental stage should write a machine-readable manifest.

Each manifest should include:

- case study name;
- input dataset path;
- AMIGA package version;
- Git commit hash;
- random seed;
- split definition;
- primary metric;
- candidate model grid;
- selected configuration;
- feature columns used;
- output paths;
- timestamp.

The implemented pipeline is restartable and avoids silently overwriting
expensive results unless explicitly requested.

Minimum artifact set for reproducibility and review:

```text
case_manifest.json
data_manifest.json
split_manifest.json
candidate_grid.json
run_manifest.json for every trained run
cv_report.json for every validation run
feature_columns.json for every trained run
valid_fold*_ranked.csv for every validation run
selected_config.json
final_test_ranked.csv
final_test_report.json
ablation_manifest.json
baseline_manifest.json
statistical_tests.json
environment_manifest.json
```

The manifests should be sufficient for an external reviewer to answer:

- which fronts were used for development and test;
- which columns were used by each model;
- why the final configuration was selected;
- which exact baselines and ablations were run;
- which code version produced the reported results.

## 13. Implemented Pipeline Stages

The experimental workflow exposes one entry point per case study.

Example interface:

```bash
scripts/experiments/amiga-exp run-all <mogeneci_case_dir>
scripts/experiments/amiga-exp run-all <bio_case_dir>
```

The pipeline executes:

```text
0. validate/build data
1. create or validate the development/test split
2. run model and label screening on development fronts
3. summarize screening results
4. select the best label formulation within each model family using paired
   average rank on Regret@5
5. run hyperparameter tuning on development fronts
6. summarize tuning results
7. freeze final AMIGA configuration using paired average rank on Regret@5
   and tie-breakers
8. run final evaluation on held-out test fronts
9. run ablation study using the frozen protocol
10. run decision baselines on the same held-out test fronts
11. run statistical tests
12. generate paper-ready tables and figures
```

All model-selection decisions must happen before step 8. The held-out test set
is only for final evaluation and paper-facing comparisons.

## 14. Reporting And Figure Policy

Primary reporting:

- every experimental phase emits `summary/primary_rank_table.csv`;
- the primary statistic is paired average rank on `Regret@5` over common
  fronts;
- in `01_model_screening`, paired ranks and Holm-adjusted p-values are computed
  independently within each `model_type`;
- lower `avg_rank` and lower `mean_regret5` are better;
- Holm-adjusted p-values are reported as statistical evidence, but p < 0.05 is
  not required to advance a configuration;
- phase winners and selected configurations are determined before held-out
  evaluation and are not changed by plotting or supplementary analysis.

Primary paper figures:

1. `01_model_screening`: model/label-mode heatmap of `avg_rank` on `Regret@5`,
   with a robust single-hue color scale, control label modes visually separated
   at the right, and the best label formulation per model family marked.
2. `02_hyperparameter_tuning`: mean-vs-variability Regret@5 scatter plot of
   tuned configurations, with color encoding paired average rank, a colorbar
   transition marking the first Holm-adjusted `p < 0.05` rank, and the frozen
   AMIGA configuration marked by a distinctive point border.
3. `03_ablation`: feature-block matrix showing which blocks are active in each
   ablation variant, with paired `Regret@5` average rank and Holm-adjusted
   p-value shown separately for development CV and held-out test evidence.
4. `04_decision_baselines`: rank plot comparing frozen AMIGA against
   non-learned decision baselines on held-out test fronts.

Supplementary reporting:

- optional Top-K context plots use `Regret@k`, `Hit@k`, and `BestAUPR@k`;
- they are generated only with `amiga-exp plot-phase --include-secondary` or
  `amiga-exp plot-all --include-secondary`;
- they live under `plots/supplementary/`;
- they are diagnostic context only and must not affect model selection,
  ablation interpretation, or decision-baseline conclusions.

Caption-ready descriptions:

- **Model Screening:** Model and label-mode screening on development fronts.
  Cells show paired average rank for `Regret@5` with Holm-adjusted p-values
  underneath; lower rank indicates better recovery of top-performing networks,
  cell color uses a robust single-hue rank scale, p-value colors distinguish
  `p < 0.05` from `p >= 0.05`, and stars mark the best label formulation within
  each model family advanced to hyperparameter tuning.
- **Hyperparameter Tuning:** Hyperparameter tuning of shortlisted AMIGA
  configurations on development fronts. Points place each configuration by mean
  `Regret@5` and its standard deviation; point color encodes paired average
  rank on `Regret@5`, with a colorbar transition marking the first
  Holm-adjusted `p < 0.05` rank. KDE density blobs group configurations sharing
  the same model family and label mode, the in-grid blob-color legend
  identifies each group, and the selected configuration is marked with a
  distinctive point border before held-out evaluation. Point labels show
  ultra-compact `p=value` hyperparameter settings; the caption should expand the
  model-specific abbreviations, and original parameter tags are retained in the
  plotted CSV.
- **Ablation:** Feature-set ablation under the frozen AMIGA protocol. Values
  show feature-set variants ordered by development paired average rank, which
  feature blocks are active in each variant, and Holm-adjusted p-values against
  the best average-rank configuration separately for development CV and
  held-out test evidence.
- **Decision Baselines:** Held-out comparison between frozen AMIGA and
  non-learned post-Pareto decision rules. Methods include single-objective
  selection, rank aggregation, WSM-style aggregation, ideal-L2, TOPSIS, VIKOR,
  augmented Tchebycheff, and simple Tchebycheff. They are compared by paired
  average rank for `Regret@5` on the same test fronts; lower ranks are better.

## 15. Resolved Decisions

The following design decisions are fixed for the paper experiments:

1. Primary metric: `Regret@5`.
2. Final evaluation: held-out development/test split by `front_id`, with
   `83` development fronts and `21` held-out test fronts per case for the
   current 104-front datasets.
3. Front stratification: preserve available source/size/family diversity where
   metadata allow it; use `seed=42`; assign singleton families to development;
   write split manifests and audit artifacts.
4. Mandatory baselines: single-objective, objective mean-rank, normalized
   objective aggregation / WSM, ideal-L2, TOPSIS, VIKOR, augmented
   Tchebycheff, and simple Tchebycheff.
5. Hyperparameter tuning: optimize paired average rank on `Regret@5` first,
   report Friedman/Holm evidence, and then apply the predefined tie-breaker
   order without using p-values as a selection gate.
6. Cross-case transfer: excluded from the main design.
7. Reproducibility: every phase must emit manifests, predictions, metrics,
   selected configurations, split definitions, and environment metadata.
8. Data contract: objective directions are minimization-only for the current
   case studies; constant expression missingness columns are excluded.
9. Reporting: primary figures use `Regret@5` rank summaries; optional Top-K
   figures are supplementary context only.
