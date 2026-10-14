# Figure description

**Five-phase experimental design for AMIGA on labelled benchmark fronts.** BIO-INSIGHT and MO-GENECI each supply 104 candidate fronts over the same 87 topology groups and are analyzed separately. Whole topology groups are kept together in five outer train/test splits, with three grouped inner folds inside each outer training complement. **(1)** Candidate reference quality is converted into alternative within-front relevance labels, screened separately for LightGBM, XGBoost and CatBoost. **(2)** Family-specific hyperparameter grids are evaluated using inner validation. Regression and classification use fixed target definitions and their own parameter searches. **(3)** Predictors are recursively retained at 100%, 75%, 50% and 25% of the original columns using centered training-only TreeSHAP contributions. Inner validation selects the feature fraction and model family. The selection criterion minimizes topology-mean Regret@5, then Regret@1; phase 3 additionally prefers fewer columns on a metric tie. **(4)** Reduced masks are relearned on the complete outer training complement, and selected supervised procedures are refitted with five final training seeds. Their predictions on the same held-out candidates are scored before using held-out AUPR for evaluation. The current presentation compares AMIGA with direct AUPR regression and top-5%, top-10% and top-20% classification. Objective-only selectors and the exact uniform-random reference form a separate panel. The AUPR oracle is a nondeployable ceiling, not an inferential competitor. Metrics are averaged over seeds within fronts and conditions within topology groups, giving equal weight to each topology. **(5)** AMIGA is refitted with 10, 20 and 40 labelled topology groups using three nested subset draws and five seeds, without changing the phase-4 recipe, exact predictor columns or outer test partitions. The full-size reference (69–70 training topologies) reuses phase-4 predictions. This is conditional training-size sensitivity, not reselection under label scarcity. All visual examples are schematic; no measured performance values are plotted. The real-data study and prediction-cost analyses are outside the five phases shown.

## Extended interpretation notes

### Data units

A Pareto front is a set of candidate consensus networks, not an independent biological network. `front_id` defines the learning-to-rank query. Topology identity controls partitioning, training weights and aggregation, but is excluded from predictors. Grouping removes exact overlap in labelled directed nonzero reference topology; it is not a proof of independence among related subnetworks or all graph isomorphisms. The reused benchmarks support internal evaluation rather than untouched external confirmation.

### Labels and fit budgets

Phase 1 includes dense ranks, average ranks, continuous within-front relevance, 5/10/15-bin quantiles, reversed labels and shuffled labels. All eight remain eligible; a control winner must be retained and reported. For LightGBM and XGBoost, continuous relevance is represented as `floor(255 * relevance)` for the library input contract; this is not another tuned parameter. Reference fits request 2,000 iterations, while later tuning, feature-selection and final fits request 3,000. There is no validation-driven early stopping. The grid domain values are preserved in `design_snapshot.json`.

### Feature selection and evaluation

Training-only TreeSHAP samples at most 32 candidates per front (seed 1501), centers contributions within the front and weights topology groups equally. Elimination is recursive with refitting; validation outcomes select the budget but do not order feature elimination. Phase 3 chooses a procedure, not one globally fixed mask. The mask is relearned on each complete outer training complement and shared across its final-seed fits. Selection seed 1101 and final seeds 1201–1205 have distinct roles.

### Metrics and statistical reporting

Regret@k is the maximum true AUPR in the front minus the expected best true AUPR among its first k recommendations, with exact uniform ordering within tied predicted scores. Regret@5 is primary. Regret@1, Hit@1 and Hit@5 provide secondary context. In the five-supervised-method presentation, the primary inferential summary is the tie-corrected asymptotic Friedman test, per generator; exploratory Holm-adjusted control comparisons use the normal approximation to mean-rank differences with AMIGA as control, not all-pairs Wilcoxon tests. The ordinary four control comparisons per generator and an eight-comparison sensitivity adjustment are distinguished in the repository. Raw metrics accompany mean ranks. Neither a first mean rank nor a nonsignificant test demonstrates superiority or equivalence.

### Learning curves

Phase 5 has 90 AMIGA learning jobs and 450 model fits. These are separate from the one Study-2 deployment fit bundled by the execution wrapper (451 combined fits). Average seeds within each subset/front, then subset draws within front, then conditions within topology. Conditional 95% bands use 10,000 topology bootstrap resamples with seed 1401. The recipe was selected using the full outer training complement; smaller labelled subsets do not repeat selection. No supervised comparator search or comparison is added here.

## Source

AMIGA commit `f31d550b2e66d8459f143e0f39eaa20aabc028b9`; source map in `SOURCES.md` and `provenance.json`.
