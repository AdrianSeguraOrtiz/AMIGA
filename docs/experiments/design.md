# Experimental design: two studies and prediction costs

AMIGA learns to prioritize candidate consensus GRNs within a Pareto front using
historical benchmarks with known external quality. Deployment requires candidate
predictors, not the new problem's gold-standard network. The evaluation separates
learning this post-Pareto decision from generating the candidate fronts.

The [workflow guide](../experiments.md) describes commands and output locations.
Detailed frozen specifications govern execution; this document describes the
current presentation and the distinction between selection and evaluation.

## Study organization

Study 1 evaluates labelled benchmarks in five phases: (1) relevance labels,
(2) hyperparameters, (3) columns and model family, (4) held-out selector comparison,
and (5) learning curves with fixed recipes. Grouping and resource calibration
precede those phases. Study 2 applies the updated model to the existing real
TCGA-BRCA front. Prepared-front prediction costs form a separate section.
The [Study 1](benchmark.md), [Study 2](real.md) and
[prediction-cost](prediction-costs.md) guides describe commands and outputs.

## Data and evaluation unit

BIO-INSIGHT and MO-GENECI each provide 104 fronts from the same collection of
87 topology groups. Analyze the generators separately. Their problems are shared
and must not be treated as 208 independent biological networks.

Topology groups identify directed nonzero reference edges under the recorded node
labels, ignoring sign and magnitude. They prevent exact topology overlap across
training and evaluation partitions. They do not establish independence of related
subnetworks or identify all graph isomorphisms. Gold-standard information used for
grouping is excluded from predictors.

Use five outer folds and three inner folds within each outer training complement.
`front_id` remains the ranking query; topology identity controls partitioning,
training weights and aggregation. All conditions of a topology stay together.
These reused benchmarks provide internal evaluation, rather than untouched
external confirmation.

## Model selection and fitting

All selection decisions use training data and inner-validation metrics. Training
seed 1101 is fixed during selection.

1. Compare relevance labels separately for LightGBM, XGBoost and CatBoost.
2. Search the complete original grids: 27 LightGBM configurations, 36 XGBoost
   configurations and 36 CatBoost configurations per formulation.
3. Compare all columns with recursively retained fractions of 75%, 50% and 25%.
   Elimination uses centered native TreeSHAP contributions from training data;
   inner validation selects the fraction and family.
4. Freeze the selected procedures, relearn reduced masks on each complete outer
   training complement, and fit final models with seeds 1201–1205.

Inner selection minimizes topology-mean Regret@5, then Regret@1, with the recorded
numerical and stable-ID tie rules. Phase 3 additionally prefers fewer columns on
a metric tie. Presenting final mean ranks does not change this selection rule.

Reference fits request 2,000 iterations; tuning, column selection and final fits
request 3,000. There is no validation-driven early stopping. Record actual
iterations when a library exhausts admissible splits.

## Main comparison

| Method | Training target | Deployment score |
| --- | --- | --- |
| AMIGA | Selected within-front relevance labels | Selected ranker's score |
| Direct AUPR regression | Original AUPR | Regression prediction |
| Classification top-5% | High-quality membership within each training front | Positive-class probability |
| Classification top-10% | High-quality membership within each training front | Positive-class probability |
| Classification top-20% | High-quality membership within each training front | Positive-class probability |

Each formulation has its own training-only parameter, family and column selection.
Classification does not search ranking relevance labels. For fraction `q`, let
`k = ceil(q * n)` and use the kth highest training AUPR as the threshold, including
all boundary ties. If the threshold equals the minimum on an informative front,
use candidates strictly above the minimum. Exclude flat training fronts and
retain them in evaluation. Probability order determines recommendations; there is
no fixed 0.5 probability cutoff.

The top-5% and top-10% arms were added as exploratory sensitivity analyses. Their searches use the same grids and selection policy as top-20% classification.

Objective-only selectors are a separate comparison on identical candidates:
individual objectives, objective-rank aggregation, weighted sum, ideal-point
rules, TOPSIS, VIKOR, Tchebycheff variants and trade-off-worthiness knee ranking.
Include the exact uniform-random reference. The AUPR oracle is a nondeployable
ceiling and is excluded from inferential comparisons.

## Metrics and statistics

Regret@5 is primary; Regret@1, Hit@1 and Hit@5 provide secondary context. Regret@k
is the difference between the front's maximum AUPR and the expected best AUPR
among the first k recommendations. Score ties use the exact expectation over
uniform within-tie ordering. Hit@k follows the quality-optimum tolerance recorded
in the partition contract (`rtol=1e-5`, `atol=1e-8`).

Average seed-specific metrics within each front, then conditions within each
topology, then give the 87 topologies equal weight. Do not average predicted
scores into an ensemble, select a seed, or treat seeds as independent problems.
Retain a 104-front macro summary as sensitivity context.

The selected five-method presentation ranks each topology's metric values,
assigning average ranks to numerical ties after rounding to 12 decimal places.
Lower ranks denote better results, including for metrics whose raw values are
maximized. Report raw metric summaries as well as mean ranks so that the size of
the observed differences remains visible.

For Regret@5, report the tie-corrected asymptotic Friedman omnibus test separately
for each generator. Exploratory posthoc comparisons use AMIGA as the control,
the normal approximation to mean-rank differences, and Holm adjustment over the
four control comparisons within each generator. Retain adjustment jointly over
all eight comparisons as additional sensitivity. Secondary ranks are descriptive
and do not introduce a new significance-test family.

Adding or removing methods changes mean ranks and multiplicity, so tables from
different method sets must not be interpreted as performance changes.
Use the [audited reporting command](reporting.md) to regenerate this presentation
from completed topology summaries.
Reused benchmarks and overlapping training sets limit inference. A first mean
rank alone does not establish superiority, and nonsignificance does not establish
equivalence or noninferiority.

## Phase 5: conditional learning curves

Reuse the exact family, labels, parameters and predictor columns fixed for each
outer fold, refitting AMIGA with 10/20/40 labelled topology groups. Three nested
subsets and five seeds retain the original test partitions. The full-size point
reuses the phase-4 predictions. The selected recipe came from the full training
complement, so this is conditional model sensitivity to training size; it does
not estimate repeating model selection under label scarcity. There is no new
supervised comparison in phase 5.

## Study 2 and prediction costs

Select the deployment recipe from stored benchmark inner-validation Regret@5,
never from outer-test outcomes or real-case support, and fit it on all available
benchmark fronts. Use the original real front, five selectors and fixed evidence
snapshot. Regulatory support is contextual evidence, not biological accuracy.

Time prediction and sorting with a saved model on a prepared front. This excludes
model loading, feature preparation, training and upstream inference. Detailed
measurements and repeated scoring belong to the separate prediction-cost section.

## Interpretation

Selection figures describe development decisions, not unbiased comparisons of
the winning configurations. Column inclusion and SHAP are predictive descriptions,
not evidence of causal importance or indispensability of correlated variables.
Five outer complements are overlapping training sets, not five independent
biological experiments. Earlier family-exclusion experiments describe their own
configuration and are not results of the current selected procedure.
