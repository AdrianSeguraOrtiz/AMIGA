# Benchmark design

AMIGA learns to prioritize candidate consensus GRNs within a Pareto front using
historical benchmarks with known external quality. Deployment requires candidate
predictors, not the new problem's gold-standard network. The evaluation separates
learning this post-Pareto decision from generating the candidate fronts.

The [workflow guide](../experiments.md) describes commands and output locations.
Detailed frozen specifications govern execution; this document describes the
current presentation and the distinction between selection and evaluation.

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

The top-5% and top-10% arms were added as exploratory sensitivity analyses. Their
full searches reuse comparator predictions from completed runs. Complete frozen
artifacts retain any additional methods originally evaluated; selecting a
presentation subset does not erase or redefine those runs.

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
in the [grouped specification](grouped-validation.md).

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

This reporting-stage analysis is additional to the original frozen paired
Wilcoxon-Holm comparisons; those retain their original method set and joint
correction. Adding or removing methods changes mean ranks and multiplicity, so
tables from different method sets must not be interpreted as performance changes.
Use the [audited reporting command](reporting.md) to regenerate this presentation
from completed topology summaries.
Reused benchmarks and overlapping training sets limit inference. A first mean
rank alone does not establish superiority, and nonsignificance does not establish
equivalence or noninferiority.

## Interpretation and remaining blocks

Selection figures describe development decisions, not unbiased comparisons of
the winning configurations. Column inclusion and SHAP are predictive descriptions,
not evidence of causal importance or indispensability of correlated variables.

Learning curves must identify their model and tuning scope. A claim about having
only a given number of labelled topologies requires restricting both fitting and
selection to those labels. Earlier fixed-model learning curves and family
exclusions cannot silently become evidence for the current selected procedure.

Deployment selection and refitting use all available benchmark fronts, without
choosing a deployment configuration from favorable outer-test outcomes. Re-score
the existing TCGA-BRCA front and update the support of its selected networks.
External regulatory resources provide contextual support rather than benchmark
AUPR or biological validation of the inferred network.
