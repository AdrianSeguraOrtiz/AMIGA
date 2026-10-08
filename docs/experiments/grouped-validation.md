# Grouped validation protocol

This document specifies a controlled evaluation of post-Pareto selection on the BIO-INSIGHT and MO-GENECI benchmarks. Its questions are whether learning improves selection over objective-only rules, how learning to rank compares with regression and classification using the same inputs, and how selection changes with training-set size and feature availability.

**Implementation status:** `amiga-exp grouped` implements contract freezing, the complete benchmark job plan, dependency-aware execution, verified resumption and paired summaries. A separate timing pilot remains available. The schema-1 contract has status `pilot_ready_not_full_eval_frozen`; the schema-2 evaluation contract embeds its partitions and adds the remaining executable policies with status `frozen_evaluation`. Implementation and synthetic tests do not establish completion of the scientific evaluation. Actual completion requires all planned artifacts and a successful summary.

These benchmarks have been reused during method development. The proposed evaluation is **internal nested validation grouped by topology**, not evaluation on an untouched external dataset. Group isolation controls data use within the new procedure; it does not remove the influence of benchmark reuse on earlier design choices.

## Data units and topology isolation

Each generator has 104 Pareto fronts corresponding to the same 104 benchmark problems. There are 87 distinct topology signatures across those problems. The two generators are evaluated separately and share group assignments; their fronts do not constitute 208 independent biological problems.

Topology signatures use node identities and directed nonzero reference edges, ignoring edge sign and magnitude. They identify exact topology agreement under the recorded node labels. They do not rule out related subnetworks or graph isomorphisms after relabeling. AUPR values are not used to construct the groups. Reference-network information used for grouping must not become a predictor or influence consensus generation.

The versioned [topology metadata](groups/topology_groups.json) records front identities,
source-relative paths, file hashes and canonical graph signatures. The source
collection is GENECI's `input_data` directory. When those source files are available,
verify their expression and reference-network hashes and reconstruct each signature:

```bash
python -m scripts.experiments.amiga_exp.grouped_validation.topology \
  --source-root /path/to/GENECI/input_data
```

This command reads the source collection without changing it. The generated
training contract depends on the versioned metadata and the existing benchmark
tables; it does not require the source collection at runtime.

The roles of the identifiers remain separate:

| Identifier | Role |
| --- | --- |
| `topology_id` | Splitting, training-set subsampling, and primary aggregation |
| `front_id` | Ranking query, within-front labels, and within-front evaluation |
| `item_id` | Candidate identity and reproducible export order |

All conditions belonging to one topology remain together in every split and subsample. The ranker still treats each front as a separate query; it does not combine all conditions of one topology into a single query.

The existing input tables are `experiments/BIO-INSIGHT/data/data_104.csv` and `experiments/MO-GENECI/data/data_104.csv`. Predictor definitions are recorded in the corresponding files under `docs/experiments/contracts/`. The full feature sets contain 104 predictors for BIO-INSIGHT and 101 for MO-GENECI, organized into evolutionary objectives, inference-technique weights, expression context, and network descriptors.

Data, feature definitions, source code, and topology mappings must be identified by hashes in generated contracts. Reusing precomputed descriptors requires checking that they do not contain transformations fitted using information outside the relevant training partition. Any transformation learned across fronts must be fitted inside that partition.

## Nested partitions

Five outer folds partition the 87 topologies, yielding 17–18 held-out topologies per fold and 69–70 training topologies. Three inner folds are generated exclusively within each outer training complement. Every front obtains outer out-of-fold predictions from models whose training and configuration selection exclude its topology.

Assignments are deterministic: sort topology IDs, permute them with the specified NumPy random generator seed, split the permutation into nearly equal chunks, and store the explicit topology and front assignments. No outcomes are used to construct the partitions. The two generators use identical assignments.

Outer predictions must not determine predictors, labels, configurations, stopping rules, training duration, or seeds. Changing the training seed does not itself create a new data split.

## Supervised formulations

The central comparison uses four CatBoost formulations with the same training candidates, feature columns, and technical exclusion rules.

| Arm | Estimator and loss | Training target | Score used for selection |
| --- | --- | --- | --- |
| `ltr_catboost` | `CatBoostRanker`, YetiRank | Within-front `rank_dense` labels for BIO-INSIGHT; `rank_avg` labels for MO-GENECI | Ranker score |
| `reg_aupr` | `CatBoostRegressor`, RMSE | Original training AUPR | Regression prediction |
| `reg_normalized` | `CatBoostRegressor`, RMSE | AUPR normalized within each training front | Regression prediction |
| `clf_top20` | `CatBoostClassifier`, Logloss | Membership in the high-quality region of each training front | Positive-class probability |

Larger scores receive higher priority. Identifiers, AUPR, AUROC, other external quality measures, labels, predictions, and target-derived columns are excluded from predictors. The CatBoost comparison tests formulations under a shared backbone; it does not establish that this model family is universally superior to other rankers.

For `reg_normalized`, the target is `(y - min(y)) / (max(y) - min(y))`, calculated separately within each training front. Scoring a new front requires neither its quality labels nor its AUPR extrema. There is no inverse transformation and no clipping of regression predictions, because only their ordering is required.

For `clf_top20`, let `k = ceil(0.20 * n)` and use the AUPR at descending position `k` as the threshold. All candidates tied at that threshold are positive. If the threshold is the minimum AUPR on a nonconstant front, candidates strictly above the minimum are positive. Record the resulting positive fraction. Labels are not split by candidate ID, and recommendations use probability order rather than a probability cutoff of 0.5.

Fronts with exactly constant training AUPR are excluded from all four training formulations because they provide no within-front quality ordering. Detect these fronts using only the current training labels. Retain them in evaluation. Report the number of excluded fronts and rows, and distinguish available labeled topologies from usable training fronts.

For regression and classification, assign row weights proportional to `1 / (m_g * n_f)`, where `m_g` is the number of usable fronts for topology `g` in the current training set and `n_f` is the number of candidates in front `f`. Normalize these weights to mean one over training rows. For YetiRank, use exactly `group_weight = 1 / m_g`, repeated identically for candidates in a front, with no additional normalization or individual sample weights. These choices reduce overrepresentation of topologies with multiple conditions; they do not make the native losses mathematically identical. See the [CatBoost ranking-loss and weight documentation](https://catboost.ai/docs/en/concepts/loss-functions-ranking).

Training preparation retains complete selected fronts, orders candidates by `front_id` and `item_id`, and rejects duplicate candidate identities, nonfinite targets, and infinite predictors. Predictor NaNs are preserved for CatBoost. Label transformations, flat-front exclusions, and weights are computed after restricting the data to the requested training fronts.

## Configuration selection and seeds

All four arms use the same six configurations:

| Parameter | Values |
| --- | --- |
| `depth` | 4, 6, 8 |
| `learning_rate` | 0.03, 0.10 |
| `l2_leaf_reg` | 3 |
| `iterations` | 3000 |
| Early stopping | Disabled |
| `use_best_model` | `False` |

The grid is a bounded comparison, not a claim to locate each formulation's optimum. Every fit uses all 3000 trees. CPU execution, package versions, effective parameters, and thread counts are recorded. The comparison must not silently switch hardware modes between arms.

Within each outer fold, select one configuration per arm using predictions from its three inner validation folds. Minimize mean Regret@5 with equal weight per topology, first averaging conditions within each topology. Round selection statistics to 12 decimal places for numerical ties; then prefer lower topology-mean Regret@1 and finally the stable configuration ID. P-values and outer-fold results are not selection criteria. Rank summaries may be reported as secondary descriptions.

| Randomness source | Fixed seed values |
| --- | --- |
| Outer partition | 20260910 |
| Inner partitions for outer folds 0–4 | 20260911, 20260912, 20260913, 20260914, 20260915 |
| Three-fold deployment partition | 20260916 |
| Configuration-search fits | 1101 |
| Final fits after configuration selection | 1201, 1202, 1203, 1204, 1205 |
| Nested learning-curve subset orders | 1301, 1302, 1303 |
| Paired topology bootstrap | 1401 |

Each selected configuration is refitted on its outer training complement with all five final seeds. Do not retune separately for each final seed or select the best seed. Average metrics across seeds within each front; averaging prediction scores to create an ensemble is a different procedure and is not the primary analysis.

The central comparison requires 720 inner-search fits and 200 final outer fits:

`2 cases × 5 outer folds × 4 arms × 6 configurations × 3 inner folds + 2 × 5 × 4 × 5 final seeds = 920 fits`.

## Objective-only selectors, random selection, and oracle

Retain the existing objective-only selectors, including individual objectives, mean objective rank, weighted sum, ideal-point distance, TOPSIS, VIKOR, and augmented Tchebycheff. Apply each selector to the same candidates available to the learned methods.

Add a uniform-random reference using constant scores and the exact expected metric under uniform ordering of all tied candidates. This avoids selecting or relying on a favorable random permutation.

Add a **trade-off-worthiness ranking**, an explicit full-ranking adaptation of the knee criterion of [Rachmawati and Srinivasan](https://doi.org/10.1109/TEVC.2009.2017515). Orient objectives toward minimization, normalize each nonconstant objective within its front to `[0,1]`, and omit constant objectives. For candidate `i` and alternative `j`, define:

```text
deterioration(i,j) = sum_l max(z[j,l] - z[i,l], 0)
improvement(i,j)   = sum_l max(z[i,l] - z[j,l], 0)
tradeoff(i,j)      = deterioration(i,j) / improvement(i,j)
worthiness(i)     = min_j tradeoff(i,j)
```

Use the complete front as the comparison set and rank worthiness from largest to smallest. Ignore comparisons between identical objective vectors. Alternatives with no positive improvement do not limit the minimum; their ratio is infinite. Record residual dominance and degenerate cases. If the objectives provide no distinguishing signal, retain the score tie rather than inventing a preferred candidate. This formulation works directly with the three- and six-objective cases without projecting the front to two dimensions.

The public [pymoo `HighTradeoffPoints` implementation](https://www.pymoo.org/mcdm/index.html#high-trade-off-points) selects a subset, rather than returning a complete top-k ranking. The scoring adaptation above must therefore be identified and checked with constructed fronts, duplicates, flat objectives, and degenerate cases.

Mean objective rank and uniform Borda aggregation induce the same order when their objective directions and tie treatment agree. Relate the existing mean-rank baseline to [OPSBC](https://doi.org/10.1016/j.eswa.2024.123803), verifying this equivalence instead of counting equivalent methods as independent competitors.

Report the maximum AUPR in a front as an oracle ceiling. The oracle uses evaluation labels, is not deployable, and is excluded from inferential comparisons between methods.

## Metrics, score ties, and aggregation

The primary outcome is **Regret@5**; **Regret@1** is the main secondary outcome for selecting one network. For a front with maximum quality `y_max`, Regret@k is `y_max - BestAUPR@k`, where BestAUPR@k is the best quality among the first `min(k,n)` candidates. With score ties, evaluate its expectation under uniform ordering within exact-equal-score blocks. Hit@k uses the same tie expectation; candidates are considered quality-optimal with `numpy.isclose(AUPR, y_max, rtol=1e-5, atol=1e-8)`.

The expectation over score ties applies equally to learned methods and baselines. Candidate IDs may provide reproducible presentation or export order, but accidental row order must not determine evaluation quality.

Also report BestAUPR@1/@5, Hit@1/@5, and Regret@3/@10 for context. The runner exports all three metric families at k=1,3,5,10. NDCG and rank correlations, if calculated separately, are diagnostic and do not select configurations. Express improvements as absolute paired differences as well as baseline and method values. A percentage reduction in regret is not a percentage increase in biological accuracy.

For the primary aggregate:

1. Average the five final-seed metrics within each front.
2. Average front metrics within each topology.
3. Average over the 87 topologies with equal weight.

Report the macro-average over 104 fronts as a sensitivity analysis. Do not give unequal-sized folds equal weight by simply averaging their means. Keep topology pairing when comparing generators; neither multiple conditions, seeds, nor generators increase the number of independent topology units.

## Statistical summaries and their limits

Show paired Regret@5 and Regret@1 differences, their distribution over topologies, and improvement/tie/deterioration counts. Retain summaries by outer fold and by training seed to distinguish these sources of variation.

Use 10,000 paired bootstrap resamples of topology units on the already obtained outer out-of-fold predictions, reporting the 2.5th and 97.5th percentiles for mean paired differences. These are **approximate intervals conditional on the fitted models and evaluated partitions**. They do not include full uncertainty from retraining and repartitioning, remove dependence from shared training sets, or establish independence between related subnetworks.

Prespecify six primary formulation comparisons on Regret@5: `ltr_catboost` against each of the three supervised alternatives, separately in each generator. As an exploratory analysis, use a two-sided paired Wilcoxon test on topology differences, Pratt handling of zeros, average ranks for ties, and the normal approximation with continuity correction. Set `p=1` when every difference is zero. Apply Holm correction jointly across these six p-values. Report Regret@1 with paired differences and descriptive intervals without introducing an additional p-value family.

Dependence between out-of-fold predictions limits interpretation of these tests; they are not confirmatory demonstrations of population-wide superiority. A Wilcoxon test and a bootstrap interval for a mean difference also need not target the same estimand. Nonsignificance is not evidence of equivalence or noninferiority.

Publish the full heuristic comparison, including paired differences and descriptive conditional intervals. The executable contract specifies **no additional family of heuristic p-values**; the six supervised comparisons are the only hypothesis tests. The oracle is excluded from paired comparisons. Do not choose comparator families after examining results. The limits of model selection and cross-validation uncertainty are discussed by [Cawley and Talbot](https://www.jmlr.org/papers/v11/cawley10a.html) and [Bengio and Grandvalet](https://www.jmlr.org/papers/v5/grandvalet04a.html).

## Fixed-configuration learning curves

Evaluate `ltr_catboost` and `reg_normalized` at 10, 20, 40, and all 69–70 available outer-training topologies. Include all conditions of a selected topology. Generate three nested subset orders using seeds 1301–1303, shared across methods and generators, and evaluate every size on the same outer held-out fold. Represent the full training complement once; repeating identical data with the same training seed would not add replication.

Use `depth=6`, `learning_rate=0.03`, `l2_leaf_reg=3`, `iterations=3000`, no early stopping, and training seed 1201 for every curve fit. Do not substitute hyperparameters selected using a larger training subset. Record selected topologies, available and usable fronts, and candidate counts.

This is a learning curve for fixed model configurations. It does not estimate the optimum of a separately retuned procedure at each size or all training uncertainty. Subsample repetitions share evaluation data and are not new test sets. Objective-only rules provide common references across training sizes.

Cost: `2 cases × 5 outer folds × 2 arms × (3 small sizes × 3 subset orders + 1 full size) = 200 fits`.

## Feature-block ablation

Evaluate eight variants: each of the four feature blocks alone, and the full feature set with each block removed in turn. Within an outer fold, reuse the full ranker's internally selected configuration without retuning the ablated variants. Use seed 1201 and compare against the full model with that same seed, already available from the central comparison.

This measures sensitivity of the selected full-model configuration to feature availability. It is not causal importance of individual columns or the maximum achievable performance of each reduced feature set. Additional per-feature explanation methods are outside this planned block.

Cost: `2 cases × 5 outer folds × 8 ablations = 80 fits`.

## Leave-family-out analysis

Plan six descriptive exclusions: BioGrid, DREAM3 (`InSilico` in metadata), GRNdb, DREAM4 (`dream4`), eipo-modular, and scale-free. These are a fixed subset of the benchmark sources; all 104 fronts still participate in the central comparison. For each exclusion, train `ltr_catboost` and `reg_normalized` with the learning-curve configuration and seed 1201. Exclude from training every topology represented in the held-out family, including any of its conditions assigned to another source. Do not tune using the excluded family.

Report effective training and evaluation sizes. In particular, small synthetic families with four topologies provide limited evidence. These exclusions characterize the specified source shifts, not general clinical transferability.

Cost: `2 cases × 6 families × 2 arms = 24 fits`.

## Deployment models and biological application

After the nested evaluation, select a configuration for each arm using three grouped folds over all 87 topologies and the same six-configuration grid. Refit each selected arm on the available benchmark fronts with seed 1201 and the same training exclusions and weights. These are deployment models; their fits do not supply another independent performance estimate.

Cost: `2 cases × 4 arms × 6 configurations × 3 folds + 2 × 4 final models = 152 fits`.

The planned TCGA-BRCA application retains its defined samples, gene universe, regulatory sources, and cutoffs. Fix nine selectors before examining biological support: the four supervised arms, ReduceNEI, mean objective rank, TOPSIS, `metricdistribution`, and the trade-off-worthiness rule. The application provides descriptive contextual evidence. Incomplete and overlapping regulatory resources are neither independent biological replicates nor a gold standard. It does not establish robustness across new cohorts, platforms, or incomplete reference networks.

## Computational plan and artifact provenance

The complete proposed study contains **1,376 fits**, excluding the timing pilot and documented technical retries:

| Block | Planned fits |
| --- | ---: |
| Central nested comparison | 920 |
| Fixed-configuration learning curves | 200 |
| Feature-block ablations | 80 |
| Leave-family-out analysis | 24 |
| Deployment tuning and final models | 152 |
| **Total** | **1,376** |

Existing evolutionary fronts are reused. Record configurations, data and source hashes, explicit groups and folds, seeds, package versions, effective thread counts, elapsed times, and peak memory. Store compact prediction tables containing identifiers, scores, and dataset references instead of copying every predictor column into each output.

Measure feature preparation, model fitting, final-model training, and scoring separately. Estimates based on observed row counts and depth endpoints are computational scenarios, not runtime guarantees. Include stated margins and distinguish measured work from unmeasured feature regeneration, analysis, and technical retries. This document does not report timing measurements from a generated run.

Technical failures stop scheduling and terminate active workers. The fixed limits are two workers, eight CatBoost threads per worker, 3600 seconds per job and 96 hours of cumulative execution, including retries. Fewer workers/threads may be specified when creating a run and are then fixed for that run. No automatic retries or skipped failures are allowed. Explicit retries preserve earlier attempts in separate directories and use identical job definitions. A hard supervisor crash is charged conservatively through the time of resumption. Completed artifacts, source hashes, resources and environment are verified before resumption; altered successful outputs cause an error. A poor quality result is not a technical failure or a reason to exclude a run, add a configuration, choose another seed, or stop for significance.

Summaries require all 1,378 jobs (1,376 fits and two baseline jobs). They export front and topology metrics, central means, paired differences, the six Holm-adjusted tests, fold/seed diagnostics, learning curves, ablations with the same-seed full-model reference, family exclusions, and runtimes. Figures and the biological support analysis are subsequent steps. `grouped rank-deployment` exports nine label-free score columns for the BIO-INSIGHT application; it does not perform the source-support analysis itself.

## Contract and timing-pilot interface

The public module layout is:

```text
scripts/experiments/amiga_exp/grouped_validation/
    contract.py
    training_data.py
    pilot.py
    topology.py
    planning.py
    models.py
    baselines.py
    metrics.py
    execution.py
    runner.py
    summary.py
    deployment.py
    commands.py
```

`contract.py` generates deterministic assignments and verifies structural isolation, coverage, feature roles, label and weight policies, and source identities without evaluating AUPR values. `training_data.py` prepares only the requested training fronts. `pilot.py` measures bounded CPU execution; it is not the complete nested evaluation runner.

From the repository root, in an environment containing the experiment dependencies, generate a contract and inspect the timing plan:

```bash
python -m scripts.experiments.amiga_exp.grouped_validation.contract \
  --output experiments/grouped-validation/protocol-contract.json

python -m scripts.experiments.amiga_exp.grouped_validation.pilot \
  --contract experiments/grouped-validation/protocol-contract.json \
  --output experiments/grouped-validation/runs/pilot-001 \
  --dry-run
```

A dry run writes planning artifacts without fitting models. Use a distinct, unused output directory for each subsequent run; existing run directories must not be overwritten. Contracts, job plans, manifests, logs, and computational summaries are generated artifacts under `experiments/grouped-validation/`.

The timing pilot uses only the training portion of inner fold 0 within outer fold 0. It plans 16 fits: two cases, four arms, and depths 4 and 8, with learning rate 0.03 and 3000 trees throughout. Defaults are two worker processes with eight CatBoost threads each, a 3600-second global scheduling budget, and a 900-second limit per job. Bounded process cleanup follows termination. Thread limits apply to pool construction and prediction as well as fitting.

The pilot supplies no `eval_set`, performs no early stopping, computes no external-fold predictions or quality metrics, and does not use accuracy to alter the grid. Training predictions are checked only for valid shape and finite values. Runtime and memory evidence must come from the generated run artifacts; a planning document or dry-run manifest is not evidence that the fits completed.

Before full evaluation, verify topology isolation, predictor roles, label ties, weights, score-tie expectations, the knee scorer, random-reference expectations, metric aggregation, and manifest reproducibility. Freeze the implemented sources and configuration before producing the full set of out-of-fold results. The operational commands, including dry run and resumption, are documented in [Experimental Workflow](../experiments.md#grouped-evaluation).
