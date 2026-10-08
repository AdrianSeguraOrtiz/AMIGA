# Top-five-percent classification sensitivity

This experiment changes the high-quality classification target from the top
20% to the top 5% within each training front. Both are post-Pareto selectors.
The percentage specifies training labels; Regret@5 evaluates the best quality
among five recommendations. Neither percentage is assumed to be optimal.

The binary positive threshold is the descending AUPR at position `ceil(n/20)`.
Include ties at that threshold. If it equals the minimum in an informative
front, positives strictly exceed the minimum, as in the top-20% comparator.
Exclude flat training fronts under the same common policy. Labels use current
training fronts only. Score unseen candidates with positive-class probabilities.
There is no prediction probability cutoff or target AUPR required at deployment.

Use the exact five outer topology folds and three inner folds of the completed
sequential experiment, the original 104/101 predictors, equal-topology training
weights, selection seed 1101, 3,000 requested trees and the original grids:
27 LightGBM, 36 XGBoost and 36 CatBoost configurations. Tune each family separately,
then apply the same training-only recursive centered tree-SHAP path at feature
fractions 100/75/50/25%. Select family and fraction by topology-mean Regret@5,
then Regret@1, fewer features and stable identifier. The binary target is fixed;
there is no relevance-label screening for a classifier.

Inner selection requires 990 parameter jobs and 30 feature-path jobs:
2 cases × 5 outer complements × 99 configurations × 3 inner fits, plus
2 × 5 × 3 families × 4 fractions × 3 inner fits = **3,330 fits in 1,020 jobs**.
This is the same parameter/feature search budget as the previous top-20% arm.
The selected ten procedures are frozen before outer predictions. Relearn reduced
feature masks on each whole outer training complement. Final evaluation uses
seeds 1201–1205, requiring **50 final fits** and at most 30 additional mask fits.

Reuse the audited phase-4 scores for ranking, top-20% classification, both
regressions and objective-only selectors. No existing source or result is replaced.
Average metrics across seeds, then conditions within topology, then equally
across the 87 topologies. Regret@5 is primary; Regret@1, Hit@1 and Hit@5 are
secondary. Save paired differences of top-5% minus ranking and top-5% minus
top-20%, including conditional descriptive 95% intervals from 10,000 paired
topology bootstrap samples (seed 1401). No new hypothesis-test family is added.
Always report both cases and both target definitions, regardless of the direction
of the differences. These reused benchmarks provide exploratory sensitivity
evidence, not fresh independent confirmation.

Use 16 worker processes × 4 threads, disjoint CPU affinity and single-thread
BLAS. Both execution stages have a two-hour per-job timeout, six-day cumulative
limit per stage and no automatic retries. The detached pipeline runs inner
selection, freezes outer selections, fits all seeds and generates audited tables
and PNG/PDF figures automatically. Completion requires all jobs, exact coverage,
checksummed sources/artifacts and recomputation of metrics from saved scores.

```bash
scripts/experiments/amiga-exp top5-classification freeze \
  --output experiments/top5-classification/selection-contract-001.json

.venv/bin/python -u -m scripts.experiments.amiga_exp.top5_classification.pipeline \
  --contract experiments/top5-classification/selection-contract-001.json \
  --output experiments/top5-classification/launches/full-001 \
  --work-output experiments/top5-classification/full-001 \
  --jobs 16 --threads 4

scripts/experiments/amiga-exp top5-classification status \
  --run experiments/top5-classification/launches/full-001
```

Resume the same pipeline with `--resume`; after inspecting a failed attempt,
add `--retry-failed`. Successful fits are retained and definitions cannot drift.
