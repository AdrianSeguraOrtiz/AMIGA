# Fixed-AMIGA learning curves and real-case prediction

The `supplement` workflow in amiga-exp 0.3.1 adds a small, fixed-configuration
analysis to completed phases 0–4. It does not rerun label screening, parameter
search, column selection or supervised comparisons. Existing benchmark results
and phase figures remain unchanged.

## Learning curves

Use the exact AMIGA family, relevance labels, hyperparameters and predictor
columns saved for each outer fold in phase 4. The configurations can differ
between folds; each stays fixed across training sizes and seeds within its fold.
Reuse the recorded nested subsets of 10, 20 and 40 labelled topology groups,
three subsets per size, all five outer folds and final seeds 1201–1205. Every
condition of a topology stays together. Test partitions are unchanged.

This is a conditional data-size sensitivity analysis of already selected models.
The configuration and columns were selected with the original full training
complement; the curve does not estimate selection from scratch under scarce
labels. Only model weights are relearned at each smaller training size. No
alternative supervised methods are included. The full-size point reuses the
existing five-seed AMIGA predictions, with 69–70 training topologies per fold.

Average seeds within subset/front, then subsets within front and conditions
within topology. Give each of the 87 evaluation topologies equal weight.
Regret@5 remains primary. The generated figure shows Regret@5 and Hit@5;
all twelve existing metrics are retained in CSV. Bands are conditional 95%
intervals from 10,000 topology bootstrap resamples with seed 1401.

## Updated TCGA-BRCA application

There is one deployment fit for BIO-INSIGHT. Reuse the phase-4 configuration
with the lowest recorded **inner-validation Regret@5**, breaking a tie by outer
fold index, then fit it on all 104 benchmark fronts with seed 1201. Its exact
saved columns remain fixed. No new search is run; outer test results and TCGA
regulatory support do not determine this choice. Model and schema are exported
in native format with integrity metadata.

Apply this model to the existing prepared TCGA-BRCA front. Retain the original
five selectors: AMIGA, ReduceNEI, mean objective rank, TOPSIS and metric
distribution. Reuse the same network reconstruction, source-support functions,
evidence snapshot and resource cutoffs: CollecTRI/DoRothEA/TRRUST/JASPAR top-250
and Cistrome BRCA-COR top-5000. No patient data download, network inference,
evolutionary optimization or new supervised comparison is performed. External
support remains contextual evidence with incomplete coverage.

## Prediction costs

After training has finished, measure AMIGA scoring and ordering of the prepared
real front five times after one warm-up. Report prediction, sorting and total
seconds, row/feature counts and thread count. Matrix preparation, model loading,
training, network reconstruction and feature generation are outside this timing
scope. This is the cost of predicting from a prepared front, not end-to-end cost.

## Run and monitor

```bash
scripts/experiments/amiga-exp supplement freeze \
  --output experiments/supplementary/fixed-contract-001.json
scripts/experiments/amiga-exp supplement run \
  --contract experiments/supplementary/fixed-contract-001.json \
  --output experiments/supplementary/fixed-001 --jobs 16 --threads 4
scripts/experiments/amiga-exp supplement status \
  --run experiments/supplementary/fixed-001
```

The plan has **91 independent jobs and 451 fits**: 90 learning-curve jobs with
five seeds each and one real-case deployment fit. There are no mask-parent fits
or configuration-selection jobs. The measured parallel layout uses 16 workers
with four disjoint CPU threads each and single-thread BLAS pools.

A complete `state.json` and verified stage manifests establish completion.
Technical interruptions require inspection and explicit `--resume --retry-failed`;
there are no automatic retries. Changed sources or policies need a new contract.
Earlier extended contracts are incompatible with this fixed-model workflow.

| Output under the pipeline directory | Contents |
| --- | --- |
| `state.json`, `run/state.json` | Pipeline status, completed jobs and active workers |
| `run/jobs/**/progress.json` | Completed/five planned fits and current seed |
| `summary/learning_curves.csv` | Metric means and conditional intervals |
| `summary/*-learning-curves.{pdf,png}` | AMIGA-only Regret@5 and Hit@5 curves |
| `summary/learning_metrics_long.csv` | Seed/subset/front-specific held-out metrics |
| `run/jobs/BIO-INSIGHT/deployment/final/` | Native model, saved columns and fitting receipt |
| `application/ranked_real.csv` | Updated real-front scores and ranks |
| `application/real_world_source_support_top1.csv` | Original five-selector support analysis |
| `application/prediction_costs.csv` | Five prepared-front prediction/sorting measurements |

## Reproducibility

The [benchmark deposit](../../benchmark-artifacts/README.md) already provides
processed inputs, completed comparisons, predictions and phase-figure evidence.
This workflow adds its models and results only after successful completion.
Packaging and verification do not fit more models or constitute another
experimental phase. The source and artifact tags are separate and immutable.
