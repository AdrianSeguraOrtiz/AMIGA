# Extended supplementary workflow (reference)

The extended execution described here was stopped before completion. This
specification documents the implemented workflow and its frozen contract;
it is not the current supplementary plan and no resumption is scheduled.
Partial fitting outputs do not establish completed learning curves, deployment
models or application results. The reduced, unexecuted scope is recorded in
the [experimental overview](../experiments.md#supplementary-evaluation-blocks).
Commands below are reference documentation, not instructions to launch the
reduced analysis.

The `supplement` workflow extends completed phases 0–4 without changing their
contracts, predictions, figures or primary metric. It uses the same processed
BIO-INSIGHT/MO-GENECI inputs, 87 topology groups and full original grids. It is
part of amiga-exp 0.3.0; the public `amiga` API is unchanged.

## Learning-curve design

For each of the five original outer folds, restrict labelled training data to
10, 20 or 40 topology groups. Use the three original nested subsamples with
seeds 1301–1303. Include all conditions of each selected topology. Unselected
training labels and outer test labels are unavailable to model selection.

For each subset, construct three grouped inner folds using the deterministic
seed recorded in the contract. Repeat all eight ranking-label reference fits,
the complete 27/36/36 parameter grids and the four recursive feature budgets
for LightGBM/XGBoost/CatBoost. Compare AMIGA with direct AUPR regression; each
has its own parameters, columns and family. Selection minimizes topology-mean
Regret@5, then Regret@1, and follows the original stable tie rules. Feature
selection is training-only centered TreeSHAP, with the original seed and sample
budget. There is no early stopping or outcome-driven reduction of the search.

Relearn selected masks on the complete available subset and fit final estimators
with all five seeds 1201–1205. Save candidate predictions before requesting test
quality labels. The test folds are identical to the original outer evaluation.
The full-size endpoint reuses its saved five-seed predictions: 69 or 70 labelled
training topologies, depending on the fold. It does not run three artificial
copies of this endpoint.

Average final-seed metrics within subset/front, then the three subset repetitions
within front, then conditions within topology. Give the 87 evaluation topologies
equal weight. Bands use 10,000 topology bootstrap resamples, seed 1401. They are
conditional descriptive intervals; overlapping training folds and subset
repetitions are not independent biological replicates. Retain every size even
if curves are nonmonotonic. Regret@5 remains primary.

## Deployment and TCGA-BRCA

Deployment selection uses three grouped folds across all 104 benchmark fronts
(87 topologies), split seed 20260916. Repeat full label/parameter/column/family
selection for AMIGA and parameter/column/family selection for direct AUPR
regression and classification top-5%, top-10% and top-20%. Relearn masks on all
benchmark inputs, then train one deployment estimator per method with the fixed
seed 1201. Persist native models and metadata; no pickle is needed.

The TCGA-BRCA front is scored using predictors only. Model choice is independent
of this front and of external regulatory support. The application compares the
five supervised methods with ReduceNEI, mean objective rank, TOPSIS, metric
distribution and the knee rule. A Top1 score tie chooses the smallest item ID;
the output records the number of tied candidates. Benchmark metrics continue
to use their original expected-tie definition.

Reconstruct Top1 networks using the original inference inputs and consensus
weights. Reuse exactly the recorded evidence snapshot for CollecTRI, DoRothEA,
TRRUSTv2, JASPAR PWM and Cistrome Cancer BRCA-COR, with the original top-250 and
top-5000 cutoffs. The snapshot has incomplete coverage; Cistrome is restricted
to its recorded source TFs. Absent pairs are not known negatives. Support is
contextual evidence, not gold-standard biological accuracy or a new cohort test.

The existing expression preprocessing used open GDC TCGA-BRCA primary-tumour
STAR Counts (`tpm_unstranded`): protein-coding genes, mean TPM over duplicate
aliquots per sample, highest-mean-TPM resolution of duplicate gene symbols,
`log2(TPM+1)`, TPM at least 1 in at least 10% of samples, and the 500 most variable
genes. The recorded matrix has 1,106 samples. The supplement consumes the
existing processed front; it does not download patients or rerun inference.

## Computational measurements

Before parallel model training, measure feature preparation in isolated
processes at 100, 250 and 500 genes, each with three repetitions. Take the first
genes of the fixed variance-ordered universe and the smallest original item ID,
independently of model scores. The sample count remains 1,106. Record expression
reading/features, construction of restricted inference inputs, weighted
consensus and network-feature extraction separately. Absolute process peak RSS
includes loading and preparation; it is not incremental memory per stage.

Stored original fitting records provide job times and peak RSS by case, method
and stage. Summed worker seconds are computational work, not elapsed time of a
parallel pipeline. Supplementary fitting has its own runtime table. Real-front
scoring is measured five times after warm-up for each saved model, with native
model-loading time recorded separately. These costs exclude upstream network
inference and evolutionary front generation. Repeated reads may use system cache.
Do not label a one-candidate measurement as measured full-front cost.

## Execution and monitoring

Freeze only after the complete preceding runs exist at their documented paths:

```bash
scripts/experiments/amiga-exp supplement freeze \
  --output experiments/supplementary/contract-001.json

scripts/experiments/amiga-exp supplement run \
  --contract experiments/supplementary/contract-001.json \
  --output experiments/supplementary/full-001 --jobs 16 --threads 4
```

The plan contains **368 jobs, 70,804 reference/tuning/feature/final fits**, plus
at most **570** parent fits to relearn final feature masks. Deployment uses five
formulations; learning curves use two. The fitting budget is six days and the
per-job timeout is twelve hours. No automatic retries or omitted failures are
allowed. CPU sets are disjoint, with four threads per worker and single-thread
BLAS pools. A smaller machine needs an appropriate fixed worker count.

```bash
scripts/experiments/amiga-exp supplement status \
  --run experiments/supplementary/full-001

scripts/experiments/amiga-exp supplement run \
  --contract experiments/supplementary/contract-001.json \
  --output experiments/supplementary/full-001 --jobs 16 --threads 4 \
  --resume --retry-failed
```

Use explicit retry only after inspecting the failure. Successful job hashes,
source/data hashes, environment and resource layout are verified on resumption.
Summary/application/cost stages publish atomically and can recover a missing
checkpoint after successful publication. Technical failure is distinct from
unfavourable metrics. Changing code or policy requires a new frozen contract.

| Output under the pipeline directory | Content |
| --- | --- |
| `state.json` | Pipeline state, completed stages and error traceback |
| `run/state.json` | Fit counters, active jobs, heartbeat and supervisor PID |
| `run/jobs/**/progress.json` | Completed/planned inner fits and current candidate |
| `summary/learning_curves.csv` | All twelve metrics, topology means and intervals |
| `summary/*-learning-curves.{pdf,png}` | Regret@5 and Hit@5 curves |
| `summary/learning_metrics_long.csv` | Seed/subset/front-specific held-out metrics |
| `summary/selected_procedures.json` | Selection decisions for every context |
| `application/` | Ten selectors, native-model identities, network support and scoring costs |
| `costs/` | Isolated feature measurements and recorded fitting-cost summaries |

Require `status: complete` and verified output manifests before using results.

## Artifact distribution

The [benchmark deposit](../../benchmark-artifacts/README.md) distributes
processed inputs and current phase evidence separately from source code. Create
a supplementary deposit only after this pipeline completes:

```bash
scripts/experiments/amiga-exp supplement archive \
  --run experiments/supplementary/full-001 \
  --output benchmark-artifacts/supplementary-v0.3.0
scripts/experiments/amiga-exp supplement verify-archive \
  --archive benchmark-artifacts/supplementary-v0.3.0
```

Archives include native deployment models, saved learning predictions, compact
metrics for every inner candidate and the derived fixed evidence snapshot. Full
inner candidate scores stay local and can be regenerated from the inputs. The
patient-level expression matrix and unfiltered external downloads are excluded;
cost receipts record their hashes. Repeating feature-preparation measurements
requires the recorded expression matrix. External evidence retains its source
attribution and terms, independently of the software's MIT license.
