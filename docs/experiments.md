# Experimental workflow

`amiga-exp` shares the AMIGA project version. The source is prepared for version
0.2.0; release tags are created when a release is finalized.

The design comprises **labelled benchmarks in five phases**, a **real-data
application**, and a separate **prediction-cost analysis**. Read the
[installation guide](experiments/reproducibility.md) and
[scientific design](experiments/design.md) first.

| Phase or analysis | Command | Reference |
| --- | --- | --- |
| 1. Relevance labels | `benchmark selection` | [Selection specification](experiments/sequential-selection.md) |
| 2. Hyperparameters | `benchmark selection` | Complete family-specific parameter grids |
| 3. Columns and family | `benchmark selection` | Recursive training TreeSHAP and inner validation |
| 4. Held-out selector comparison | `benchmark comparison` | [Evaluation](experiments/outer-evaluation.md) |
| 5. Fixed-AMIGA learning curves | `benchmark learning` | [Benchmark guide](experiments/benchmark.md) |
| Real TCGA-BRCA application | `real` | [Application guide](experiments/real.md) |
| Prediction and sorting costs | `costs` | [Cost guide](experiments/prediction-costs.md) |

Phases 1–3 run together because each decision depends on the previous one.
Five outer folds hold out whole topologies, with three grouped inner folds for
selection. Selection uses seed 1101; final fits use seeds 1201–1205.
The five seeds repeat final estimation, not the entire parameter search.

Phase 4 compares AMIGA with direct AUPR regression and top-5%, top-10% and
top-20% classification. Top-5% and top-10% are exploratory threshold sensitivity
analyses. Objective selectors appear in a separate comparison panel.
`benchmark statistics` and `benchmark figures` read saved outputs; they fit
no models. Regret@5 is primary, with Hit@1 and Hit@5 as complementary metrics.
See [reporting](experiments/reporting.md) and [figures](experiments/figures.md).

Phase 5 fixes each outer fold's selected family, labels, parameters and exact
columns. It refits AMIGA at 10/20/40 labelled topology groups, three nested
subsets and five seeds. The full-size endpoint reuses phase-4 predictions.
These are conditional learning curves, without reselection at each data size.

The real-data analysis applies a benchmark-selected model to TCGA-BRCA and
compares the original five selectors using a fixed external evidence snapshot.
External support provides context with incomplete coverage. Prediction timing
covers scoring and sorting a prepared front, excluding feature extraction,
model loading, training and upstream network inference.

## Commands and local outputs

```bash
scripts/experiments/amiga-exp benchmark selection --help
scripts/experiments/amiga-exp benchmark comparison --help
scripts/experiments/amiga-exp benchmark learning status
scripts/experiments/amiga-exp benchmark learning report
scripts/experiments/amiga-exp real status
scripts/experiments/amiga-exp real report
scripts/experiments/amiga-exp costs report
scripts/experiments/amiga-exp organize --output experiments/studies
```

The [shared execution guide](experiments/supplementary.md) explains `prepare`,
`run` and `status`. This execution performs the phase-5 fits and one real-data
model fit together; the scientific analyses remain separate.

```text
experiments/studies/
├── benchmark/
│   ├── phases-1-4/
│   └── learning/
├── real/
│   ├── tcga-brca/
│   └── model/
└── costs/
    └── prediction_costs.csv
```

The index contains relative links to verified local outputs. It moves no files
and fits no models. All generated data, results, models and experimental figures
remain Git-ignored. GitHub contains the implementation and protocol metadata,
without numerical result deposits.
