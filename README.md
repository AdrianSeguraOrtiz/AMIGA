# AMIGA

<img src="https://raw.githubusercontent.com/AdrianSeguraOrtiz/AMIGA/main/docs/logo.png" width="38%" align="right" alt="AMIGA logo">

**AMIGA** (*Automated Multi-objective Individual GRN Assessment*) is a Python
package and command-line tool for ranking candidate consensus Gene Regulatory
Networks (GRNs) produced by multi-objective evolutionary algorithms.

AMIGA acts as a post-Pareto decision layer. It learns from labelled benchmark
fronts where candidate quality is known, and applies the trained ranker to new
fronts where no gold standard is available.

The project is distributed on PyPI as `amiga-grn`; the import package and CLI
entry point are both named `amiga`.

[![AMIGA workflow: learning from labelled benchmark fronts and ranking new candidates without reference labels](docs/graphical_abstract/AMIGA_graphical_abstract_preview.png)](docs/graphical_abstract/AMIGA_graphical_abstract.pdf)

The [graphical overview](docs/graphical_abstract/README.md) includes editable SVG,
vector PDF and the code needed to regenerate the figure.

## Installation

```bash
pip install amiga-grn
```

For local development:

```bash
git clone https://github.com/AdrianSeguraOrtiz/AMIGA.git
cd AMIGA
poetry install
```

Check the CLI:

```bash
amiga --help
```

## What AMIGA Uses

AMIGA expects one row per candidate solution in a Pareto front. A training table
usually contains:

- `front_id`: identifier of the Pareto front used as a learning-to-rank group;
- `item_id`: candidate identifier, generated when absent;
- a target column such as `AUPR` for labelled benchmark fronts;
- numerical predictors describing the candidate, objectives, expression context
  and consensus network.

At prediction time, the target column is not required. The new front must provide
the same predictor schema expected by the trained model.

## Quickstart

The usual AMIGA workflow is:

```mermaid
flowchart LR
    subgraph IN["Input files"]
        A(["Gold standard"])
        B["Front CSV"]
        T(["target_col<br/>e.g. AUPR"])
        C["Expression CSV"]
        D["Base GRNs<br/>GRN_*.csv"]
    end

    A -.-> T
    B -.-> T
    B --> E(["amiga build-data<br/>--target-col AUPR"])
    T -. "labelled benchmark" .-> E
    C --> E
    D --> E

    B --> F(["amiga build-data<br/>--allow-unlabeled"])
    C --> F
    D --> F

    E --> G["Labelled table<br/>target + features"]
    F --> H["Unlabelled table<br/>features only"]
    S{{"Required for ranking:<br/>same feature columns<br/>as the trained model"}}

    subgraph CV["Model selection"]
        I(["amiga train-cv"]) --> J(["amiga summarize-cv"]) --> K(["amiga plot-cv"])
    end

    subgraph FINAL["Final ranking"]
        L(["amiga train-full"]) --> M["trained model<br/>feature schema"]
        M --> N(["amiga rank-csv"])
    end

    G --> I
    G --> L
    H --> N
    S -.-> H
    S -.-> M
    N --> O["Ranked front<br/>score + rank"]

    classDef input fill:#eef6ff,stroke:#5083c7,color:#1d2b3a;
    classDef command fill:#14796f,stroke:#0a4f49,color:#ffffff,font-weight:bold;
    classDef artifact fill:#fff8e6,stroke:#d39b28,color:#3a2b10;
    classDef report fill:#f4edff,stroke:#8065bd,color:#2d2148;
    classDef chip fill:#fff3d9,stroke:#d39b28,color:#3a2b10;
    classDef note fill:#f8fafc,stroke:#64748b,stroke-dasharray: 4 3,color:#334155;
    classDef group fill:#ffffff,stroke:#cbd5e1,color:#334155;

    class B,C,D input;
    class A,T chip;
    class E,F,I,J,K,L,N command;
    class G,H,M,O artifact;
    class J,K report;
    class S note;
    class IN,CV,FINAL group;
```

### 1. Build AMIGA Tables

`build-data` is the main adapter between a consensus front and AMIGA. It takes
a candidate table, the expression matrix used to generate the front, and the
folder of base GRNs. The front table must contain one row per candidate and
weight columns named like `GRN_*.csv`.

For labelled benchmark fronts:

```bash
amiga build-data front.csv expression.csv base_networks/ \
  --front-id 1 \
  --target-col AUPR \
  --out labelled_front.csv
```

For real or unlabelled fronts:

```bash
amiga build-data front.csv expression.csv base_networks/ \
  --front-id 1 \
  --allow-unlabeled \
  --out unlabelled_front.csv
```

`build-data` preserves candidate/objective columns, assigns `front_id` and
`item_id`, extracts expression-level descriptors once per front, reconstructs
each weighted consensus GRN from the base networks and appends network
descriptors as `grn_*` columns. Useful options include:

- `--drop-front-cols`: remove columns from the original front before training;
- `--threads`: parallelize per-candidate consensus reconstruction;
- `--target-col`: choose the supervised quality column in labelled fronts;
- `--allow-unlabeled`: allow inference tables without a target column.

If you already have a complete AMIGA-compatible table, you can skip
`build-data` and train directly from that CSV.

### 2. Train And Evaluate With Grouped CV

`train-cv` trains one model per fold using fronts as groups. This is useful for
model selection, parameter tuning and reporting.

```bash
amiga train-cv labelled_fronts.csv \
  --model LGBMRanker \
  --label-mode continuous \
  --n-splits 5 \
  --model-params-json params.json \
  --out-dir cv_results/
```

Common options:

- `--model`: `LGBMRanker`, `XGBRanker` or `CatBoostRanker`;
- `--label-mode`: label construction strategy inside each front;
- `--label-quantiles`: number of bins when using quantile labels;
- `--front-col`, `--target-col`, `--id-col`: control-column names;
- `--drop-cols`: extra columns excluded from the feature matrix;
- `--model-params-json`: backend-specific hyperparameters;
- `--random-state`: base seed for reproducibility.

### 3. Summarise And Plot CV Results

`summarize-cv` aggregates one or more `cv_report.json` files into CSV tables.

```bash
amiga summarize-cv <cv_reports_dir>/*/cv_report.json \
  --out summary/ \
  --stats metric_rank_stats
```

`plot-cv` renders generic figures from the summaries:

```bash
amiga plot-cv --input-dir summary/ --plot dotplot_overview
```

Available plot names include `dotplot_overview`, `topk_curves`,
`metric_rank_heatmap` and `metric_scatter`.

### 4. Train A Final Model

```bash
amiga train-full labelled_fronts.csv \
  --model LGBMRanker \
  --label-mode continuous \
  --model-params-json params.json \
  --out-dir trained_model/
```

`train-full` uses the complete labelled table and stores `model.pkl`,
`feature_columns.json` and model metadata.

### 5. Rank A New Front

```bash
amiga rank-csv unlabelled_front.csv trained_model/model.pkl \
  --out-csv ranked_front.csv
```

When `feature_columns.json` is present next to the model, `rank-csv` uses the
same feature order learned during training.

### Optional Feature Utilities

`extract-expr-features` and `extract-grn-features` are standalone diagnostics
for inspecting descriptor blocks. They are not required before `build-data`,
because `build-data` calls the same feature extractors internally.

```bash
amiga extract-expr-features expression.csv --out expression_features.json
amiga extract-grn-features network.csv --out network_features.json
```

## Rankers And Labels

AMIGA currently supports three tree-based learning-to-rank backends:

- `LGBMRanker`
- `XGBRanker`
- `CatBoostRanker`

Labels are constructed within each front, never across unrelated fronts.
Supported label modes include continuous intra-front normalization, dense ranks,
average ranks, quantiles and negative-control modes for validation experiments.

Ranker-specific hyperparameters can be passed with `--model-params-json` in
training commands. This keeps AMIGA's CLI stable while still allowing advanced
configuration of the underlying backend.

## Reporting Metrics

`train-cv` evaluates rankings per front and then aggregates across fronts. The
main top-k metrics include:

- `Regret@k`: gap between the best true quality in the front and the best true
  quality found among the top-k recommendations;
- `BestAUPR@k`: best true AUPR recovered in the top-k;
- `Hit@k`: whether at least one true best candidate appears in the top-k;
- `NDCG@k`, Spearman and Kendall as complementary ranking diagnostics.

This design focuses on the practical decision problem: placing high-quality
candidate networks near the top of a short recommendation list.

## Experimental workflow

[![AMIGA experimental design in five phases](docs/experimental_design/AMIGA_experimental_design_preview.png)](docs/experimental_design/AMIGA_experimental_design.pdf)

The [experimental-design diagram](docs/experimental_design/README.md) includes
editable sources, vector exports and instructions to regenerate all five phases.

The repository includes `amiga-exp` for the documented benchmark protocol,
real-data application and prediction-cost measurements. It shares the AMIGA
project version. The source prepared for the next release is **0.2.0**;
`pip install amiga-grn` installs the currently published PyPI package.

Install the repository workflow with:

```bash
git clone https://github.com/AdrianSeguraOrtiz/AMIGA.git
cd AMIGA
POETRY_VIRTUALENVS_IN_PROJECT=true poetry install --with experiments
scripts/experiments/amiga-exp --help
scripts/experiments/amiga-exp --version
```

The commands follow the scientific design:

| Scope | Command | Purpose |
| --- | --- | --- |
| Benchmark phases 1–3 | `benchmark selection` | Labels, parameters, columns and ranker family |
| Benchmark phase 4 | `benchmark comparison` | Held-out supervised and objective-selector comparisons |
| Benchmark phase 5 | `benchmark learning` | Fixed-AMIGA learning curves |
| Real-data application | `real` | Saved-model ranking and contextual regulatory support |
| Prediction costs | `costs` | Prediction and sorting of a prepared candidate front |

`benchmark figures` and `benchmark statistics` generate figures and statistical
reports from saved outputs. `prepare`, `run` and `status` manage the shared
learning-curve and deployment execution after the comparison is complete.
`organize` creates a local index of verified results.

The experimental component runs from the repository; it is not installed by
the PyPI package. The repository contains source, tests, dependency locks and
input/protocol metadata. Datasets, fitted models, numerical results and
experimental figures remain in Git-ignored local folders.

See the [workflow guide](docs/experiments.md),
[installation and input requirements](docs/experiments/reproducibility.md), and
[scientific design](docs/experiments/design.md).

## Citation

If you use AMIGA, please cite the software using [CITATION.cff](CITATION.cff).

## License

AMIGA is released under the MIT License.
