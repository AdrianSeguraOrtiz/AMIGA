# 🧬 AMIGA: Autonomous Multi-objective Individual GRN Autoselection

<img src="./docs/logo.png" width="40%" align="right" style="margin: 1em">

**AMIGA** is an intelligent advisor for the **automatic selection of promising individuals** within Pareto fronts obtained by multi-objective algorithms for **Gene Regulatory Network (GRN)** consensus inference.

It arises as a **complementary contribution** to the consolidated [GENECI](https://github.com/AdrianSeguraOrtiz/GENECI) research line, being fully compatible with all consensus-based software packages developed along it, including [MO-GENECI](https://github.com/AdrianSeguraOrtiz/MO-GENECI) and [BIO-INSIGHT](https://github.com/AdrianSeguraOrtiz/BIO-INSIGHT).

---

## 🔬 Context and Motivation

During the development of **MO-GENECI**, it was observed that **neighboring solutions in the Pareto front showed very similar precision levels** (in terms of AUROC and AUPR) when evaluated against the gold standards. This behavior suggested that **local regions of the front were topologically stable**, meaning that individuals close to a high-performing one were likely to be biologically relevant as well. Such insight motivated the design of **[PBEvoGen](https://github.com/AdrianSeguraOrtiz/PBEvoGen)**, which enabled experts to **steer the evolutionary search toward promising regions** of the objective space by exploiting this neighborhood consistency.

As the framework evolved into **BIO-INSIGHT**, the number of optimization objectives increased to **six**, considerably expanding the dimensionality and complexity of the fronts and making expert guidance or visual interpretation far more challenging. To address this, **AMIGA** was conceived — an **Autonomous Multi-objective Individual GRN Autoselection** system that **learns from previously evaluated fronts** across benchmark datasets to **predict and prioritize new individuals likely to achieve high precision**.

AMIGA relies on **Learning-to-Rank (LTR)** models trained on rich descriptors that combine:

* **expression-based features** from the gene expression matrices,
* **topological properties** of the consensus networks associated with each individual,
* **weights assigned to inference techniques**, and
* **objective-level optimization metrics** defining the individual’s position within the front.

---

## 🧩 Knowledge Sources for Learning

The learning model in AMIGA is trained from **four complementary sources of information**, each capturing a distinct layer of biological and algorithmic knowledge that characterizes every individual in a Pareto front. These sources are harmonized into a unified representation used by the Learning-to-Rank models.

### 1️⃣ Technique Weights per Individual

Each individual in the evaluated front encodes a **unique vector of weights assigned to the base inference techniques** (e.g., GENIE3, CLR, ARACNE, etc.).
These weights determine **how much each underlying GRN contributes to the individual’s consensus network**.

* In AMIGA’s dataset, each column labeled as `GRN_*.csv` represents the contribution of one technique.
* Together, these weights summarize the **algorithmic composition** of the inferred network and reflect the **search behavior of the consensus optimizer**.
* They serve as the first evidence source for the model, providing insight into how different inference strategies interact within each solution.

### 2️⃣ Objective Optimization Levels

Every individual also carries **its specific level of optimization for each objective** used in the multi-objective consensus process.

* These metrics describe the **position of the individual within the objective space**, i.e., its trade-off among the optimization goals (e.g., network sparsity, predictive accuracy, diversity, biological consistency, etc.).
* Their semantics depend on the **originating consensus algorithm**:

  * For individuals from **MO-GENECI**, objectives correspond to the bi-objective or tri-objective evolutionary configuration.
  * For **BIO-INSIGHT**, objectives encompass up to six criteria, integrating both structural and biological indicators.
* Consequently, these values capture the **global optimization context** that shaped each solution, providing the model with a quantitative reference of how “optimal” an individual is within its front.

### 3️⃣ Expression-based Features (shared across the front)

These features are computed once per dataset from the **gene expression matrix** (genes × conditions) using the routines in [`features/expression.py`](./amiga/features/expression.py).
They represent the **statistical and structural context** in which all individuals of a front are generated.

* They include **global descriptors** (size, variance, skewness, kurtosis),
* **per-gene and per-condition aggregates** (mean, standard deviation, coefficient of variation, missing/zero ratios),
* **pairwise correlations** (gene–gene and condition–condition),
* **dimensionality properties** from **PCA** (variance ratios, effective rank, spectral condition), and
* optionally **time-series trends** if the conditions follow temporal order.

These descriptors allow AMIGA to internalize the **intrinsic variability and structure of the input expression data**, providing a shared contextual layer for all individuals within the same front.

### 4️⃣ Consensus Network Features per Individual

Finally, each individual’s **consensus GRN** — reconstructed from its technique weights — is analyzed to extract **graph-level features** using the metrics implemented in [`features/grn.py`](./amiga/features/grn.py).
This set of descriptors captures **topological and modular properties** of the inferred regulatory network, including:

* **Global weighted indicators** such as density, total/average confidence, and Gini coefficient of edge weights.
* **Node-level statistics** (in/out/total strength, hub abundance, and strength heterogeneity).
* **Topological organization** metrics: assortativity, average clustering, shortest-path statistics, and modularity via Louvain communities.
* **Reciprocity** of regulatory relationships (bidirectional regulation).
* **Advanced entropy-based summaries** to measure distributional diversity in weights and strengths.

Together, these metrics provide a **structural fingerprint** of each network, enabling AMIGA to learn how certain network configurations correlate with higher biological accuracy or stability.

---

## ⚙️ Architecture Overview

```
amiga/
 ├─ core/
 │   └─ main.py                # Core logic: LTR training, ranking, and feature extraction
 ├─ features/
 │   ├─ expression.py          # Metrics from gene expression matrices
 │   └─ grn.py                 # Metrics from GRNs
 ├─ selection/
 │   └─ learn2rank.py          # Learning-to-Rank models (LightGBM / XGBoost / CatBoost)
 ├─ utils.py                   # I/O utilities and data normalization
 └─ cli.py                     # Typer-based command-line interface
```

AMIGA is implemented in Python and designed to interoperate with **GENECI**, **MO-GENECI**, and **BIO-INSIGHT**, using the same file formats and feature conventions for GRNs and expression data.

---

## 🚀 Installation

AMIGA is managed with **Poetry**, which handles both dependency resolution and virtual environment management.

### 1️⃣ Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2️⃣ Install Poetry (if not already installed)

```bash
pip install poetry
```

### 3️⃣ Install AMIGA and all its dependencies

```bash
poetry install
```

This command will automatically create a local environment (if not active) and install all dependencies listed in `pyproject.toml`, including:

* **GENECI 4.0.1.1**
* **LightGBM 4.6.0**
* **XGBoost 3.0.5**
* **CatBoost 1.2.8**
* **scikit-learn**, **scipy**, **networkx**, **python-louvain**, etc.

Once installed, the `amiga` CLI command becomes available system-wide.


## 🧩 Basic Usage

### 1️⃣ Cross-validated training

```bash
amiga train-cv data/training.csv --model LGBMRanker --n-splits 5 --label-mode rank_dense -o output/
```

Generates:

* One model per fold (`model_fold1.pkl`, ...)
* CSV with predictions per fold (`valid_fold1_ranked.csv`, ...)
* Cross-validation report (`cv_report.json`)
* Feature metadata (`feature_columns.json`)

---

### 2️⃣ Full-training for production

```bash
amiga train-full data/training.csv --out ./output_full/
```

Produces a single production model (`model.pkl`) and metadata files.

---

### 3️⃣ Ranking new fronts

```bash
amiga rank-csv data/new_front.csv output_full/model.pkl --out ranked.csv
```

Generates a ranked CSV including `score` and `rank_in_front` columns.

---

### 4️⃣ Extracting expression-level features

```bash
amiga extract-expr-features data/expression.csv -o features_expr.json
```

Computes global, per-gene, per-condition, correlation, PCA, and (optionally) time-series metrics.

---

### 5️⃣ Extracting GRN-level features

```bash
amiga extract-grn-features data/GRN_GENIE3.csv -o features_grn.json
```

Computes weighted graph metrics such as density, strength, assortativity, clustering, community modularity, and reciprocity.

---

### 6️⃣ Building training datasets

Combine an evaluated front, a gene expression matrix, and a folder of inferred GRNs:

```bash
amiga build-data evaluated_front.csv expression.csv ./lists --front-id 1 --out data.csv
```

Creates a complete dataset merging all features and target metrics for LTR training.

---

## 🧠 Citation

If you use **AMIGA** in your research, please cite it as:

> Segura-Ortiz, A., Giménez-Orenga, K., García-Nieto, J., Oltra, E., & Aldana-Montes, J. F. (2025). Multifaceted evolution focused on maximal exploitation of domain knowledge for the consensus inference of Gene Regulatory Networks. Computers in Biology and Medicine, 196, 110632.



