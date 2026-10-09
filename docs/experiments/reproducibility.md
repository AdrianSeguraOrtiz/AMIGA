# Installing and reproducing amiga-exp 0.3.0

`amiga-exp` is distributed in this Git repository and versioned independently
of the `amiga-grn` PyPI package. The release tag is `amiga-exp-v0.3.0`.
Installing `amiga-grn` alone does not install this workflow or its benchmark data.

## Installation

Use Python 3.11–3.13 and Poetry. Python 3.13 and Poetry 1.6.1 were used for the
recorded execution environment; each run additionally records actual libraries
and numerical thread pools. CPU training is used by the current protocol.

```bash
git clone --branch amiga-exp-v0.3.0 https://github.com/AdrianSeguraOrtiz/AMIGA.git
cd AMIGA
POETRY_VIRTUALENVS_IN_PROJECT=true poetry install --with experiments
poetry run scripts/experiments/amiga-exp --version
poetry run scripts/experiments/amiga-exp --help
```

The installation command creates the repository-local `.venv` used by the
pipeline recipes in a fresh checkout. The committed `poetry.lock` pins resolved
dependencies. Keep it with the release;
do not regenerate it to reproduce the recorded environment. The wrapper uses the
repository's `.venv` when present, otherwise `python3`; `poetry run` ensures it
uses Poetry's selected environment. `PYTHON=/path/to/python` explicitly selects
an interpreter. Always run the commands below from the repository root.

## Input data and release scope

The release contains source code, scientific specifications, case/feature
metadata, topology groups, dependency lock and tests. Generated datasets,
contracts, model outputs and figures under `experiments/` are separate artifacts.
The [versioned benchmark deposit](../../benchmark-artifacts/README.md) distributes
processed inputs, completed figure evidence and held-out predictions in verified
compressed containers. It does not include all biological raw inputs.

Full benchmark execution needs these exact processed CSVs:

| Case | Required path | SHA-256 |
| --- | --- | --- |
| BIO-INSIGHT | `experiments/BIO-INSIGHT/data/data_104.csv` | `b95b2613481682e315250c7cf33433373f30db1100db352941cca43433c008f7` |
| MO-GENECI | `experiments/MO-GENECI/data/data_104.csv` | `c851b5c24dcd72e7f240cf36da284d29953325e1c1456619f054e6ab9d3f6cd6` |

Both contain 31,200 candidates across 104 fronts. Case metadata under
`docs/experiments/contracts/` records columns, predictor sets and identifiers.
The topology map under `docs/experiments/groups/` assigns the fronts to 87 groups.
These processed inputs suffice for fitting and evaluating the post-Pareto
selectors; rerunning the evolutionary generators is a separate task.

For independent verification of the topology map, obtain the GENECI expression
and reference-network collection identified by `source_collection` and
`source_path` in `topology_groups.json`, then run:

```bash
poetry run python -m scripts.experiments.amiga_exp.grouped_validation.topology \
  --source-root /path/to/GENECI/input_data
```

Restore the processed data and evidence with:

```bash
git fetch origin tag amiga-exp-benchmarks-v0.3.0
git restore --source=amiga-exp-benchmarks-v0.3.0 -- benchmark-artifacts/comparison-v0.3.0
scripts/experiments/amiga-exp supplement restore-archive \
  --archive benchmark-artifacts/comparison-v0.3.0 --destination .
```

For a fresh full execution, copy only the two processed input CSVs from the
restored evidence to another checkout. Restoring completed summaries/job
receipts and then trying to use their paths as new execution destinations will
correctly fail. TCGA-BRCA resources and supplementary analyses have a separate
scope and status in the [experimental overview](../experiments.md#supplementary-evaluation-blocks).

## Execution order

1. Install dependencies and place the exact input CSVs at the paths above.
2. Use [sequential selection](sequential-selection.md) to calibrate resources,
   freeze the complete-grid contract and execute phases 0–3.
3. Use [outer evaluation](outer-evaluation.md) to freeze the selected procedures,
   refit with five seeds and audit held-out predictions and metrics.
4. Run [top-5% classification](top5-classification.md), followed by
   [top-10% classification](top10-classification.md). Their contracts reuse
   preceding comparator outputs and require the documented directory layout.
5. Run the [supervised report](reporting.md) to regenerate the final rank tables,
   exploratory tests and figures from those completed summaries.
6. Generate the [phase figures](figures.md). This completes reproduction of the
   recorded benchmark phases; supplementary analyses have a separate scope and
   are not required to reproduce these results.

The detailed specifications contain freeze, run, summarize, resume and status
commands. Completed recorded runs use the `full-001` and `evaluation-001`
identifiers. In a fresh clone those destinations do not exist and can be used
as documented. To repeat the whole sequence, use another checkout/workspace:
the classifier contract builders intentionally refer to these preceding run
locations. Do not rename only one dependency directory or alter frozen source
files to bypass hash checks.

The measured layout was 16 processes × 4 disjoint CPU threads on a 64-CPU
Linux machine. Calibrate on the target hardware, respect affinity/quota, and
record the selected layout. Linux CPU affinity is required by the execution
supervisors. A different machine need not reproduce wall-clock time or bitwise
floating-point outputs. Resumption of a run requires its recorded environment,
source/data identities and resource layout to remain compatible.

## Verification without benchmark inputs

```bash
poetry run pytest tests/experiments tests/test_experiment_cli.py \
  tests/test_experiment_legacy_cleanup.py tests/test_package_metadata.py --no-cov
```

These tests use synthetic temporary inputs, including small native model fits,
and require no private archive or full benchmark execution. Statistical-report
tests cover tied ranks, optimization directions, control selection, tie-corrected
Friedman, Holm adjustment, missing evidence and artifact integrity.

The current benchmark pipeline is separate from the retained legacy
`run-all`, `run-phase`, `plot-all` and `summarize-paper` commands. Those commands
operate their original contracts; they do not run the new five-formulation
comparison. See [the workflow guide](../experiments.md) for completion scope
and evaluation blocks that still need separate execution.
