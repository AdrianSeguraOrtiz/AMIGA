# Experimental workflow releases

The experimental workflow has its own version and Git tag. Its version does
not change the independently distributed `amiga-grn` PyPI package version.

## 0.3.0 — amiga-exp-v0.3.0

- Current phase 1–4 figures, including recursive column-selection diagnostics
  and horizontal supervised/objective comparison panels.
- Full-grid learning-curve selection at 10, 20 and 40 labelled topologies, three
  nested subsamples, five outer folds and five final seeds; recorded full-size
  endpoint reused.
- Grouped benchmark selection and native deployment model export for all five
  supervised formulations; label-free TCGA-BRCA scoring and fixed-snapshot
  contextual regulatory support.
- Isolated feature-preparation profiling at 100/250/500 genes, recorded fitting
  costs and repeated model scoring measurements.
- Checksummed, portable data/result deposits with non-overwriting restoration;
  completion checks, progress monitoring and explicit technical resumption.

This source release provides the executable supplementary workflow. Its current
execution status and separately versioned artifact deposits are documented in
[`benchmark-artifacts/README.md`](../../benchmark-artifacts/README.md).
The `amiga` core and its PyPI version remain unchanged.

## 0.2.0 — amiga-exp-v0.2.0

- Topology-grouped five-fold outer and three-fold inner benchmark validation.
- Complete LightGBM, XGBoost and CatBoost relevance/parameter searches and
  recursive training-only TreeSHAP column selection.
- Five-seed held-out evaluation with objective-only decision comparators and
  independently selected supervised formulations.
- Full-budget top-5% and top-10% classification sensitivity pipelines.
- Audited `report supervised` CSV/LaTeX tables and PNG/PDF figures, fixed-control
  exploratory Friedman/Holm statistics and raw metric summaries.
- `--version`, a committed dependency lock, installation/data documentation and
  integrity/isolation tests.

The release preserves source identities needed to verify completed runs. Earlier
execution adapters and specifications remain available where current contracts
depend on them. It does not change the public `amiga` API, publish a new PyPI
wheel or provide completed learning-curve/deployment experiments for the current
selected procedure. Benchmark data and full outputs require separate distribution.
