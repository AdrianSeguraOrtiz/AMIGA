# Experimental workflow releases

The experimental workflow has its own version and Git tag. Its version does
not change the independently distributed `amiga-grn` PyPI package version.

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
