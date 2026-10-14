# Sources and implementation map

Repository revision: `f31d550b2e66d8459f143e0f39eaa20aabc028b9`.

This figure describes **Study 1**. Phase numbers are scientific questions, not individual commands or a literal scheduler DAG. The real-data study and prediction costs are deliberately outside the five numbered phases.

- [latest_commit](https://github.com/AdrianSeguraOrtiz/AMIGA/commit/f31d550b2e66d8459f143e0f39eaa20aabc028b9): Update AMIGA experimental workflow and documentation. Git object: `f31d550b2e66d8459f143e0f39eaa20aabc028b9`.
- [docs/experiments/design.md](https://github.com/AdrianSeguraOrtiz/AMIGA/blob/f31d550b2e66d8459f143e0f39eaa20aabc028b9/docs/experiments/design.md): Five phases, data units, nested evaluation, formulation scope and statistical interpretation. Git object: `463454f06d6412b40dcc8ad9e482c9dac941d1df`.
- [docs/experiments/benchmark.md](https://github.com/AdrianSeguraOrtiz/AMIGA/blob/f31d550b2e66d8459f143e0f39eaa20aabc028b9/docs/experiments/benchmark.md): Five-phase organization; fixed-recipe learning curves and 450-fit scope. Git object: `d79c92ddde0cd536f07dd2e74743d542148e58b5`.
- [docs/experiments/sequential-selection.md](https://github.com/AdrianSeguraOrtiz/AMIGA/blob/f31d550b2e66d8459f143e0f39eaa20aabc028b9/docs/experiments/sequential-selection.md): Training-only labels, full grids, training TreeSHAP and selection policies. Git object: `f8c2a576863ebab7e9a87a30c604ccb13dc6ef23`.
- [docs/experiments/outer-evaluation.md](https://github.com/AdrianSeguraOrtiz/AMIGA/blob/f31d550b2e66d8459f143e0f39eaa20aabc028b9/docs/experiments/outer-evaluation.md): Outer refits, mask relearning, score-before-label evaluation and seed aggregation. Git object: `39379d0920b04f33c6c017536dff7477b3d9ff1d`.
- [docs/experiments/supplementary.md](https://github.com/AdrianSeguraOrtiz/AMIGA/blob/f31d550b2e66d8459f143e0f39eaa20aabc028b9/docs/experiments/supplementary.md): Shared orchestration is not a sixth scientific phase. Git object: `9ba12bb848de3e7f10a93b6039f9c5c2a866716c`.
- [scripts/experiments/amiga_exp/sequential_selection/spec.py](https://github.com/AdrianSeguraOrtiz/AMIGA/blob/f31d550b2e66d8459f143e0f39eaa20aabc028b9/scripts/experiments/amiga_exp/sequential_selection/spec.py): Executable grid definitions and phase 1–3 job dependencies. Git object: `c87961b83ae99d7ef2cfb50515d37d958ed1a153`.
- [scripts/experiments/amiga_exp/sequential_selection/models.py](https://github.com/AdrianSeguraOrtiz/AMIGA/blob/f31d550b2e66d8459f143e0f39eaa20aabc028b9/scripts/experiments/amiga_exp/sequential_selection/models.py): Eight label schemes; all three families; training-only recursive TreeSHAP. Git object: `6528e66c7e9477f4f05db5febd0cbeb5ae1828c5`.
- [scripts/experiments/amiga_exp/supplementary/spec.py](https://github.com/AdrianSeguraOrtiz/AMIGA/blob/f31d550b2e66d8459f143e0f39eaa20aabc028b9/scripts/experiments/amiga_exp/supplementary/spec.py): Exact phase-4 recipe/column reuse; AMIGA-only fixed learning contexts. Git object: `63087c99f84298dd59696dee44e55d15143c03f5`.

## Reading boundaries

The current design and benchmark documentation were read in full. The executable selection specification, model adapters/feature-path definitions (lines 1–180), and fixed-supplement specification were also inspected. Historical result archives and unpublished run directories were not used. Rendering the diagram requires no inference, model training, benchmark metric recomputation or AMIGA experiment execution.

## Simplifications made in the drawing

- A miniature network card denotes a topology group; all recorded conditions of that group remain in one partition. A front is the collection of alternative candidate consensus networks for one benchmark condition.
- The two cases contain 104 fronts each from the same 87 topology groups. They are two generator-specific analyses, not 208 independent biological networks.
- Five outer partitions and three inner folds are drawn as a small fold matrix. It is a grouping diagram, not the observed allocation of individual topologies.
- Phase 1 applies to learning-to-rank. Regression and the classification formulations have fixed targets and bypass the relevance-label search; their own parameter, family and column searches follow the same phase-2/3 policy.
- The five-candidate rank-label example uses relevances 4/3/2/1/0, matching the implemented dense/average labels for distinct quality values. Higher relevance means better candidate quality.
- The phase-4 panel shows AMIGA, direct AUPR regression and classification at top-5%, top-10% and top-20% without asterisks or separate implementation-status labels. The classifiers use the same parameter and feature-selection policy; their methodological interpretation is documented in the linked experimental design.
- Only the domain products (27/36/36) are numeric counts in the parameter diagram. The miniature grids are icons, not one tile per tested setting.
- Retained feature percentages are fractions of the original column set. The elimination path is training-only, but inner validation selects the retained fraction and family. Masks must be relearned on the outer training complement, not copied from an inner fit.
- The feature path refits the model and recalculates training-only importance before each subsequent reduction; it does not rank all columns once and reuse that ordering throughout.
- Phase 4 averages seed-specific metrics, not prediction scores. The five final training seeds apply to the supervised estimators, not to deterministic objective-only rules.
- The quality bars explain the definition of Regret@5. Their lengths are illustrative and are not measured AUPR values. Score ties are handled by the exact expectation in the experiment implementation.
- The phase-5 axis deliberately contains a question mark and no measured y-values or invented trend. Its full-size point refers to existing phase-4 predictions.
- Phase 5 keeps exact phase-4 columns and procedures. It is conditional data-size sensitivity, not reselection under scarce labels and not another supervised-method comparison.
- The aggregation footer distinguishes phase 4 from phase 5. Phase 4 averages seed-specific metrics within fronts, then conditions within topology. Phase 5 first averages seeds within each subset/front, then the three subset draws within each front, then conditions within topology. Topologies receive equal final weight in both phases.
