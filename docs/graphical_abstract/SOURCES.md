# Figure-to-implementation map

Repository: https://github.com/AdrianSeguraOrtiz/AMIGA

Pinned revision: `1eaef45bee2d58807e8563b01eefee4724fe4b2f`.

The figure is based on the source revision above. Its integration changes only
documentation and figure assets; scientific runs and the `amiga` core are
unchanged. Drawing sources and rendered artifacts have their own hashes.

## S1 — Scope and end-to-end CLI workflow

https://github.com/AdrianSeguraOrtiz/AMIGA/blob/1eaef45bee2d58807e8563b01eefee4724fe4b2f/README.md

Sections: opening description, What AMIGA Uses, Quickstart, Build AMIGA Tables,
Train And Evaluate With Grouped CV, Train A Final Model, Rank A New Front,
Rankers And Labels, Research Workflow.

Supports AMIGA as a post-Pareto decision layer; upstream reference-based label
construction; feature tables; training/prediction separation; saved model and
feature schema; the distinction between the reusable core and the article
research protocol.

## S2 — Feature construction and supervised fitting

https://github.com/AdrianSeguraOrtiz/AMIGA/blob/1eaef45bee2d58807e8563b01eefee4724fe4b2f/amiga/core/main.py

Functions: `build_data`, `extract_expression_features`, `extract_grnet_features`,
`train_ltr_cv`, `train_ltr_full`, `rank_with_model`.

Supports preservation of mixture-weight/objective columns; reconstruction of a
candidate's weighted consensus via `weighted_confidence`; per-candidate GRN
features; once-per-front expression features; supervised-target and identifier
exclusion; `GroupKFold` by `front_id`; full-data refit; reuse of trained predictor
order; exported scores and within-front ranks.

## S3 — Labels, supported rankers and ranking output

https://github.com/AdrianSeguraOrtiz/AMIGA/blob/1eaef45bee2d58807e8563b01eefee4724fe4b2f/amiga/selection/learn2rank.py

Symbols: `ModelType`, `LabelMode`, `build_labels`, `assign_rank_in_front`.

Supports within-front relevance construction, LightGBM/XGBoost/CatBoost ranking
backends, descending-score ranks, and deterministic tie handling. The figure
does not imply that a score is a calibrated AUPR estimate or that the winning
candidate is proven correct.

## Deliberate visual simplifications

- Five schematic candidates show the idea of a front; five is not an AMIGA limit.
- Two abstract objectives illustrate trade-offs; the implementation is not
  restricted to two objectives.
- Three base-network cards stand for an arbitrary supplied collection.
- Each matrix colour denotes a feature family, not a fixed number of predictors.
- Grouped validation is conceptual, not a displayed fold count. The core groups
  by `front_id`; the current experimental workflow additionally keeps whole
  topology groups together. The figure's integrity note applies to both.
- The final selected graph is a use of the ranked table. `rank-csv` itself
  exports a table, not the recommendation illustration.
- No empirical dataset, learned model, benchmark metric or real regulatory edge
  is visualized. These are illustrative shapes authored in the figure code.
