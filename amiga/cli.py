# cli.py
"""
CLI for training, ranking, and feature extraction.

Each command delegates core logic to `amiga.core.main` and focuses on I/O:
reading CSV/PKL/JSON files and persisting artifacts (models, metrics, rankings).
"""

from pathlib import Path
from typing import List, Optional
import pandas as pd
import typer

from amiga.selection.learn2rank import (
    CV_REPORT, FEATURES_META, LabelMode, ModelType, MODEL_PREFIX
)
from amiga.utils import load_expression_matrix, load_json, load_pickle, clean, save_json, save_pickle

from amiga.core.main import (
    train_ltr_cv, train_ltr_full, rank_with_model, extract_expression_features, extract_grnet_features, build_data,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Tools to:\n"
        "  • Train a Learn-to-Rank model with group-based cross-validation.\n"
        "  • Produce per-front rankings using a trained model.\n"
        "  • Extract features from gene expression matrices or weighted GRNs.\n"
    ),
)


@app.command(
    name="train-cv",
    help=(
        "Train a Learn-to-Rank (LTR) model using GroupKFold by 'fronts' and save:\n"
        "  • Per-fold models (*.pkl)\n"
        "  • Per-fold validation rankings (valid_foldX_ranked.csv)\n"
        "  • Cross-validation report (cv_report.json) and feature columns metadata (feature_columns.json)\n"
    ),
)
def train_cv(
    csv_path: Path = typer.Argument(
        ...,
        help=(
            "Path to the training CSV. Must contain control columns and features. "
            "Defaults expected: front_id, AUPR, item_id + feature columns."
        ),
    ),
    model_type: ModelType = typer.Option(
        ModelType.LGBMRanker, "--model", "-m",
        help="Ranker type to train (enum)."
    ),
    front_col: str = typer.Option(
        "front_id",
        help="Name of the column identifying the group/front (used as groups for GroupKFold)."
    ),
    target_col: str = typer.Option(
        "AUPR",
        help="Target column (quality metric to be transformed into labels)."
    ),
    id_col: str = typer.Option(
        "item_id",
        help="Identifier column for the individual (not used as a feature)."
    ),
    drop_cols: List[str] = typer.Option(
        [],
        help="Additional columns to exclude from the feature set (on top of front/target/id)."
    ),
    label_mode: LabelMode = typer.Option(
        LabelMode.RANK_DENSE, "--label-mode",
        case_sensitive=False,
        help="Labeling strategy for LTR (e.g., RANK_DENSE, QUANTILES, CONTINUOUS...)."
    ),
    label_quantiles: int = typer.Option(
        20, min=2,
        help="Number of quantiles Q if label_mode=QUANTILES (ignored in other modes)."
    ),
    n_splits: int = typer.Option(
        5, min=2,
        help="Number of folds for GroupKFold (splits by 'front_col')."
    ),
    random_state: int = typer.Option(
        42,
        help="Base seed for reproducibility (offset per fold)."
    ),
    out_dir: Path = typer.Option(
        Path("./output"),
        help="Output directory where models, rankings, and reports will be stored."
    ),
):
    """
    Workflow:
      1) Read the training CSV.
      2) Build labels according to `label_mode`.
      3) Run GroupKFold and train a model per fold.
      4) Persist artifacts to `out_dir`.

    Files produced:
      - {MODEL_PREFIX}{i}.pkl: Trained model for fold i.
      - valid_fold{i}_ranked.csv: Validation predictions and per-front ranking for fold i.
      - cv_report.json: Fold-level metrics/metadata.
      - feature_columns.json: Ordered list of feature columns used during training.
    """
    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    df = pd.read_csv(csv_path)

    # Delegate training to core
    res = train_ltr_cv(
        df,
        model_type=model_type,
        front_col=front_col,
        target_col=target_col,
        id_col=id_col,
        drop_cols=drop_cols,
        label_mode=label_mode,
        label_quantiles=label_quantiles,
        n_splits=n_splits,
        random_state=random_state,
    )

    # Persist per-fold models
    for i, model in enumerate(res.models, start=1):
        save_pickle(model, out_dir / f"{MODEL_PREFIX}{i}.pkl")

    # Persist per-fold validation rankings (sorted within fronts by score desc)
    for i, df_va in enumerate(res.valid_folds, start=1):
        df_va.sort_values([front_col, "score"], ascending=[True, False]).to_csv(
            out_dir / f"valid_fold{i}_ranked.csv", index=False
        )

    # Persist CV report and feature metadata
    fold_reports = [fr.__dict__ for fr in res.fold_reports]
    save_json(fold_reports, out_dir / CV_REPORT)
    save_json({"feature_columns": res.feature_columns}, out_dir / FEATURES_META)

    typer.secho(f"Training completed. Artifacts stored at: {out_dir}", fg=typer.colors.CYAN)


@app.command(
    name="train-full",
    help=(
        "Train a single LTR model on the ENTIRE CSV (production). "
        "Saves model.pkl, feature_columns.json, and model_meta.json."
    ),
)
def train_full(
    csv_path: Path = typer.Argument(..., help="Training CSV with control columns and features."),
    model_type: ModelType = typer.Option(ModelType.LGBMRanker, "--model", "-m"),
    front_col: str = typer.Option("front_id"),
    target_col: str = typer.Option("AUPR"),
    id_col: str = typer.Option("item_id"),
    drop_cols: List[str] = typer.Option([]),
    label_mode: LabelMode = typer.Option(LabelMode.RANK_DENSE, "--label-mode", case_sensitive=False),
    label_quantiles: int = typer.Option(20, min=2),
    random_state: int = typer.Option(42),
    out_dir: Path = typer.Option(Path("./output_full")),
):
    """
    Train a production model using all available rows.

    Inputs
    ------
    csv_path : Path
        Path to the training CSV with columns [front_col, target_col, id_col] + features.
    model_type : ModelType
        Ranker to be trained.
    front_col, target_col, id_col : str
        Control columns. 'id_col' is not used as a feature.
    drop_cols : List[str]
        Additional columns to exclude from features.
    label_mode : LabelMode
        Strategy to transform the target metric into LTR labels.
    label_quantiles : int
        Number of bins if using QUANTILES.
    random_state : int
        Seed for reproducibility.
    out_dir : Path
        Output directory for persisted artifacts.

    Outputs
    -------
    - model.pkl
    - feature_columns.json
    - model_meta.json
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    # Delegate training to core
    fit = train_ltr_full(
        df=df,
        model_type=model_type,
        front_col=front_col,
        target_col=target_col,
        id_col=id_col,
        drop_cols=drop_cols,
        label_mode=label_mode,
        label_quantiles=label_quantiles,
        random_state=random_state,
    )

    # Persist artifacts
    save_pickle(fit.model, out_dir / "model.pkl")
    save_json({"feature_columns": fit.feature_columns}, out_dir / "feature_columns.json")
    save_json({
        "label_mode": fit.label_mode,
        "label_quantiles": fit.label_quantiles,
        **fit.metadata
    }, out_dir / "model_meta.json")

    typer.secho(f"Production model saved to: {out_dir}", fg=typer.colors.CYAN)


@app.command(
    name="rank-csv",
    help=(
        "Generate a per-front ranking using a trained model (.pkl).\n"
        "If 'feature_columns.json' is present next to the model, the feature order from training is respected."
    ),
)
def rank_csv(
    csv_path: Path = typer.Argument(
        ...,
        help=(
            "Path to the inference CSV (fronts without gold labels). Must contain the same feature columns used in training."
        ),
    ),
    model_path: Path = typer.Argument(
        ...,
        help="Path to the trained model (*.pkl). If present, 'feature_columns.json' from the same folder will be used."
    ),
    front_col: str = typer.Option(
        "front_id",
        help="Name of the column identifying the group/front to rank within."
    ),
    id_col: str = typer.Option(
        "item_id",
        help="Identifier column for the individual (not used as a feature)."
    ),
    drop_cols: List[str] = typer.Option(
        [],
        help="Additional columns to exclude from the feature set (on top of front/target/id)."
    ),
    out_csv: Path = typer.Option(
        Path("ranked_output.csv"),
        help="Output CSV path with original columns + score + rank_in_front."
    ),
):
    """
    Workflow:
      1) Load the model and (optionally) 'feature_columns.json' metadata.
      2) Read the inference CSV.
      3) Predict 'score', compute 'rank_in_front', and save the ranked CSV.

    Notes
    -----
    - 'rank_in_front' is computed within each group defined by `front_col`.
    - If feature metadata is available, columns are aligned to the original training order.
    """
    # Load model and optional feature order
    model = load_pickle(model_path)
    meta_path = model_path.parent / "feature_columns.json"
    feature_columns_hint: Optional[List[str]] = None
    if meta_path.exists():
        feature_columns_hint = list(load_json(meta_path).get("feature_columns", []))

    # Load inference data
    df = pd.read_csv(csv_path)

    # Delegate inference/ranking to core
    res = rank_with_model(
        df,
        model=model,
        front_col=front_col,
        id_col=id_col,
        drop_cols=drop_cols,
        feature_columns_hint=feature_columns_hint,
    )

    # Persist ranked output
    res.df_ranked.to_csv(out_csv, index=False)
    typer.secho(f"Ranking saved to: {out_csv}", fg=typer.colors.GREEN)


@app.command(
    help=(
        "Extract features from a gene expression matrix (rows=genes, columns=conditions/timepoints). "
        "Supports toggling groups of metrics (global, per-gene, per-condition, correlations, PCA, time-series)."
    )
)
def extract_expr_features(
    csv_path: Path = typer.Argument(
        ...,
        help="Path to the expression CSV. By default, the first column is treated as index (gene identifiers)."
    ),
    output_json: Path = typer.Option(
        ..., "--out", "-o",
        help="Path to the output JSON containing a flat dictionary of metrics."
    ),
    include_global: bool = typer.Option(
        True, help="Include global matrix size/statistics."
    ),
    include_gene_stats: bool = typer.Option(
        True, help="Include aggregated statistics per gene."
    ),
    include_condition_stats: bool = typer.Option(
        True, help="Include aggregated statistics per condition."
    ),
    include_correlations: bool = typer.Option(
        True, help="Include gene–gene and condition–condition correlations (optional sampling)."
    ),
    include_pca: bool = typer.Option(
        True, help="Include PCA (SVD) and explained-variance ratios (top_k)."
    ),
    include_timeseries: bool = typer.Option(
        False, help="Include temporal metrics per gene (assumes columns are ordered in time)."
    ),
    corr_max_pairs: Optional[int] = typer.Option(
        None, help="Max number of sampled pairs per correlation family to speed up (None = no limit)."
    ),
    pca_top_k: int = typer.Option(
        5, min=1, help="Number of principal components to report (PCA)."
    ),
    pca_center_by: str = typer.Option(
        "condition",
        help="Centering prior to PCA: 'condition' to center by column, 'gene' to center by row."
    ),
):
    """
    Steps:
      1) Load the expression matrix (optionally applies standard pre-processing inside core).
      2) Compute the selected groups of features.
      3) Save a flattened dictionary of metrics to JSON.
    """
    # Read expression matrix (utils handles indexing/cleaning)
    df_expr = load_expression_matrix(csv_path=csv_path)

    # Delegate feature extraction to core
    res = extract_expression_features(
        df_expr=df_expr,
        include_global=include_global,
        include_gene_stats=include_gene_stats,
        include_condition_stats=include_condition_stats,
        include_correlations=include_correlations,
        include_pca=include_pca,
        include_timeseries=include_timeseries,
        corr_max_pairs=corr_max_pairs,
        pca_top_k=pca_top_k,
        pca_center_by=pca_center_by
    )

    # Persist metrics
    output_json.parent.mkdir(parents=True, exist_ok=True)
    save_json(clean(res.metrics), output_json)
    typer.secho(f"✔ Metrics saved to: {output_json}", fg=typer.colors.GREEN)


@app.command(
    help=(
        "Extract WEIGHTED features from a directed GRN CSV with columns [Source, Target, Confidence]. "
        "Centralities and motifs are intentionally EXCLUDED."
    )
)
def extract_grn_features(
    csv_path: Path = typer.Argument(
        ..., help="Path to the CSV with columns in order: Source,Target,Confidence (no header)."
    ),
    output_json: Path = typer.Option(
        ..., "--out", "-o",
        help="Path to the output JSON containing a flat dictionary of metrics."
    ),
    include_global: bool = typer.Option(
        True, help="Include weighted global metrics (density, weight distribution, top-X concentration...)."
    ),
    include_strength: bool = typer.Option(
        True, help="Include in/out strength metrics (weighted sum of incoming/outgoing edges)."
    ),
    include_assortativity: bool = typer.Option(
        True, help="Include weighted out→in assortativity."
    ),
    include_paths: bool = typer.Option(
        True, help="Include path-based metrics (average distance, weighted p95 diameter)."
    ),
    include_clustering: bool = typer.Option(
        True, help="Include weighted average clustering (treated as undirected for the coefficient)."
    ),
    include_communities: bool = typer.Option(
        True, help="Include community detection (Louvain) on weighted graph."
    ),
    include_reciprocity: bool = typer.Option(
        True, help="Include weighted reciprocity."
    ),
    include_advanced: bool = typer.Option(
        False, help="Include advanced metrics (e.g., entropies)."
    ),
    top_frac: float = typer.Option(
        0.10, min=0.01, max=0.5,
        help="Fraction used for weight_topX_ratio (e.g., 0.10 = top 10% edges by weight)."
    ),
):
    """
    Steps:
      1) Read edges CSV (no header) and build a weighted directed graph.
      2) Compute the selected groups of features.
      3) Save a flattened dictionary of metrics to JSON.

    Notes
    -----
    - Input format is strict: three columns (Source, Target, Confidence) with no header row.
    - Centrality and motif-based features are purposely excluded to avoid overlap with optimization objectives.
    """
    # Load edges with explicit column names
    df_edges = pd.read_csv(csv_path, header=None, names=["Source", "Target", "Confidence"])

    # Delegate feature extraction to core
    res = extract_grnet_features(
        df_edges=df_edges,
        include_global=include_global,
        include_strength=include_strength,
        include_assortativity=include_assortativity,
        include_paths=include_paths,
        include_clustering=include_clustering,
        include_communities=include_communities,
        include_reciprocity=include_reciprocity,
        include_advanced=include_advanced,
        top_frac=top_frac,
    )

    # Persist metrics
    output_json.parent.mkdir(parents=True, exist_ok=True)
    save_json(clean(res.metrics), output_json)
    typer.secho(f"✔ Metrics saved to: {output_json}", fg=typer.colors.GREEN)


@app.command(
    name="build-data",
    help=(
        "Build a training dataset by combining: "
        "an evaluated front (includes AUPR, objective levels, and per-technique weights for GRN_*.csv), "
        "an expression matrix, and a folder with GRN_*.csv files. "
        "Copies all columns from the front except those in 'drop', and appends expr_*/grn_* features."
    ),
)
def build_data_cmd(
    evaluated_front_csv: Path = typer.Argument(..., help="CSV of the evaluated front (must include AUPR and GRN_*.csv weights)."),
    expression_csv: Path = typer.Argument(..., help="CSV of the expression matrix (genes x conditions)."),
    grn_folder: Path = typer.Argument(..., help="Folder containing GRN_*.csv files (one network per technique)."),
    front_id: int = typer.Option(..., "--front-id", help="Numeric identifier to assign to the front."),
    out_csv: Path = typer.Option(
        "data_front.csv", "--out", help="Path to the output CSV (similar to demo_data)."
    ),
    drop_front_cols: List[str] = typer.Option(
        ["Accuracy Mean", "AUROC"],
        help="Front columns to exclude from the output. Default: ['Accuracy Mean','AUROC']",
    ),
):
    """
    Steps:
      1) Read the evaluated front and expression matrix.
      2) For each row in the front, build the consensus network as required by the core logic, and
         compute expression and GRN feature blocks.
      3) Build the output row by copying front columns (except drop_front_cols) and appending expr_*/grn_* metrics.
      4) Persist the final dataset to CSV.

    Inputs
    ------
    evaluated_front_csv : Path
        Front CSV with AUPR, objective levels, and per-technique weights.
    expression_csv : Path
        Expression matrix (rows=genes, columns=conditions/timepoints).
    grn_folder : Path
        Directory with GRN_*.csv files to be combined/weighted.
    front_id : int
        Numeric identifier assigned to all resulting rows (useful for grouping).
    out_csv : Path
        Output CSV path for the consolidated dataset.
    drop_front_cols : List[str]
        Front columns that will be excluded from the output.

    Output
    ------
    CSV with all preserved front columns + expr_* and grn_* feature columns.
    """
    # Load front and expression data
    df_front = pd.read_csv(evaluated_front_csv)
    df_expr = load_expression_matrix(expression_csv)

    # Delegate dataset building to core
    df_out = build_data(
        df_front=df_front,
        df_expr=df_expr,
        grn_dir=grn_folder,
        front_id=front_id,
        drop_front_cols=drop_front_cols,
    )

    # Persist final dataset
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    typer.secho(f"✔ Dataset built and saved to: {out_csv}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
