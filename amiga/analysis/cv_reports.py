from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from amiga.analysis.stat_tests import compute_global_metric_stats

METRIC_TIER: Dict[str, str] = {
    "Regret@1": "primary",
    "Regret@3": "primary",
    "Regret@5": "primary",
    "Regret@10": "primary",
    "BestAUPR@1": "primary",
    "BestAUPR@3": "primary",
    "BestAUPR@5": "primary",
    "BestAUPR@10": "primary",
    "Hit@1": "primary",
    "Hit@3": "primary",
    "Hit@5": "primary",
    "Hit@10": "primary",
    "NDCG@1": "secondary",
    "NDCG@3": "secondary",
    "NDCG@5": "secondary",
    "NDCG@10": "secondary",
    "Spearman": "tertiary",
    "KendallTau": "tertiary",
    "n_items": "aux",
}

METRIC_ORDER: Dict[str, int] = {
    "Regret@1": 10,
    "Regret@3": 11,
    "Regret@5": 12,
    "Regret@10": 13,
    "BestAUPR@1": 20,
    "BestAUPR@3": 21,
    "BestAUPR@5": 22,
    "BestAUPR@10": 23,
    "Hit@1": 30,
    "Hit@3": 31,
    "Hit@5": 32,
    "Hit@10": 33,
    "NDCG@1": 40,
    "NDCG@3": 41,
    "NDCG@5": 42,
    "NDCG@10": 43,
    "Spearman": 50,
    "KendallTau": 51,
    "n_items": 99,
}

def metric_order(metric: str) -> int:
    return METRIC_ORDER.get(metric, 1000)


def metric_tier(metric: str) -> str:
    return METRIC_TIER.get(metric, "other")


def is_lower_better(metric: str) -> bool:
    return metric.startswith("Regret@")


def sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in value)


def build_identity_label_map(model_names: Sequence[str]) -> Dict[str, str]:
    return {model_name: model_name for model_name in model_names}


def infer_run_name(path: Path) -> str:
    if path.name == "cv_report.json" and path.parent.name:
        return path.parent.name
    return path.stem


def apply_plot_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        palette="colorblind",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 18,
            "grid.alpha": 0.2,
        },
    )


def build_model_palette(model_names: Sequence[str]) -> Dict[str, tuple[float, float, float]]:
    unique_names = list(dict.fromkeys(model_names))
    colors = sns.color_palette("colorblind", n_colors=max(1, len(unique_names)))
    return {model_name: colors[idx] for idx, model_name in enumerate(unique_names)}


def load_report_payload(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of folds, got {type(data)}")
    return data


def load_report(path: Path) -> pd.DataFrame:
    data = load_report_payload(path)

    rows = []
    for item in data:
        if not isinstance(item, dict) or "fold" not in item or "agg" not in item:
            raise ValueError(f"{path}: each element must contain 'fold' and 'agg'")
        agg = item["agg"] or {}
        if not isinstance(agg, dict):
            raise ValueError(f"{path}: agg must be a dictionary")
        rows.append({"fold": int(item["fold"]), **agg})

    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def load_report_group_metrics(path: Path, *, front_col: str = "front_id") -> pd.DataFrame:
    data = load_report_payload(path)
    rows = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError(f"{path}: each fold entry must be a dictionary.")
        fold = item.get("fold")
        groups = item.get("groups")
        if groups is None:
            raise ValueError(
                f"{path}: metric_rank_stats requires per-front 'groups' data in cv_report.json."
            )
        if not isinstance(groups, list):
            raise ValueError(f"{path}: groups must be a list.")
        for group in groups:
            if not isinstance(group, dict):
                raise ValueError(f"{path}: each group entry must be a dictionary.")
            if front_col not in group:
                raise ValueError(f"{path}: group entry is missing required key '{front_col}'.")
            rows.append({"fold": int(fold), **group})
    if not rows:
        raise ValueError(f"{path}: no per-front group metrics were found in cv_report.json.")
    return pd.DataFrame(rows).sort_values(["fold", front_col]).reset_index(drop=True)


def tidy_long(
    dfs: Dict[str, pd.DataFrame],
    metrics: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for model_name, df in dfs.items():
        cols = [col for col in df.columns if col != "fold"]
        if metrics is not None:
            cols = [col for col in cols if col in metrics]
        for metric in cols:
            rows.append(
                pd.DataFrame(
                    {
                        "model": model_name,
                        "fold": df["fold"].astype(int),
                        "metric": metric,
                        "value": pd.to_numeric(df[metric], errors="coerce"),
                    }
                )
            )
    if not rows:
        return pd.DataFrame(columns=["model", "fold", "metric", "value", "tier", "priority"])

    long_df = pd.concat(rows, ignore_index=True)
    long_df["tier"] = long_df["metric"].map(metric_tier)
    long_df["priority"] = long_df["metric"].map(metric_order)
    return long_df


def summarize(long_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = (
        long_df.dropna(subset=["value"])
        .groupby(["model", "metric", "tier", "priority"], as_index=False)
        .agg(mean=("value", "mean"), std=("value", "std"), n=("value", "count"))
        .sort_values(["priority", "model"])
        .reset_index(drop=True)
    )
    return summary_df


def build_metric_ranks(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for metric, metric_df in summary_df.groupby("metric", sort=False):
        ascending = is_lower_better(metric)
        metric_df = metric_df.sort_values(["mean", "model"], ascending=[ascending, True]).reset_index(drop=True)
        metric_df = metric_df.copy()
        metric_df["rank"] = np.arange(1, len(metric_df) + 1)
        rows.append(metric_df)
    if not rows:
        return pd.DataFrame(columns=["model", "metric", "tier", "priority", "mean", "std", "n", "rank"])
    return pd.concat(rows, ignore_index=True)


def format_rank_value(value: float) -> str:
    return f"r={value:.2f}"


def format_pvalue(value: float) -> str:
    if pd.isna(value):
        return "winner"
    if value < 0.001:
        return "p<0.001"
    return f"p={value:.3f}"


def build_stat_heatmap_pivots(
    stats_df: pd.DataFrame,
    *,
    metric_cols: Sequence[str],
    row_order: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pivot_rank = stats_df.pivot(index="model", columns="metric", values="avg_rank").reindex(index=row_order, columns=metric_cols)
    pivot_p = stats_df.pivot(index="model", columns="metric", values="holm_p_adj").reindex(index=row_order, columns=metric_cols)
    pivot_winner = stats_df.pivot(index="model", columns="metric", values="is_winner").reindex(index=row_order, columns=metric_cols)
    return pivot_rank, pivot_p, pivot_winner


def save_dotplots(
    summary_df: pd.DataFrame,
    out_path: Path,
    *,
    label_map: Optional[Mapping[str, str]] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        return
    apply_plot_theme()
    if label_map is None:
        label_map = build_identity_label_map(summary_df["model"].unique())
    k_color_map = {
        1: "#1f77b4",
        3: "#ff7f0e",
        5: "#2ca02c",
        10: "#d62728",
    }
    model_palette = build_model_palette(summary_df["model"].unique())
    grouped_specs = [
        ("Regret", summary_df["metric"].str.startswith("Regret@")),
        ("BestAUPR", summary_df["metric"].str.startswith("BestAUPR@")),
        ("Hit", summary_df["metric"].str.startswith("Hit@")),
        ("NDCG", summary_df["metric"].str.startswith("NDCG@")),
        ("Spearman", summary_df["metric"] == "Spearman"),
        ("KendallTau", summary_df["metric"] == "KendallTau"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(18, 18), squeeze=False)
    legend_handles = {}

    for ax, (title, mask) in zip(axes.flat, grouped_specs):
        group_df = summary_df[mask].copy()
        if group_df.empty:
            ax.axis("off")
            continue

        if title in {"Spearman", "KendallTau"}:
            metric_df = group_df.sort_values("mean", ascending=False)
            metric_df["label"] = metric_df["model"].map(label_map)
            sns.scatterplot(
                data=metric_df,
                x="mean",
                y="label",
                hue="model",
                palette=model_palette,
                legend=False,
                s=65,
                ax=ax,
            )
            for _, row in metric_df.iterrows():
                ax.errorbar(
                    x=row["mean"],
                    y=row["label"],
                    xerr=float(0.0 if pd.isna(row["std"]) else row["std"]),
                    fmt="none",
                    ecolor=model_palette[row["model"]],
                    alpha=0.85,
                    capsize=3,
                    linewidth=1.25,
                )
            legend_handles[title] = plt.Line2D([], [], marker="o", linestyle="", color="#444444")
        else:
            group_df["k"] = group_df["metric"].map(lambda metric: _extract_k(metric, title + "@"))
            model_order = (
                group_df.groupby("model", as_index=False)["mean"]
                .mean()
                .sort_values("mean", ascending=title == "Regret")
                ["model"]
                .tolist()
            )
            y_base = {model: idx for idx, model in enumerate(model_order)}
            k_values = [k for k in sorted(group_df["k"].dropna().unique()) if pd.notna(k)]
            offsets = np.linspace(-0.24, 0.24, num=max(1, len(k_values)))

            for offset, k in zip(offsets, k_values):
                k_df = group_df[group_df["k"] == k].copy()
                k_df["y"] = k_df["model"].map(y_base).astype(float) + offset
                color = k_color_map.get(int(k), "#444444")
                sns.scatterplot(
                    data=k_df,
                    x="mean",
                    y="y",
                    color=color,
                    s=55,
                    legend=False,
                    ax=ax,
                )
                for _, row in k_df.iterrows():
                    ax.errorbar(
                        x=row["mean"],
                        y=row["y"],
                        xerr=float(0.0 if pd.isna(row["std"]) else row["std"]),
                        fmt="none",
                        ecolor=color,
                        alpha=0.85,
                        capsize=3,
                        linewidth=1.2,
                    )
                legend_handles[f"@{int(k)}"] = plt.Line2D([], [], marker="o", linestyle="", color=color)

            ax.set_yticks(np.arange(len(model_order)))
            ax.set_yticklabels([label_map[name] for name in model_order], fontsize=8)

        ax.invert_yaxis()
        ax.set_title(title, fontweight="semibold")
        ax.grid(axis="x", alpha=0.18)
        sns.despine(ax=ax, left=False, bottom=False)

    fig.suptitle("Dotplot overview (mean ± std over folds)", fontweight="semibold")
    ordered_labels = ["@1", "@3", "@5", "@10", "Spearman", "KendallTau"]
    handles = [legend_handles[label] for label in ordered_labels if label in legend_handles]
    labels = [label for label in ordered_labels if label in legend_handles]
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=True)
        fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _extract_k(metric: str, prefix: str) -> Optional[int]:
    if not metric.startswith(prefix):
        return None
    try:
        return int(metric.split("@", 1)[1])
    except ValueError:
        return None


def save_topk_curves(
    summary_df: pd.DataFrame,
    out_dir: Path,
    *,
    label_map: Optional[Mapping[str, str]] = None,
    prefixes: Sequence[str] = ("Regret@", "Hit@", "BestAUPR@", "NDCG@"),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        return
    apply_plot_theme()
    if label_map is None:
        label_map = build_identity_label_map(summary_df["model"].unique())
    model_palette = build_model_palette(summary_df["model"].unique())
    for prefix, ylabel in (
        ("Regret@", "Regret"),
        ("Hit@", "Hit rate"),
        ("BestAUPR@", "Best AUPR recovered"),
        ("NDCG@", "NDCG"),
    ):
        if prefix not in prefixes:
            continue
        prefix_df = summary_df[summary_df["metric"].str.startswith(prefix)]
        if prefix_df.empty:
            continue
        prefix_df = prefix_df.copy()
        prefix_df["k"] = prefix_df["metric"].map(lambda metric: _extract_k(metric, prefix))
        prefix_df = prefix_df.dropna(subset=["k"])
        prefix_df["label"] = prefix_df["model"].map(label_map)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        sns.lineplot(
            data=prefix_df.sort_values(["model", "k"]),
            x="k",
            y="mean",
            hue="label",
            style="label",
            markers=True,
            dashes=False,
            palette={label_map[name]: color for name, color in model_palette.items()},
            linewidth=2.0,
            ax=ax,
        )
        ax.set_xlabel("k")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(prefix_df["k"].dropna().unique()))
        ax.set_title(f"{ylabel} across top-k", fontweight="semibold")
        ax.grid(alpha=0.18)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
        sns.despine(ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / f"topk_{sanitize(prefix[:-1].lower())}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def save_global_stat_rank_heatmap(
    global_stats_df: pd.DataFrame,
    out_path: Path,
    *,
    label_map: Optional[Mapping[str, str]] = None,
) -> None:
    if global_stats_df.empty:
        return
    apply_plot_theme()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if label_map is None:
        label_map = build_identity_label_map(global_stats_df["model"].unique())
    metric_cols = sorted(global_stats_df["metric"].dropna().unique(), key=metric_order)
    row_order = sorted(global_stats_df["model"].unique(), key=lambda model_name: label_map[model_name])
    pivot_rank, pivot_p, pivot_winner = build_stat_heatmap_pivots(
        global_stats_df,
        metric_cols=metric_cols,
        row_order=row_order,
    )

    fig_width = max(14, 0.9 * len(metric_cols))
    fig_height = max(6, 0.52 * len(row_order))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        pivot_rank,
        cmap="crest_r",
        vmin=1,
        vmax=max(1, len(pivot_rank.index)),
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Average rank"},
        ax=ax,
    )
    ax.set_title("Global Statistical Metric Rank Heatmap", fontweight="semibold")
    ax.set_xticklabels(pivot_rank.columns, rotation=35, ha="right", fontsize=11)
    ax.set_yticklabels([label_map[name] for name in pivot_rank.index], fontsize=9)
    for i in range(pivot_rank.shape[0]):
        for j in range(pivot_rank.shape[1]):
            rank_value = pivot_rank.iloc[i, j]
            if pd.isna(rank_value):
                continue
            p_value = pivot_p.iloc[i, j]
            is_winner = bool(pivot_winner.iloc[i, j]) if pd.notna(pivot_winner.iloc[i, j]) else False
            text_color = "black" if rank_value <= max(2, len(pivot_rank.index) / 2) else "white"
            stat_color = "#1f1f1f" if is_winner else ("#0b6e4f" if pd.isna(p_value) or p_value >= 0.05 else "#8b0000")
            ax.text(j + 0.5, i + 0.38, format_rank_value(float(rank_value)), ha="center", va="center", fontsize=9, color=text_color)
            ax.text(j + 0.5, i + 0.68, "winner" if is_winner else format_pvalue(float(p_value)), ha="center", va="center", fontsize=8, color=stat_color, fontweight="bold")
    sns.despine(ax=ax, left=False, bottom=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_metric_scatter(
    summary_df: pd.DataFrame,
    out_path: Path,
    *,
    x_metric: str,
    y_metric: str,
    label_map: Optional[Mapping[str, str]] = None,
) -> None:
    if x_metric == y_metric:
        raise ValueError("x_metric and y_metric must be different.")
    apply_plot_theme()

    metric_df = summary_df[summary_df["metric"].isin([x_metric, y_metric])].copy()
    if metric_df.empty:
        raise ValueError("Requested metrics are not present in metrics_summary.csv.")

    scatter_df = (
        metric_df[["model", "metric", "mean"]]
        .pivot(index="model", columns="metric", values="mean")
        .dropna(subset=[x_metric, y_metric])
        .reset_index()
    )
    if scatter_df.empty:
        raise ValueError("No model has both requested metrics available.")

    if label_map is None:
        label_map = build_identity_label_map(scatter_df["model"].tolist())
    scatter_df["label"] = scatter_df["model"].map(label_map)
    palette = build_model_palette(scatter_df["model"].tolist())

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=scatter_df,
        x=x_metric,
        y=y_metric,
        hue="label",
        palette={label_map[name]: color for name, color in palette.items()},
        s=90,
        legend=False,
        ax=ax,
    )

    for _, row in scatter_df.iterrows():
        ax.annotate(
            row["label"],
            (row[x_metric], row[y_metric]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title(f"{x_metric} vs {y_metric}", fontweight="semibold")
    ax.grid(alpha=0.18)
    sns.despine(ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


SummaryStatName = Literal["metric_rank_stats"]
PlotName = Literal["dotplot_overview", "topk_curves", "metric_rank_heatmap", "metric_scatter"]

ALL_STATS: Tuple[SummaryStatName, ...] = ("metric_rank_stats",)
ALL_PLOTS: Tuple[PlotName, ...] = (
    "dotplot_overview",
    "topk_curves",
    "metric_rank_heatmap",
    "metric_scatter",
)
TOPK_PREFIXES: Dict[str, str] = {
    "Regret": "Regret@",
    "Hit": "Hit@",
    "BestAUPR": "BestAUPR@",
    "NDCG": "NDCG@",
}


def normalize_stat_selection(stats: Optional[Sequence[str]]) -> List[SummaryStatName]:
    if not stats:
        return []
    invalid = [stat for stat in stats if stat not in ALL_STATS]
    if invalid:
        raise ValueError(
            "Unsupported stats name(s): "
            + ", ".join(sorted(invalid))
            + ". Valid names are: "
            + ", ".join(ALL_STATS)
        )
    return list(stats)  # type: ignore[return-value]


def normalize_plot_name(plot: str) -> PlotName:
    if plot not in ALL_PLOTS:
        raise ValueError(
            f"Unsupported plot name: {plot}. Valid names are: {', '.join(ALL_PLOTS)}"
        )
    return plot  # type: ignore[return-value]


def load_summary_table(input_dir: Path, filename: str) -> pd.DataFrame:
    path = input_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Required summary file not found: {path}")
    return pd.read_csv(path)


def resolve_plot_output_path(
    *,
    input_dir: Path,
    default_filename: str,
    out: Optional[Path] = None,
) -> Path:
    default_dir = input_dir / "plots"
    if out is None:
        return default_dir / default_filename
    if out.exists() and out.is_dir():
        return out / default_filename
    if out.suffix:
        return out
    return out / default_filename


def summarize_cv_reports(
    report_paths: Sequence[Path],
    out_dir: Path,
    *,
    metrics: Optional[Sequence[str]] = None,
    stats: Optional[Sequence[str]] = None,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stat_selection = normalize_stat_selection(stats)

    dfs: Dict[str, pd.DataFrame] = {}
    per_front_metrics: Dict[str, pd.DataFrame] = {}
    for report_path in report_paths:
        path = Path(report_path)
        run_name = infer_run_name(path)
        dfs[run_name] = load_report(path)
        if "metric_rank_stats" in stat_selection:
            per_front_metrics[run_name] = load_report_group_metrics(path)

    long_df = tidy_long(dfs, metrics=metrics)
    long_df = long_df.sort_values(["priority", "metric", "model", "fold"]).reset_index(drop=True)
    summary_df = summarize(long_df)
    rank_df = build_metric_ranks(summary_df)

    metrics_long_path = out_dir / "metrics_long.csv"
    metrics_summary_path = out_dir / "metrics_summary.csv"
    metric_ranks_path = out_dir / "metric_ranks.csv"
    long_df.to_csv(metrics_long_path, index=False)
    summary_df.to_csv(metrics_summary_path, index=False)
    rank_df.to_csv(metric_ranks_path, index=False)

    outputs: Dict[str, Path] = {
        "metrics_long": metrics_long_path,
        "metrics_summary": metrics_summary_path,
        "metric_ranks": metric_ranks_path,
    }

    if "metric_rank_stats" in stat_selection:
        metric_names = [metric for metric in summary_df["metric"].dropna().unique() if metric != "n_items"]
        metric_rank_stats_df = compute_global_metric_stats(per_front_metrics, metric_names) if metric_names else pd.DataFrame()
        metric_rank_stats_path = out_dir / "metric_rank_stats.csv"
        metric_rank_stats_df.to_csv(metric_rank_stats_path, index=False)
        outputs["metric_rank_stats"] = metric_rank_stats_path

    return outputs


def plot_cv_summary(
    input_dir: Path,
    *,
    plot: str,
    out: Optional[Path] = None,
    metrics: Optional[Sequence[str]] = None,
    metric_prefix: Optional[str] = None,
    x_metric: Optional[str] = None,
    y_metric: Optional[str] = None,
) -> Path:
    input_dir = Path(input_dir)
    plot_name = normalize_plot_name(plot)

    if plot_name == "dotplot_overview":
        summary_df = load_summary_table(input_dir, "metrics_summary.csv")
        if metrics is not None:
            summary_df = summary_df[summary_df["metric"].isin(metrics)].copy()
        if summary_df.empty:
            raise ValueError("No rows remain after applying the requested metric filter.")
        output_path = resolve_plot_output_path(
            input_dir=input_dir,
            default_filename="dotplot_overview.png",
            out=out,
        )
        save_dotplots(summary_df, output_path)
        return output_path

    if plot_name == "topk_curves":
        if metric_prefix is None:
            raise ValueError("plot 'topk_curves' requires --metric-prefix.")
        if metric_prefix not in TOPK_PREFIXES:
            raise ValueError(
                "Unsupported metric prefix: "
                f"{metric_prefix}. Valid values are: {', '.join(TOPK_PREFIXES)}"
            )
        prefix = TOPK_PREFIXES[metric_prefix]
        summary_df = load_summary_table(input_dir, "metrics_summary.csv")
        prefix_df = summary_df[summary_df["metric"].str.startswith(prefix)].copy()
        if prefix_df.empty:
            raise ValueError(f"No metrics with prefix '{metric_prefix}' are present in metrics_summary.csv.")
        output_path = resolve_plot_output_path(
            input_dir=input_dir,
            default_filename=f"topk_{sanitize(metric_prefix.lower())}.png",
            out=out,
        )
        save_topk_curves(prefix_df, output_path.parent, prefixes=(prefix,))
        generated = output_path.parent / f"topk_{sanitize(prefix[:-1].lower())}.png"
        if generated != output_path:
            generated.replace(output_path)
        return output_path

    if plot_name == "metric_rank_heatmap":
        if metrics is None or len(metrics) == 0:
            raise ValueError("plot 'metric_rank_heatmap' requires at least one metric via --metrics.")
        stats_df = load_summary_table(input_dir, "metric_rank_stats.csv")
        stats_df = stats_df[stats_df["metric"].isin(metrics)].copy()
        if stats_df.empty:
            raise ValueError("No rows remain after applying the requested metric filter.")
        output_path = resolve_plot_output_path(
            input_dir=input_dir,
            default_filename="metric_rank_heatmap.png",
            out=out,
        )
        save_global_stat_rank_heatmap(stats_df, output_path)
        return output_path

    if plot_name == "metric_scatter":
        if not x_metric or not y_metric:
            raise ValueError("plot 'metric_scatter' requires both --x-metric and --y-metric.")
        summary_df = load_summary_table(input_dir, "metrics_summary.csv")
        output_path = resolve_plot_output_path(
            input_dir=input_dir,
            default_filename=f"metric_scatter__{sanitize(x_metric)}__vs__{sanitize(y_metric)}.png",
            out=out,
        )
        save_metric_scatter(summary_df, output_path, x_metric=x_metric, y_metric=y_metric)
        return output_path

    raise ValueError(f"Unsupported plot name: {plot_name}")
