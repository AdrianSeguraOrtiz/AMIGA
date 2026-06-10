"""Plot preparation helpers for AMIGA experiment-specific paper figures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import gaussian_kde

from scripts.experiments.amiga_exp.context import CaseContext
from scripts.experiments.amiga_exp.manifests import (
    ManifestWriteStatus,
    WriteOptions,
    common_manifest_fields,
    repo_relative,
    results_layout,
    write_manifest,
)
from scripts.experiments.amiga_exp.primary_rank_table import PRIMARY_RANK_TABLE_COLUMNS


PLOT_PHASES = (
    "01_model_screening",
    "02_hyperparameter_tuning",
    "03_ablation",
    "04_decision_baselines",
)

PHASE_PRIMARY_PLOT_FILES = {
    "01_model_screening": "model_screening_heatmap",
    "02_hyperparameter_tuning": "hyperparameter_regret_scatter",
    "03_ablation": "ablation_feature_matrix",
    "04_decision_baselines": "decision_baseline_rank",
}

SUPPLEMENTARY_METRIC_FAMILIES = ("Regret", "Hit", "BestAUPR")
HYPERPARAMETER_GROUP_COLORS = (
    "#FF6B6B",
    "#35B6E8",
    "#FDBA45",
    "#7A6FF0",
    "#8CC152",
    "#C77CFF",
)
CONTROL_LABEL_MODES = {"reversed", "shuffled"}
ABLATION_BLOCK_COLUMNS = (
    ("objectives", "Objectives"),
    ("technique_weights", "Technique weights"),
    ("expression", "Expression"),
    ("network", "Network"),
)
ABLATION_FEATURE_BLOCKS = {
    "full": {
        "objectives": True,
        "technique_weights": True,
        "expression": True,
        "network": True,
    },
    "no_objectives": {
        "objectives": False,
        "technique_weights": True,
        "expression": True,
        "network": True,
    },
    "no_technique_weights": {
        "objectives": True,
        "technique_weights": False,
        "expression": True,
        "network": True,
    },
    "no_expression": {
        "objectives": True,
        "technique_weights": True,
        "expression": False,
        "network": True,
    },
    "no_network": {
        "objectives": True,
        "technique_weights": True,
        "expression": True,
        "network": False,
    },
    "objectives_only": {
        "objectives": True,
        "technique_weights": False,
        "expression": False,
        "network": False,
    },
    "technique_weights_only": {
        "objectives": False,
        "technique_weights": True,
        "expression": False,
        "network": False,
    },
    "expression_only": {
        "objectives": False,
        "technique_weights": False,
        "expression": True,
        "network": False,
    },
    "network_only": {
        "objectives": False,
        "technique_weights": False,
        "expression": False,
        "network": True,
    },
}
SUPPLEMENTARY_POLICY = (
    "Complementary context only. These outputs must not change phase winners "
    "or selected configurations."
)


class PlotPreparationError(ValueError):
    """Raised when experiment plot artifacts cannot be prepared safely."""


@dataclass(frozen=True)
class PlotPhaseResult:
    """Prepared plotting artifacts for one phase."""

    phase: str
    phase_dir: Path
    plots_dir: Path
    primary_rank_table: Path
    plot_manifest: Path
    planned_outputs: dict[str, dict[str, Path]]
    generated_outputs: dict[str, dict[str, Path]]
    status: ManifestWriteStatus


@dataclass(frozen=True)
class PlotAllResult:
    """Prepared plotting artifacts for all plot-capable phases."""

    results: tuple[PlotPhaseResult, ...]


def prepare_phase_plots(
    context: CaseContext,
    phase: str,
    *,
    options: WriteOptions | None = None,
    include_secondary: bool = False,
) -> PlotPhaseResult:
    """Prepare the plot directory and manifest for one experimental phase."""
    options = options or WriteOptions()
    options.validate()
    if phase not in PLOT_PHASES:
        raise PlotPreparationError(
            f"unsupported plot phase '{phase}'. Valid phases are: {', '.join(PLOT_PHASES)}"
        )

    layout = results_layout(context)
    phase_dir = layout.phases[phase]
    summary_dir = phase_dir / "summary"
    plots_dir = phase_dir / "plots"
    primary_rank_table = summary_dir / "primary_rank_table.csv"
    metrics_summary = summary_dir / "metrics_summary.csv"
    statistical_tests = layout.summaries / "statistical_tests.csv"
    primary_table = load_primary_rank_table(primary_rank_table)
    planned_outputs = _planned_outputs(phase, plots_dir, include_secondary=include_secondary)

    if not options.dry_run:
        plots_dir.mkdir(parents=True, exist_ok=True)
    plot_manifest = plots_dir / "plot_manifest.json"
    _ensure_plot_manifest_writable(plot_manifest, options)
    generated_outputs = _generate_phase_outputs(
        phase,
        phase_dir=phase_dir,
        metrics_summary=metrics_summary,
        statistical_tests=statistical_tests,
        primary_table=primary_table,
        planned_outputs=planned_outputs,
        options=options,
        skip_generation=plot_manifest.exists() and options.skip_existing and not options.force,
    )
    payload = _plot_manifest_payload(
        context,
        phase=phase,
        plots_dir=plots_dir,
        primary_rank_table=primary_rank_table,
        statistical_tests=statistical_tests if phase == "03_ablation" else None,
        primary_table=primary_table,
        include_secondary=include_secondary,
        planned_outputs=planned_outputs,
        generated_outputs=generated_outputs,
    )
    status = write_manifest(plot_manifest, payload, options)
    return PlotPhaseResult(
        phase=phase,
        phase_dir=phase_dir,
        plots_dir=plots_dir,
        primary_rank_table=primary_rank_table,
        plot_manifest=plot_manifest,
        planned_outputs=planned_outputs,
        generated_outputs=generated_outputs,
        status=status,
    )


def prepare_all_phase_plots(
    context: CaseContext,
    *,
    options: WriteOptions | None = None,
    include_secondary: bool = False,
) -> PlotAllResult:
    """Prepare plot manifests for every phase with a planned paper figure."""
    options = options or WriteOptions()
    options.validate()
    return PlotAllResult(
        results=tuple(
            prepare_phase_plots(
                context,
                phase,
                options=options,
                include_secondary=include_secondary,
            )
            for phase in PLOT_PHASES
        )
    )


def load_primary_rank_table(path: Path) -> pd.DataFrame:
    """Load and validate the canonical plotting input table."""
    path = Path(path)
    if not path.exists():
        raise PlotPreparationError(f"primary rank table does not exist: {path}")
    table = pd.read_csv(path)
    missing = [column for column in PRIMARY_RANK_TABLE_COLUMNS if column not in table.columns]
    if missing:
        raise PlotPreparationError(f"primary rank table is missing required column(s): {missing}")
    if table.empty:
        raise PlotPreparationError(f"primary rank table is empty: {path}")
    return table.copy()


def load_metrics_summary(path: Path) -> pd.DataFrame:
    """Load the generic summarize-cv metric summary used by supplementary plots."""
    path = Path(path)
    if not path.exists():
        raise PlotPreparationError(f"metrics summary does not exist: {path}")
    table = pd.read_csv(path)
    required = ["model", "metric", "mean", "std", "n"]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise PlotPreparationError(f"metrics summary is missing required column(s): {missing}")
    if "rank" not in table.columns:
        table["rank"] = pd.NA
    return table.loc[:, [*required, "rank"]].copy()


def load_ablation_statistical_tests(path: Path) -> pd.DataFrame:
    """Load held-out ablation rank evidence written by ``summarize-paper``."""
    path = Path(path)
    if not path.exists():
        raise PlotPreparationError(
            "phase 03 ablation plot requires held-out statistical tests; "
            f"run 'amiga-exp summarize-paper' first or provide: {path}"
        )
    table = pd.read_csv(path)
    required = [
        "comparison_type",
        "metric",
        "method",
        "avg_rank",
        "holm_p_adj",
        "n_fronts",
    ]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise PlotPreparationError(f"statistical tests table is missing required column(s): {missing}")
    ablation = table[
        (table["comparison_type"].astype(str) == "ablation")
        & (table["metric"].astype(str) == "Regret@5")
    ].copy()
    if ablation.empty:
        raise PlotPreparationError(
            "statistical tests table does not contain ablation rows for Regret@5"
        )
    return ablation


def planned_phase_plot_outputs(phase_dir: Path, phase: str) -> dict[str, Any]:
    """Return manifest-ready paths for the plot artifacts planned for a phase."""
    if phase not in PLOT_PHASES:
        raise PlotPreparationError(
            f"unsupported plot phase '{phase}'. Valid phases are: {', '.join(PLOT_PHASES)}"
        )

    plots_dir = Path(phase_dir) / "plots"
    planned_outputs = _planned_outputs(phase, plots_dir)
    return {
        "plot_manifest": repo_relative(plots_dir / "plot_manifest.json"),
        "primary": {
            output_type: repo_relative(output_path)
            for output_type, output_path in planned_outputs["primary"].items()
        },
    }


def _planned_outputs(
    phase: str,
    plots_dir: Path,
    *,
    include_secondary: bool = False,
) -> dict[str, dict[str, Path]]:
    stem = PHASE_PRIMARY_PLOT_FILES[phase]
    outputs = {
        "primary": {
            "png": plots_dir / f"{stem}.png",
            "pdf": plots_dir / f"{stem}.pdf",
        }
    }
    if phase in {"02_hyperparameter_tuning", "03_ablation", "04_decision_baselines"}:
        outputs["primary"]["csv"] = plots_dir / f"{stem}.csv"
    if include_secondary:
        supplementary_dir = plots_dir / "supplementary"
        outputs["supplementary_topk_curves"] = {
            "png": supplementary_dir / "topk_metric_curves.png",
            "pdf": supplementary_dir / "topk_metric_curves.pdf",
            "csv": supplementary_dir / "topk_metric_curves.csv",
        }
        outputs["supplementary_metric_appendix"] = {
            "csv": supplementary_dir / "secondary_metric_appendix.csv",
            "caption": supplementary_dir / "secondary_metric_caption.txt",
        }
    return outputs


def _generate_phase_outputs(
    phase: str,
    *,
    phase_dir: Path,
    metrics_summary: Path,
    statistical_tests: Path,
    primary_table: pd.DataFrame,
    planned_outputs: Mapping[str, Mapping[str, Path]],
    options: WriteOptions,
    skip_generation: bool = False,
) -> dict[str, dict[str, Path]]:
    if options.dry_run:
        return {}
    if phase not in {
        "01_model_screening",
        "02_hyperparameter_tuning",
        "03_ablation",
        "04_decision_baselines",
    }:
        return {}

    if skip_generation:
        return _existing_outputs(planned_outputs)

    generated_outputs: dict[str, dict[str, Path]] = {}
    outputs = {key: Path(path) for key, path in planned_outputs["primary"].items()}
    if not _plot_outputs_are_writable(outputs.values(), options):
        generated_outputs["primary"] = {
            key: path for key, path in outputs.items() if path.exists()
        }
    else:
        if phase == "01_model_screening":
            selected_configs = _load_phase01_selected_configs(phase_dir / "shortlisted_configs.json")
            save_model_screening_heatmap(
                primary_table,
                outputs=outputs,
                selected_configs=selected_configs,
            )
        elif phase == "02_hyperparameter_tuning":
            selected_config = _load_phase02_selected_config(phase_dir / "selected_config.json")
            save_hyperparameter_regret_scatter(
                primary_table,
                outputs=outputs,
                selected_config=selected_config,
            )
        elif phase == "03_ablation":
            ablation_test_table = load_ablation_statistical_tests(statistical_tests)
            save_ablation_feature_matrix_plot(
                primary_table,
                ablation_test_table=ablation_test_table,
                outputs=outputs,
            )
        elif phase == "04_decision_baselines":
            save_decision_baseline_rank_plot(
                primary_table,
                outputs=outputs,
            )
        generated_outputs["primary"] = outputs

    if "supplementary_topk_curves" in planned_outputs:
        generated_outputs.update(
            _generate_supplementary_outputs(
                metrics_summary,
                primary_table=primary_table,
                planned_outputs=planned_outputs,
                options=options,
            )
        )
    return generated_outputs


def _ensure_plot_manifest_writable(plot_manifest: Path, options: WriteOptions) -> None:
    if options.dry_run or options.force or options.skip_existing:
        return
    if plot_manifest.exists():
        raise PlotPreparationError(
            f"refusing to overwrite existing plot manifest without --force: {plot_manifest}"
        )


def _plot_outputs_are_writable(paths: Any, options: WriteOptions) -> bool:
    existing = [Path(path) for path in paths if Path(path).exists()]
    if not existing or options.force:
        return True
    if options.skip_existing:
        return False
    raise PlotPreparationError(
        "refusing to overwrite existing plot output(s) without --force: "
        + ", ".join(str(path) for path in existing)
    )


def _existing_outputs(
    planned_outputs: Mapping[str, Mapping[str, Path]]
) -> dict[str, dict[str, Path]]:
    existing: dict[str, dict[str, Path]] = {}
    for group, outputs in planned_outputs.items():
        group_existing = {
            output_type: Path(path)
            for output_type, path in outputs.items()
            if Path(path).exists()
        }
        if group_existing:
            existing[group] = group_existing
    return existing


def _generate_supplementary_outputs(
    metrics_summary: Path,
    *,
    primary_table: pd.DataFrame,
    planned_outputs: Mapping[str, Mapping[str, Path]],
    options: WriteOptions,
) -> dict[str, dict[str, Path]]:
    metric_summary_df = load_metrics_summary(metrics_summary)
    topk_frame = _supplementary_topk_frame(metric_summary_df, primary_table)
    generated: dict[str, dict[str, Path]] = {}

    topk_outputs = {
        key: Path(path)
        for key, path in planned_outputs["supplementary_topk_curves"].items()
    }
    if not _plot_outputs_are_writable(topk_outputs.values(), options):
        generated["supplementary_topk_curves"] = {
            key: path for key, path in topk_outputs.items() if path.exists()
        }
    else:
        save_supplementary_topk_curves(topk_frame, outputs=topk_outputs)
        generated["supplementary_topk_curves"] = topk_outputs

    appendix_outputs = {
        key: Path(path)
        for key, path in planned_outputs["supplementary_metric_appendix"].items()
    }
    if not _plot_outputs_are_writable(appendix_outputs.values(), options):
        generated["supplementary_metric_appendix"] = {
            key: path for key, path in appendix_outputs.items() if path.exists()
        }
    else:
        save_supplementary_metric_appendix(topk_frame, outputs=appendix_outputs)
        generated["supplementary_metric_appendix"] = appendix_outputs

    return generated


def _load_phase01_selected_configs(path: Path) -> set[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PlotPreparationError(f"phase 01 shortlisted configs file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PlotPreparationError(f"invalid phase 01 shortlisted configs JSON: {path}") from exc

    configs = payload.get("configs") if isinstance(payload, dict) else None
    if not isinstance(configs, list) or not configs:
        raise PlotPreparationError(f"phase 01 shortlisted configs JSON has no non-empty 'configs' list: {path}")
    selected = {str(config.get("run_id", "")) for config in configs if isinstance(config, dict)}
    selected.discard("")
    if not selected:
        raise PlotPreparationError(f"phase 01 shortlisted configs JSON has no selectable run_id values: {path}")
    return selected


def _load_phase02_selected_config(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PlotPreparationError(f"phase 02 selected config file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PlotPreparationError(f"invalid phase 02 selected config JSON: {path}") from exc

    selected_config = payload.get("selected_config") if isinstance(payload, dict) else None
    if not isinstance(selected_config, dict):
        raise PlotPreparationError(f"phase 02 selected config JSON has no 'selected_config' object: {path}")
    run_id = str(selected_config.get("run_id", ""))
    if not run_id:
        raise PlotPreparationError(f"phase 02 selected config JSON has no selected run_id: {path}")
    return run_id


def save_supplementary_topk_curves(
    topk_frame: pd.DataFrame,
    *,
    outputs: Mapping[str, Path],
) -> None:
    """Render complementary Top-K metric curves from summarize-cv outputs."""
    plot_df = topk_frame.copy()
    families = [
        family
        for family in SUPPLEMENTARY_METRIC_FAMILIES
        if family in set(plot_df["metric_family"])
    ]
    if not families:
        raise PlotPreparationError("no supplementary Top-K metric families are available")

    ordered_configs = (
        plot_df[["config", "primary_rank_order"]]
        .drop_duplicates()
        .sort_values(["primary_rank_order", "config"], kind="mergesort")["config"]
        .tolist()
    )
    palette = dict(
        zip(
            ordered_configs,
            sns.color_palette("colorblind", n_colors=max(1, len(ordered_configs))),
            strict=False,
        )
    )

    sns.set_theme(
        style="whitegrid",
        context="paper",
        palette="colorblind",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        },
    )
    fig_height = max(4.8, 2.45 * len(families) + 1.4)
    fig, axes = plt.subplots(
        len(families),
        1,
        figsize=(8.7, fig_height),
        sharex=True,
        squeeze=False,
    )
    legend_handles = []
    legend_labels = []
    legend_limit = 12

    for axis_idx, family in enumerate(families):
        ax = axes[axis_idx][0]
        family_df = plot_df[plot_df["metric_family"] == family]
        for config in ordered_configs:
            config_df = family_df[family_df["config"] == config].sort_values("k")
            if config_df.empty:
                continue
            order = int(config_df["primary_rank_order"].iloc[0])
            line, = ax.plot(
                config_df["k"],
                config_df["mean"],
                marker="o",
                linewidth=2.0 if order <= 3 else 1.1,
                markersize=4.6 if order <= 3 else 3.8,
                alpha=0.92 if order <= 3 else 0.58,
                color=palette[config],
                label=config,
            )
            if axis_idx == 0 and len(legend_handles) < legend_limit:
                legend_handles.append(line)
                legend_labels.append(config)

        direction = "lower is better" if family == "Regret" else "higher is better"
        ax.set_title(f"{family}@k ({direction})")
        ax.set_ylabel("Mean")
        ax.grid(axis="x", alpha=0.18)
        ax.grid(axis="y", alpha=0.22)

    axes[-1][0].set_xlabel("k")
    fig.suptitle("Complementary Top-K metric curves", fontweight="semibold")
    fig.text(
        0.5,
        0.018,
        "Complementary context only; selections remain fixed by the primary Regret@5 ranking.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.055),
            ncol=min(3, len(legend_labels)),
            frameon=True,
        )
    if len(ordered_configs) > legend_limit:
        fig.text(
            0.985,
            0.055,
            f"{len(ordered_configs) - legend_limit} additional configs shown without legend",
            ha="right",
            va="center",
            fontsize=8,
            color="#666666",
        )
    fig.tight_layout(rect=(0, 0.12, 1, 0.96))

    plotted_csv = outputs.get("csv")
    if plotted_csv is not None:
        plotted_csv.parent.mkdir(parents=True, exist_ok=True)
        plot_df.to_csv(plotted_csv, index=False)
    for key, output_path in outputs.items():
        if key == "csv":
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".png":
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        else:
            fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_supplementary_metric_appendix(
    topk_frame: pd.DataFrame,
    *,
    outputs: Mapping[str, Path],
) -> None:
    """Write the compact complementary metric appendix table and caption."""
    appendix = _supplementary_appendix_frame(topk_frame)
    csv_path = outputs.get("csv")
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        appendix.to_csv(csv_path, index=False)

    caption_path = outputs.get("caption")
    if caption_path is not None:
        caption_path.parent.mkdir(parents=True, exist_ok=True)
        caption_path.write_text(
            "Supplementary Top-K context metrics generated from summarize-cv outputs. "
            f"{SUPPLEMENTARY_POLICY}\n",
            encoding="utf-8",
        )


def save_model_screening_heatmap(
    primary_table: pd.DataFrame,
    *,
    outputs: Mapping[str, Path],
    selected_configs: set[str],
) -> None:
    """Render the phase-01 model-screening heatmap."""
    plot_df = _model_screening_plot_frame(primary_table, selected_configs=selected_configs)
    model_order = _ordered_unique(plot_df, "model", order_map=_MODEL_ORDER)
    label_order = _ordered_unique(plot_df, "label", order_map=_LABEL_ORDER)
    pivot = plot_df.pivot(index="model", columns="label", values="avg_rank").reindex(
        index=model_order,
        columns=label_order,
    )
    row_lookup = {
        (str(row.model), str(row.label)): row
        for row in plot_df.itertuples(index=False)
    }

    sns.set_theme(
        style="white",
        context="paper",
        palette="colorblind",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        },
    )
    fig_width = max(7.8, 1.35 * len(label_order) + 2.8)
    fig_height = max(4.6, 0.78 * len(model_order) + 2.3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    rank_cmap, rank_norm, cbar_extend = _model_screening_rank_color_scale(plot_df)
    sns.heatmap(
        pivot,
        cmap=rank_cmap,
        norm=rank_norm,
        annot=False,
        linewidths=0.7,
        linecolor="white",
        cbar_kws={
            "label": "Avg. rank on Regret@5",
            "shrink": 0.82,
            "pad": 0.02,
            "extend": cbar_extend,
        },
        ax=ax,
    )
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=11)
        cbar.ax.set_ylabel("Avg. rank on Regret@5", fontsize=12.5)
    control_boundary = _control_label_boundary(label_order)
    if control_boundary is not None:
        ax.axvline(
            control_boundary,
            color="#343434",
            linewidth=1.35,
            zorder=6,
        )

    for y_idx, model in enumerate(model_order):
        for x_idx, label in enumerate(label_order):
            row = row_lookup.get((model, label))
            if row is None:
                continue
            ax.text(
                x_idx + 0.5,
                y_idx + 0.40,
                f"{float(row.avg_rank):.2f}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold" if bool(row.selected) else "normal",
                color="#111111",
            )
            ax.text(
                x_idx + 0.5,
                y_idx + 0.64,
                str(row.p_value_label),
                ha="center",
                va="center",
                fontsize=10.5,
                fontweight="bold",
                color=str(row.p_value_color),
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.64,
                },
            )
            if bool(row.selected):
                ax.text(
                    x_idx + 0.90,
                    y_idx + 0.18,
                    "*",
                    ha="center",
                    va="center",
                    fontsize=20,
                    fontweight="bold",
                    color="#111111",
                )

    selected_df = plot_df[plot_df["selected"]]
    if not selected_df.empty:
        ax.scatter(
            [],
            [],
            marker="*",
            s=140,
            color="#111111",
            edgecolor="white",
            label="Selected for tuning",
        )

    ax.set_xlabel("Label mode")
    ax.set_ylabel("Model family")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.scatter([], [], marker="s", s=52, color="#D55E00", label="p < 0.05")
    ax.scatter([], [], marker="s", s=52, color="#0072B2", label="p >= 0.05")
    ax.scatter([], [], marker="s", s=52, color="#111111", label="Best within model")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=True,
        columnspacing=1.2,
        handletextpad=0.5,
        fontsize=11,
    )
    fig.tight_layout()

    for output_path in outputs.values():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".png":
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        else:
            fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_hyperparameter_regret_scatter(
    primary_table: pd.DataFrame,
    *,
    outputs: Mapping[str, Path],
    selected_config: str,
) -> None:
    """Render the phase-02 regret mean/std hyperparameter scatter plot."""
    plot_df = _hyperparameter_plot_frame(primary_table, selected_config=selected_config)
    plot_df = plot_df.sort_values(["avg_rank", "config"], kind="mergesort").reset_index(drop=True)

    sns.set_theme(
        style="whitegrid",
        context="paper",
        palette="colorblind",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        },
    )
    fig_width = 11.2
    fig_height = 7.4
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=[0.055, 0.08, 1.0, 0.30],
        height_ratios=[0.24, 1.0],
        wspace=0.08,
        hspace=0.06,
    )
    cbar_ax = fig.add_subplot(grid[1, 0])
    top_ax = fig.add_subplot(grid[0, 2])
    ax = fig.add_subplot(grid[1, 2], sharex=top_ax)
    right_ax = fig.add_subplot(grid[1, 3], sharey=ax)
    legend_ax = fig.add_subplot(grid[0, 3])
    fig.add_subplot(grid[0, 0]).axis("off")
    fig.add_subplot(grid[0, 1]).axis("off")
    fig.add_subplot(grid[1, 1]).axis("off")

    group_order = _ordered_hyperparameter_groups(plot_df)
    group_palette = dict(
        zip(group_order, _hyperparameter_group_palette(len(group_order)))
    )
    for group in group_order:
        group_df = plot_df[plot_df["base_config"] == group]
        _add_group_density_blob(
            ax,
            group_df,
            facecolor=group_palette[group],
            edgecolor=group_palette[group],
        )

    rank_cmap, rank_norm, significance_rank = _hyperparameter_rank_color_scale(plot_df)
    scatter = ax.scatter(
        plot_df["mean_regret5"],
        plot_df["std_regret5"],
        c=plot_df["avg_rank"],
        cmap=rank_cmap,
        norm=rank_norm,
        s=92,
        edgecolors="white",
        linewidth=0.85,
        zorder=4,
    )
    selected_df = plot_df[plot_df["selected"]]
    ax.scatter(
        selected_df["mean_regret5"],
        selected_df["std_regret5"],
        s=148,
        marker="o",
        facecolors="none",
        edgecolors="#111111",
        linewidths=2.25,
        zorder=8,
    )

    parameter_traces = _hyperparameter_parameter_trace_frame(plot_df)
    _add_hyperparameter_parameter_marginals(
        top_ax,
        right_ax,
        parameter_traces,
        group_order=group_order,
        group_palette=group_palette,
    )
    _add_hyperparameter_parameter_legend(
        legend_ax,
        parameter_traces,
        group_order=group_order,
        group_palette=group_palette,
    )

    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.set_title("Avg. rank\nRegret@5", fontsize=8, pad=7)
    _add_rank_significance_marker(cbar, significance_rank)

    x_range = max(1e-9, float(plot_df["mean_regret5"].max() - plot_df["mean_regret5"].min()))
    y_range = max(1e-9, float(plot_df["std_regret5"].max() - plot_df["std_regret5"].min()))
    x_pad = 0.24 * x_range
    y_pad = 0.24 * y_range
    ax.set_xlim(
        left=max(0.0, float(plot_df["mean_regret5"].min()) - x_pad),
        right=float(plot_df["mean_regret5"].max()) + x_pad,
    )
    ax.set_ylim(
        bottom=max(0.0, float(plot_df["std_regret5"].min()) - y_pad),
        top=float(plot_df["std_regret5"].max()) + y_pad,
    )
    ax.set_xlabel("Mean Regret@5")
    ax.set_ylabel("Std. deviation of Regret@5")
    ax.grid(axis="both", alpha=0.18)
    _add_hyperparameter_group_legend(ax, plot_df, group_order, group_palette)
    plt.setp(top_ax.get_xticklabels(), visible=False)
    plt.setp(right_ax.get_yticklabels(), visible=False)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.095, top=0.975)
    _match_normalized_marginal_axis_lengths(fig, top_ax, right_ax)

    plotted_csv = outputs.get("csv")
    if plotted_csv is not None:
        plotted_csv.parent.mkdir(parents=True, exist_ok=True)
        plot_df.to_csv(plotted_csv, index=False)
    for key, output_path in outputs.items():
        if key == "csv":
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".png":
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        else:
            fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_ablation_feature_matrix_plot(
    primary_table: pd.DataFrame,
    *,
    ablation_test_table: pd.DataFrame,
    outputs: Mapping[str, Path],
) -> None:
    """Render the phase-03 feature-block ablation matrix."""
    plot_df = _ablation_matrix_plot_frame(
        primary_table,
        ablation_test_table=ablation_test_table,
    )

    sns.set_theme(
        style="white",
        context="paper",
        palette="colorblind",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 10.5,
        },
    )
    fig_width = 8.25
    fig_height = max(5.0, 0.58 * len(plot_df) + 1.7)
    fig = plt.figure(figsize=(fig_width, fig_height))
    grid = fig.add_gridspec(1, 2, width_ratios=[4.9, 2.35], wspace=0.035)
    block_ax = fig.add_subplot(grid[0, 0])
    rank_ax = fig.add_subplot(grid[0, 1])

    block_keys = [key for key, _ in ABLATION_BLOCK_COLUMNS]
    block_labels = [label for _, label in ABLATION_BLOCK_COLUMNS]
    block_values = plot_df[block_keys].astype(int)
    block_cmap = mcolors.ListedColormap(["#F4F4F4", "#2E9F7D"])
    block_norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], block_cmap.N)
    sns.heatmap(
        block_values,
        cmap=block_cmap,
        norm=block_norm,
        cbar=False,
        linewidths=0.75,
        linecolor="white",
        xticklabels=block_labels,
        yticklabels=plot_df["display_label"].tolist(),
        ax=block_ax,
    )

    for y_idx, row in plot_df.iterrows():
        for x_idx, block_key in enumerate(block_keys):
            is_active = bool(row[block_key])
            block_ax.text(
                x_idx + 0.5,
                y_idx + 0.5,
                "ON" if is_active else "OFF",
                ha="center",
                va="center",
                fontsize=10.5,
                fontweight="bold" if is_active else "normal",
                color="white" if is_active else "#777777",
            )

    rank_values = plot_df[["development_avg_rank", "test_avg_rank"]]
    rank_min = float(rank_values.min().min())
    rank_max = float(rank_values.max().max())
    sns.heatmap(
        rank_values,
        cmap=sns.light_palette("#2A6FBB", as_cmap=True, reverse=True),
        vmin=rank_min,
        vmax=rank_max,
        cbar=False,
        linewidths=0.75,
        linecolor="white",
        xticklabels=["Development", "Held-out\ntest"],
        yticklabels=False,
        ax=rank_ax,
    )

    for y_idx, row in plot_df.iterrows():
        for x_idx, prefix in enumerate(("development", "test")):
            p_value = row[f"{prefix}_p_value"]
            is_winner = pd.isna(p_value)
            rank_ax.text(
                x_idx + 0.5,
                y_idx + 0.39,
                f"{float(row[f'{prefix}_avg_rank']):.2f}",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold" if is_winner else "normal",
                color="#111111",
            )
            rank_ax.text(
                x_idx + 0.5,
                y_idx + 0.64,
                str(row[f"{prefix}_p_value_label"]),
                ha="center",
                va="center",
                fontsize=9.2,
                fontweight="bold",
                color=str(row[f"{prefix}_p_value_color"]),
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.68,
                },
            )

    block_ax.set_xlabel("Feature block")
    block_ax.set_ylabel("Ablation variant")
    block_ax.set_xticklabels(block_ax.get_xticklabels(), rotation=25, ha="right")
    block_ax.set_yticklabels(block_ax.get_yticklabels(), rotation=0)
    rank_ax.set_xlabel("")
    rank_ax.set_ylabel("")
    rank_ax.set_yticks([])
    rank_ax.tick_params(axis="y", left=False, labelleft=False)
    rank_ax.set_xticklabels(rank_ax.get_xticklabels(), rotation=0)
    for axis in (block_ax, rank_ax):
        axis.tick_params(axis="both", length=0)
    fig.subplots_adjust(left=0.27, right=0.985, bottom=0.18, top=0.985, wspace=0.035)

    plotted_csv = outputs.get("csv")
    if plotted_csv is not None:
        plotted_csv.parent.mkdir(parents=True, exist_ok=True)
        plot_df.to_csv(plotted_csv, index=False)
    for key, output_path in outputs.items():
        if key == "csv":
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".png":
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        else:
            fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_decision_baseline_rank_plot(
    primary_table: pd.DataFrame,
    *,
    outputs: Mapping[str, Path],
) -> None:
    """Render the phase-04 decision-baseline average-rank comparison."""
    plot_df = _decision_baseline_plot_frame(primary_table)
    plot_df["y"] = range(len(plot_df))
    amiga_rank = float(plot_df.loc[plot_df["is_amiga"], "avg_rank"].iloc[0])

    sns.set_theme(
        style="whitegrid",
        context="paper",
        palette="colorblind",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 10.2,
        },
    )
    fig_width = 7.2
    fig_height = max(5.0, 0.55 * len(plot_df) + 1.7)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.axvline(
        amiga_rank,
        color="#009E73",
        linestyle=(0, (4, 3)),
        linewidth=1.25,
        alpha=0.72,
        zorder=0,
    )
    for _, row in plot_df.iterrows():
        x_position = float(row["avg_rank"])
        ax.hlines(
            y=int(row["y"]),
            xmin=min(amiga_rank, x_position),
            xmax=max(amiga_rank, x_position),
            color="#009E73" if bool(row["is_amiga"]) else "#B8BDC3",
            linewidth=2.0 if bool(row["is_amiga"]) else 1.7,
            alpha=0.95 if bool(row["is_amiga"]) else 0.76,
            zorder=1,
        )
    baseline_df = plot_df[~plot_df["is_amiga"]]
    amiga_df = plot_df[plot_df["is_amiga"]]
    ax.scatter(
        baseline_df["avg_rank"],
        baseline_df["y"],
        s=110,
        color="#7D848C",
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )
    ax.scatter(
        amiga_df["avg_rank"],
        amiga_df["y"],
        s=210,
        marker="D",
        color="#009E73",
        edgecolor="#111111",
        linewidth=1.4,
        zorder=4,
    )

    x_span = max(1e-9, float(plot_df["avg_rank"].max() - plot_df["avg_rank"].min()))
    label_offset = max(0.06, 0.035 * x_span)
    for _, row in plot_df.iterrows():
        x_position = float(row["avg_rank"])
        label_side = 1 if x_position >= amiga_rank else -1
        if bool(row["is_amiga"]):
            label_side = 1
        ax.text(
            x_position + label_side * label_offset,
            int(row["y"]),
            row["p_value_label"],
            va="center",
            ha="left" if label_side > 0 else "right",
            fontsize=10,
            color=str(row["p_value_color"]),
            fontweight="bold" if row["is_amiga"] else "normal",
            bbox={
                "boxstyle": "round,pad=0.14",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.74,
            },
            zorder=5,
        )

    ax.set_yticks(plot_df["y"])
    ax.set_yticklabels(plot_df["display_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Average rank on Regret@5 (lower is better)")
    ax.set_ylabel("Method")
    ax.grid(axis="x", alpha=0.22)
    ax.grid(axis="y", alpha=0.08)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor="#009E73",
            markeredgecolor="#111111",
            markeredgewidth=1.2,
            markersize=7,
            label="AMIGA",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#7D848C",
            markeredgecolor="white",
            markeredgewidth=0.9,
            markersize=7,
            label="Decision baseline",
        ),
        Line2D(
            [0],
            [0],
            color=_p_value_color(0.01),
            linewidth=2.0,
            label="p < 0.05",
        ),
        Line2D(
            [0],
            [0],
            color=_p_value_color(0.50),
            linewidth=2.0,
            label="p >= 0.05",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        frameon=True,
        title="Method and p-value",
        fontsize=10,
        title_fontsize=10.5,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    x_min = float(plot_df["avg_rank"].min())
    x_max = float(plot_df["avg_rank"].max())
    x_pad_left = max(0.12, 0.08 * x_span)
    x_pad_right = max(0.42, 0.22 * x_span)
    ax.set_xlim(left=max(0.0, x_min - x_pad_left), right=x_max + x_pad_right)
    fig.tight_layout()

    plotted_csv = outputs.get("csv")
    if plotted_csv is not None:
        plotted_csv.parent.mkdir(parents=True, exist_ok=True)
        plot_df.drop(columns=["y"]).to_csv(plotted_csv, index=False)
    for key, output_path in outputs.items():
        if key == "csv":
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".png":
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        else:
            fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


_MODEL_ORDER = {
    "LGBMRanker": 0,
    "XGBRanker": 1,
    "CatBoostRanker": 2,
}

_LABEL_ORDER = {
    "rank dense": 0,
    "rank avg": 1,
    "quantiles q5": 2,
    "quantiles q10": 3,
    "quantiles q15": 4,
    "continuous": 5,
    "reversed": 6,
    "shuffled": 7,
}


def _model_screening_rank_color_scale(
    plot_df: pd.DataFrame,
) -> tuple[mcolors.Colormap, mcolors.PowerNorm, str]:
    values = pd.to_numeric(plot_df["avg_rank"], errors="raise").astype(float)
    vmin = float(values.min())
    vmax = float(values.max())
    non_control = plot_df[~plot_df["label"].isin(CONTROL_LABEL_MODES)]["avg_rank"]
    if not non_control.empty:
        robust_vmax = float(pd.to_numeric(non_control, errors="raise").max())
    else:
        robust_vmax = vmax
    robust_vmax = max(robust_vmax, vmin + 1e-9)

    cmap = sns.light_palette("#2A6FBB", as_cmap=True, reverse=True)
    cmap = cmap.copy()
    cmap.set_over("#EEF2F5")
    cbar_extend = "max" if vmax > robust_vmax else "neither"
    return (
        cmap,
        mcolors.PowerNorm(gamma=0.65, vmin=vmin, vmax=robust_vmax, clip=False),
        cbar_extend,
    )


def _control_label_boundary(label_order: Sequence[str]) -> int | None:
    control_positions = [
        idx
        for idx, label in enumerate(label_order)
        if label in CONTROL_LABEL_MODES
    ]
    if not control_positions:
        return None
    first_control = min(control_positions)
    return first_control if first_control > 0 else None


def _model_screening_plot_frame(primary_table: pd.DataFrame, *, selected_configs: set[str]) -> pd.DataFrame:
    rows = []
    for row in primary_table.itertuples(index=False):
        config = str(row.config)
        model, label = _parse_screening_config(config)
        rows.append(
            {
                "config": config,
                "model": model,
                "label": label,
                "avg_rank": float(row.avg_rank),
                "p_value": _optional_plot_float(row.p_value),
                "p_value_label": _format_p_value(row.p_value),
                "p_value_significance": _p_value_significance(row.p_value),
                "p_value_color": _p_value_color(row.p_value),
                "selected": config in selected_configs,
            }
        )
    plot_df = pd.DataFrame(rows)
    if not plot_df["selected"].any():
        missing = sorted(selected_configs - set(plot_df["config"]))
        raise PlotPreparationError(
            "phase 01 heatmap cannot mark shortlisted config(s) because they are absent "
            f"from primary_rank_table.csv: {missing}"
        )
    return plot_df


def _hyperparameter_plot_frame(primary_table: pd.DataFrame, *, selected_config: str) -> pd.DataFrame:
    plot_df = primary_table.copy()
    plot_df["config"] = plot_df["config"].astype(str)
    plot_df["avg_rank"] = pd.to_numeric(plot_df["avg_rank"], errors="raise")
    plot_df["mean_regret5"] = pd.to_numeric(plot_df["mean_regret5"], errors="raise")
    plot_df["std_regret5"] = pd.to_numeric(plot_df["std_regret5"], errors="raise")
    plot_df["selected"] = plot_df["config"] == selected_config
    if not plot_df["selected"].any():
        raise PlotPreparationError(
            "phase 02 scatter cannot mark selected config because it is absent "
            f"from primary_rank_table.csv: {selected_config}"
        )
    parsed = plot_df["config"].map(_parse_hyperparameter_config).apply(pd.Series)
    plot_df = pd.concat([plot_df, parsed], axis=1)
    plot_df["display_label"] = plot_df["config"].map(_hyperparameter_display_label)
    plot_df["p_value_label"] = plot_df["p_value"].map(_format_p_value)
    return plot_df


def _ablation_matrix_plot_frame(
    primary_table: pd.DataFrame,
    *,
    ablation_test_table: pd.DataFrame,
) -> pd.DataFrame:
    plot_df = primary_table.copy()
    plot_df["config"] = plot_df["config"].astype(str)
    if "full" not in set(plot_df["config"]):
        raise PlotPreparationError(
            "phase 03 ablation matrix requires a 'full' reference row in primary_rank_table.csv"
        )
    unknown_configs = sorted(set(plot_df["config"]) - set(ABLATION_FEATURE_BLOCKS))
    if unknown_configs:
        raise PlotPreparationError(
            "phase 03 ablation matrix cannot infer feature blocks for config(s): "
            f"{unknown_configs}"
        )
    plot_df["development_avg_rank"] = pd.to_numeric(plot_df["avg_rank"], errors="raise")
    plot_df["development_mean_regret5"] = pd.to_numeric(plot_df["mean_regret5"], errors="raise")
    plot_df["development_std_regret5"] = pd.to_numeric(plot_df["std_regret5"], errors="coerce")
    plot_df["development_p_value"] = pd.to_numeric(plot_df["p_value"], errors="coerce")
    plot_df["display_label"] = plot_df["config"].map(_feature_set_display_label)
    plot_df["development_p_value_label"] = plot_df["development_p_value"].map(_format_p_value)
    plot_df["development_p_value_significance"] = plot_df["development_p_value"].map(_p_value_significance)
    plot_df["development_p_value_color"] = plot_df["development_p_value"].map(_p_value_color)
    plot_df["development_n_fronts"] = pd.to_numeric(plot_df["n_fronts"], errors="raise").astype(int)
    for block_key, _ in ABLATION_BLOCK_COLUMNS:
        plot_df[block_key] = plot_df["config"].map(
            lambda config, key=block_key: bool(ABLATION_FEATURE_BLOCKS[str(config)][key])
        )

    test_df = _ablation_test_plot_frame(ablation_test_table)
    plot_df = plot_df.merge(test_df, on="config", how="left", validate="one_to_one")
    missing_test = sorted(
        str(config)
        for config in plot_df.loc[plot_df["test_avg_rank"].isna(), "config"].tolist()
    )
    if missing_test:
        raise PlotPreparationError(
            "phase 03 ablation matrix cannot find held-out test evidence for config(s): "
            f"{missing_test}"
        )

    plot_df = plot_df.sort_values(["development_avg_rank", "config"], kind="mergesort").reset_index(drop=True)
    return plot_df[
        [
            "config",
            "display_label",
            *[key for key, _ in ABLATION_BLOCK_COLUMNS],
            "development_avg_rank",
            "development_p_value",
            "development_p_value_label",
            "development_p_value_significance",
            "development_p_value_color",
            "development_mean_regret5",
            "development_std_regret5",
            "development_n_fronts",
            "test_avg_rank",
            "test_p_value",
            "test_p_value_label",
            "test_p_value_significance",
            "test_p_value_color",
            "test_n_fronts",
        ]
    ]


def _ablation_test_plot_frame(statistical_tests: pd.DataFrame) -> pd.DataFrame:
    test_df = statistical_tests.copy()
    test_df["method"] = test_df["method"].astype(str)
    test_df["config"] = test_df["method"].replace({"AMIGA_final": "full"})
    unknown_configs = sorted(set(test_df["config"]) - set(ABLATION_FEATURE_BLOCKS))
    if unknown_configs:
        raise PlotPreparationError(
            "phase 03 ablation held-out tests contain unknown ablation config(s): "
            f"{unknown_configs}"
        )
    test_df["_amiga_final_priority"] = (test_df["method"] == "AMIGA_final").astype(int)
    test_df = (
        test_df.sort_values(["config", "_amiga_final_priority"], ascending=[True, False])
        .drop_duplicates("config", keep="first")
        .copy()
    )
    test_df["test_avg_rank"] = pd.to_numeric(test_df["avg_rank"], errors="raise")
    test_df["test_p_value"] = pd.to_numeric(test_df["holm_p_adj"], errors="coerce")
    test_df["test_p_value_label"] = test_df["test_p_value"].map(_format_p_value)
    test_df["test_p_value_significance"] = test_df["test_p_value"].map(_p_value_significance)
    test_df["test_p_value_color"] = test_df["test_p_value"].map(_p_value_color)
    test_df["test_n_fronts"] = pd.to_numeric(test_df["n_fronts"], errors="raise").astype(int)
    return test_df[
        [
            "config",
            "test_avg_rank",
            "test_p_value",
            "test_p_value_label",
            "test_p_value_significance",
            "test_p_value_color",
            "test_n_fronts",
        ]
    ]


def _decision_baseline_plot_frame(primary_table: pd.DataFrame) -> pd.DataFrame:
    plot_df = primary_table.copy()
    plot_df["config"] = plot_df["config"].astype(str)
    if "AMIGA_final" not in set(plot_df["config"]):
        raise PlotPreparationError(
            "phase 04 decision-baseline plot requires an 'AMIGA_final' reference row "
            "in primary_rank_table.csv"
        )
    plot_df["avg_rank"] = pd.to_numeric(plot_df["avg_rank"], errors="raise")
    plot_df["p_value"] = pd.to_numeric(plot_df["p_value"], errors="coerce")
    plot_df["mean_regret5"] = pd.to_numeric(plot_df["mean_regret5"], errors="raise")
    plot_df["std_regret5"] = pd.to_numeric(plot_df["std_regret5"], errors="coerce")
    plot_df["is_amiga"] = plot_df["config"] == "AMIGA_final"
    plot_df["display_label"] = plot_df["config"].map(_baseline_display_label)
    plot_df["p_value_label"] = plot_df["p_value"].map(_format_p_value)
    plot_df["p_value_significance"] = plot_df["p_value"].map(_p_value_significance)
    plot_df["p_value_color"] = plot_df["p_value"].map(_p_value_color)
    plot_df = plot_df.sort_values(["avg_rank", "config"], kind="mergesort").reset_index(drop=True)
    return plot_df[
        [
            "config",
            "display_label",
            "avg_rank",
            "p_value",
            "p_value_label",
            "p_value_significance",
            "p_value_color",
            "mean_regret5",
            "std_regret5",
            "n_fronts",
            "is_amiga",
        ]
    ]


def _supplementary_topk_frame(
    metrics_summary: pd.DataFrame,
    primary_table: pd.DataFrame,
) -> pd.DataFrame:
    primary_order_df = primary_table.copy()
    primary_order_df["config"] = primary_order_df["config"].astype(str)
    primary_order_df["avg_rank"] = pd.to_numeric(primary_order_df["avg_rank"], errors="raise")
    primary_order_df = primary_order_df.sort_values(
        ["avg_rank", "config"],
        kind="mergesort",
    ).reset_index(drop=True)
    primary_order_df["primary_rank_order"] = primary_order_df.index + 1
    rank_lookup = primary_order_df.set_index("config")["avg_rank"].to_dict()
    order_lookup = primary_order_df.set_index("config")["primary_rank_order"].to_dict()

    rows = []
    for row in metrics_summary.itertuples(index=False):
        parsed = _parse_topk_metric(str(row.metric))
        if parsed is None:
            continue
        family, k_value = parsed
        config = str(row.model)
        rows.append(
            {
                "config": config,
                "metric_family": family,
                "k": k_value,
                "metric": str(row.metric),
                "mean": float(row.mean),
                "std": float(row.std) if not pd.isna(row.std) else pd.NA,
                "n": int(row.n),
                "metric_rank": float(row.rank) if not pd.isna(row.rank) else pd.NA,
                "avg_rank": rank_lookup.get(config, pd.NA),
                "primary_rank_order": int(order_lookup.get(config, len(order_lookup) + 1)),
                "direction": "lower_is_better" if family == "Regret" else "higher_is_better",
                "role": "primary_context" if str(row.metric) == "Regret@5" else "complementary",
            }
        )
    if not rows:
        families = ", ".join(f"{family}@k" for family in SUPPLEMENTARY_METRIC_FAMILIES)
        raise PlotPreparationError(f"metrics summary has no supported supplementary metrics: {families}")

    plot_df = pd.DataFrame(rows)
    family_order = {family: idx for idx, family in enumerate(SUPPLEMENTARY_METRIC_FAMILIES)}
    plot_df["family_order"] = plot_df["metric_family"].map(family_order)
    return (
        plot_df.sort_values(
            ["family_order", "primary_rank_order", "config", "k"],
            kind="mergesort",
        )
        .drop(columns=["family_order"])
        .reset_index(drop=True)
    )


def _supplementary_appendix_frame(topk_frame: pd.DataFrame) -> pd.DataFrame:
    id_columns = ["config", "primary_rank_order", "avg_rank"]
    metric_order = (
        topk_frame[["metric_family", "k", "metric"]]
        .drop_duplicates()
        .assign(
            family_order=lambda df: df["metric_family"].map(
                {family: idx for idx, family in enumerate(SUPPLEMENTARY_METRIC_FAMILIES)}
            )
        )
        .sort_values(["family_order", "k", "metric"], kind="mergesort")["metric"]
        .tolist()
    )
    index_df = (
        topk_frame[id_columns]
        .drop_duplicates()
        .sort_values(["primary_rank_order", "config"], kind="mergesort")
        .reset_index(drop=True)
    )
    appendix = index_df.copy()
    for metric in metric_order:
        metric_df = topk_frame[topk_frame["metric"] == metric][
            ["config", "mean", "std", "n", "metric_rank"]
        ].copy()
        metric_df = metric_df.rename(
            columns={
                "mean": f"{metric}_mean",
                "std": f"{metric}_std",
                "n": f"{metric}_n",
                "metric_rank": f"{metric}_rank",
            }
        )
        appendix = appendix.merge(metric_df, on="config", how="left")
    return appendix


def _parse_topk_metric(metric: str) -> tuple[str, int] | None:
    for family in SUPPLEMENTARY_METRIC_FAMILIES:
        prefix = f"{family}@"
        if metric.startswith(prefix):
            raw_k = metric.removeprefix(prefix)
            if raw_k.isdigit():
                return family, int(raw_k)
    return None


def _hyperparameter_rank_color_scale(
    plot_df: pd.DataFrame,
) -> tuple[mcolors.Colormap, mcolors.Normalize, float | None]:
    ranks = pd.to_numeric(plot_df["avg_rank"], errors="raise")
    rank_min = float(ranks.min())
    rank_max = float(ranks.max())
    norm = mcolors.Normalize(vmin=rank_min, vmax=rank_max)
    if "p_value" not in plot_df.columns or rank_max <= rank_min:
        return sns.light_palette("#6A3D9A", as_cmap=True, reverse=True), norm, None

    p_values = pd.to_numeric(plot_df["p_value"], errors="coerce")
    significant_ranks = ranks[p_values < 0.05]
    if significant_ranks.empty:
        return sns.light_palette("#6A3D9A", as_cmap=True, reverse=True), norm, None

    threshold = float(significant_ranks.min())
    if not rank_min < threshold < rank_max:
        return sns.light_palette("#6A3D9A", as_cmap=True, reverse=True), norm, threshold

    transition = (threshold - rank_min) / (rank_max - rank_min)
    left = max(0.0, transition - 0.001)
    right = min(1.0, transition + 0.001)
    if left == 0.0:
        right = min(1.0, 0.002)
    if right == 1.0:
        left = max(0.0, 0.998)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "amiga_hyperparameter_rank_significance",
        [
            (0.0, "#4B1F75"),
            (left, "#C7B6DD"),
            (right, "#BCE4D8"),
            (1.0, "#007A66"),
        ],
    )
    return cmap, norm, threshold


def _ordered_hyperparameter_groups(plot_df: pd.DataFrame) -> list[str]:
    order_df = (
        plot_df[["base_config", "model_type", "label"]]
        .drop_duplicates()
        .assign(
            model_order=lambda df: df["model_type"].map(_MODEL_ORDER).fillna(len(_MODEL_ORDER)),
            label_order=lambda df: df["label"].map(_LABEL_ORDER).fillna(len(_LABEL_ORDER)),
        )
        .sort_values(["model_order", "label_order", "base_config"], kind="mergesort")
    )
    return order_df["base_config"].astype(str).tolist()


def _hyperparameter_group_palette(n_colors: int) -> list[Any]:
    if n_colors <= len(HYPERPARAMETER_GROUP_COLORS):
        return list(sns.color_palette(HYPERPARAMETER_GROUP_COLORS[:n_colors]))
    return list(sns.color_palette(HYPERPARAMETER_GROUP_COLORS, n_colors=n_colors))


def _add_rank_significance_marker(colorbar: Any, threshold: float | None) -> None:
    if threshold is None:
        return
    norm = colorbar.mappable.norm
    if not norm.vmin < threshold < norm.vmax:
        return
    colorbar.ax.axhline(threshold, color="#222222", linewidth=1.05)
    colorbar.ax.annotate(
        "p=0.05",
        xy=(-0.60, threshold),
        xycoords=colorbar.ax.get_yaxis_transform(),
        xytext=(0, 5),
        textcoords="offset points",
        va="bottom",
        ha="right",
        fontsize=7.5,
        color="#222222",
        clip_on=False,
    )


def _hyperparameter_parameter_trace_frame(plot_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in plot_df.itertuples(index=False):
        values = _parse_param_tag_values(str(row.param_tag))
        for parameter, raw_value in values.items():
            numeric_value = _parameter_numeric_value(raw_value)
            if numeric_value is None:
                continue
            rows.append(
                {
                    "config": str(row.config),
                    "base_config": str(row.base_config),
                    "group_label": str(row.group_label),
                    "parameter": parameter,
                    "parameter_label": _parameter_display_label(parameter),
                    "parameter_short_label": _parameter_short_label(parameter),
                    "parameter_value": numeric_value,
                    "std_regret5": float(row.std_regret5),
                    "mean_regret5": float(row.mean_regret5),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "config",
                "base_config",
                "group_label",
                "parameter",
                "parameter_label",
                "parameter_short_label",
                "parameter_value",
                "parameter_norm",
                "std_regret5",
                "mean_regret5",
            ]
        )

    trace_df = pd.DataFrame(rows)
    keep_keys = []
    for (group, parameter), param_df in trace_df.groupby(["base_config", "parameter"], sort=False):
        if param_df["parameter_value"].nunique(dropna=True) > 1:
            keep_keys.append((group, parameter))
    if not keep_keys:
        return trace_df.iloc[0:0].assign(parameter_norm=pd.Series(dtype=float))

    keep_index = pd.MultiIndex.from_tuples(keep_keys, names=["base_config", "parameter"])
    trace_df = trace_df[
        pd.MultiIndex.from_frame(trace_df[["base_config", "parameter"]]).isin(keep_index)
    ].copy()
    trace_df["parameter_norm"] = trace_df.groupby(
        ["base_config", "parameter"],
        sort=False,
    )["parameter_value"].transform(_normalize_parameter_values)
    return trace_df


def _add_hyperparameter_parameter_marginals(
    top_ax: plt.Axes,
    right_ax: plt.Axes,
    trace_df: pd.DataFrame,
    *,
    group_order: Sequence[str],
    group_palette: Mapping[str, Any],
) -> None:
    if trace_df.empty:
        for axis in (top_ax, right_ax):
            axis.axis("off")
        return

    parameter_colors = _hyperparameter_parameter_colors(
        trace_df,
        group_order=group_order,
        group_palette=group_palette,
    )
    for group in group_order:
        group_df = trace_df[trace_df["base_config"] == group]
        if group_df.empty:
            continue
        parameters = _ordered_unique(group_df, "parameter", order_map={})
        for parameter in parameters:
            color = parameter_colors[(group, parameter)]
            param_df = group_df[group_df["parameter"] == parameter]
            x_values, y_values = _smooth_parameter_trace(
                param_df["mean_regret5"].to_numpy(dtype=float),
                param_df["parameter_norm"].to_numpy(dtype=float),
            )
            if len(x_values) >= 2:
                top_ax.plot(
                    x_values,
                    y_values,
                    color=color,
                    linewidth=1.35,
                    alpha=0.82,
                    zorder=3,
                )
            y_axis_values, x_norm_values = _smooth_parameter_trace(
                param_df["std_regret5"].to_numpy(dtype=float),
                param_df["parameter_norm"].to_numpy(dtype=float),
            )
            if len(y_axis_values) >= 2:
                right_ax.plot(
                    x_norm_values,
                    y_axis_values,
                    color=color,
                    linewidth=1.0,
                    alpha=0.55,
                    zorder=3,
                )

    top_ax.set_ylim(-0.06, 1.06)
    top_ax.set_ylabel("Norm.\nparam", fontsize=8)
    top_ax.set_yticks([0.0, 0.5, 1.0])
    top_ax.tick_params(axis="x", bottom=False)
    top_ax.tick_params(axis="y", labelsize=7)
    top_ax.grid(axis="both", alpha=0.16)

    right_ax.set_xlim(-0.06, 1.06)
    right_ax.set_xlabel("Norm.\nparam", fontsize=8)
    right_ax.set_xticks([0.0, 0.5, 1.0])
    right_ax.tick_params(axis="y", left=False)
    right_ax.tick_params(axis="x", labelsize=7)
    right_ax.grid(axis="both", alpha=0.16)


def _add_hyperparameter_parameter_legend(
    legend_ax: plt.Axes,
    trace_df: pd.DataFrame,
    *,
    group_order: Sequence[str],
    group_palette: Mapping[str, Any],
) -> None:
    legend_ax.axis("off")
    if trace_df.empty:
        return

    parameter_colors = _hyperparameter_parameter_colors(
        trace_df,
        group_order=group_order,
        group_palette=group_palette,
    )
    legend_ax.text(
        0.0,
        0.98,
        "Hyperparameter colors",
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color="#222222",
        transform=legend_ax.transAxes,
    )
    visible_groups = [
        group
        for group in group_order
        if not trace_df[trace_df["base_config"] == group].empty
    ]
    if not visible_groups:
        return
    y_positions = np.linspace(0.78, 0.24, num=len(visible_groups))
    for group, y_position in zip(visible_groups, y_positions):
        group_df = trace_df[trace_df["base_config"] == group]
        group_label = str(group_df["group_label"].iloc[0])
        legend_ax.text(
            0.0,
            y_position,
            group_label,
            ha="left",
            va="center",
            fontsize=6.8,
            color=mcolors.to_hex(group_palette[group]),
            fontweight="bold",
            transform=legend_ax.transAxes,
        )
        item_y = y_position - 0.12
        x_position = 0.0
        for parameter in _ordered_unique(group_df, "parameter", order_map={}):
            color = parameter_colors[(group, parameter)]
            short_label = _parameter_short_label(parameter)
            legend_ax.plot(
                [x_position, x_position + 0.08],
                [item_y, item_y],
                color=color,
                linewidth=1.8,
                solid_capstyle="round",
                transform=legend_ax.transAxes,
                clip_on=False,
            )
            legend_ax.text(
                x_position + 0.095,
                item_y,
                short_label,
                ha="left",
                va="center",
                fontsize=6.5,
                color=color,
                fontweight="bold",
                transform=legend_ax.transAxes,
            )
            x_position += 0.24
            if x_position > 0.72:
                x_position = 0.0
                item_y -= 0.10


def _match_normalized_marginal_axis_lengths(
    fig: plt.Figure,
    top_ax: plt.Axes,
    right_ax: plt.Axes,
) -> None:
    """Make the two normalized-parameter axes physically comparable."""
    top_box = top_ax.get_position()
    right_box = right_ax.get_position()
    fig_width, fig_height = fig.get_size_inches()
    target_width = top_box.height * fig_height / fig_width
    right_margin = max(0.0, 1.0 - right_box.x0)
    new_width = min(target_width, right_margin)
    if new_width <= 0.0:
        return
    right_ax.set_position(
        [
            right_box.x0,
            right_box.y0,
            new_width,
            right_box.height,
        ]
    )


def _hyperparameter_parameter_colors(
    trace_df: pd.DataFrame,
    *,
    group_order: Sequence[str],
    group_palette: Mapping[str, Any],
) -> dict[tuple[str, str], Any]:
    colors: dict[tuple[str, str], Any] = {}
    for group in group_order:
        group_df = trace_df[trace_df["base_config"] == group]
        if group_df.empty:
            continue
        parameters = _ordered_unique(group_df, "parameter", order_map={})
        palette = _parameter_subtone_palette(group_palette[group], len(parameters))
        for color, parameter in zip(palette, parameters):
            colors[(group, parameter)] = color
    return colors


def _smooth_parameter_trace(axis_values: np.ndarray, norm_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite_mask = np.isfinite(axis_values) & np.isfinite(norm_values)
    axis_values = axis_values[finite_mask]
    norm_values = norm_values[finite_mask]
    if axis_values.size == 0:
        return np.array([]), np.array([])

    order = np.argsort(axis_values, kind="mergesort")
    axis_values = axis_values[order]
    norm_values = norm_values[order]
    unique_axis = np.unique(axis_values)
    if unique_axis.size < 3:
        centers = []
        means = []
        for axis_value in unique_axis:
            value_mask = axis_values == axis_value
            centers.append(float(axis_value))
            means.append(float(np.mean(norm_values[value_mask])))
        return np.asarray(centers), np.asarray(means)

    n_grid = min(90, max(28, axis_values.size * 3))
    grid = np.linspace(float(axis_values.min()), float(axis_values.max()), n_grid)
    axis_range = max(float(np.ptp(axis_values)), 1e-12)
    bandwidth = max(axis_range * 0.09, float(np.std(axis_values, ddof=1)) * 0.18, 1e-12)
    smoothed = []
    for grid_value in grid:
        distances = (axis_values - grid_value) / bandwidth
        weights = np.exp(-0.5 * distances * distances)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            smoothed.append(float("nan"))
            continue
        smoothed.append(float(np.sum(weights * norm_values) / weight_sum))
    smoothed_array = np.clip(np.asarray(smoothed, dtype=float), 0.0, 1.0)
    finite = np.isfinite(smoothed_array)
    return grid[finite], smoothed_array[finite]


def _normalize_parameter_values(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    value_min = float(numeric.min())
    value_max = float(numeric.max())
    if not np.isfinite(value_min) or not np.isfinite(value_max) or value_max <= value_min:
        return pd.Series(np.zeros(len(numeric)), index=values.index)
    return (numeric - value_min) / (value_max - value_min)


def _parameter_numeric_value(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parameter_display_label(parameter: str) -> str:
    labels = {
        "d": "depth",
        "l2": "L2 regularization",
        "lr": "learning rate",
        "mcs": "min child samples",
        "mcw": "min child weight",
        "md": "max depth",
        "nl": "num leaves",
        "ss": "subsample",
    }
    return labels.get(parameter, parameter)


def _parameter_short_label(parameter: str) -> str:
    labels = {
        "d": "D",
        "l2": "L2",
        "lr": "lr",
        "mcs": "MCS",
        "mcw": "MCW",
        "md": "MD",
        "nl": "NL",
        "ss": "SS",
    }
    return labels.get(parameter, parameter)


def _parameter_subtone_palette(base_color: Any, n_colors: int) -> list[Any]:
    if n_colors <= 0:
        return []
    base_rgb = np.asarray(mcolors.to_rgb(base_color))
    white_rgb = np.ones(3)
    black_rgb = np.zeros(3)
    start = white_rgb * 0.46 + base_rgb * 0.54
    end = black_rgb * 0.22 + base_rgb * 0.78
    if n_colors == 1:
        return [mcolors.to_hex(base_rgb)]
    return [
        mcolors.to_hex(start + (end - start) * (idx / (n_colors - 1)))
        for idx in range(n_colors)
    ]


def _add_group_density_blob(
    ax: plt.Axes,
    group_df: pd.DataFrame,
    *,
    facecolor: Any,
    edgecolor: Any,
) -> None:
    x = group_df["mean_regret5"].astype(float).to_numpy()
    y = group_df["std_regret5"].astype(float).to_numpy()
    if len(group_df) < 2:
        return

    x_range = max(float(np.ptp(x)), 1e-9)
    y_range = max(float(np.ptp(y)), 1e-9)
    if len(group_df) >= 4 and x_range > 1e-9 and y_range > 1e-9:
        try:
            _add_manual_group_kde(
                ax,
                x=x,
                y=y,
                facecolor=facecolor,
                edgecolor=edgecolor,
            )
            return
        except (ValueError, np.linalg.LinAlgError):
            pass

    _add_compact_group_hull(
        ax,
        x=x,
        y=y,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )


def _add_manual_group_kde(
    ax: plt.Axes,
    *,
    x: np.ndarray,
    y: np.ndarray,
    facecolor: Any,
    edgecolor: Any,
) -> None:
    x_range = max(float(np.ptp(x)), 1e-9)
    y_range = max(float(np.ptp(y)), 1e-9)
    x_pad = max(x_range * 1.10, float(np.std(x, ddof=1)) * 1.35, 1e-9)
    y_pad = max(y_range * 1.10, float(np.std(y, ddof=1)) * 1.35, 1e-9)
    x_grid = np.linspace(float(np.min(x)) - x_pad, float(np.max(x)) + x_pad, 180)
    y_grid = np.linspace(float(np.min(y)) - y_pad, float(np.max(y)) + y_pad, 180)
    xx, yy = np.meshgrid(x_grid, y_grid)

    kde = gaussian_kde(np.vstack([x, y]), bw_method=0.42)
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    density = np.asarray(density, dtype=float)
    finite_density = density[np.isfinite(density) & (density > 0.0)]
    if finite_density.size == 0:
        raise ValueError("empty KDE density")

    density_max = float(np.max(finite_density))
    outer_level = density_max * 0.13
    if not 0.0 < outer_level < density_max:
        raise ValueError("invalid KDE contour levels")
    levels = np.linspace(outer_level, density_max, 7)

    ax.contourf(
        xx,
        yy,
        density,
        levels=levels,
        colors=[mcolors.to_rgba(facecolor, alpha=0.23)],
        antialiased=True,
        zorder=1,
    )
    ax.contour(
        xx,
        yy,
        density,
        levels=levels,
        colors=[mcolors.to_hex(edgecolor)],
        alpha=0.44,
        linewidths=0.65,
        zorder=2,
    )


def _add_compact_group_hull(
    ax: plt.Axes,
    *,
    x: np.ndarray,
    y: np.ndarray,
    facecolor: Any,
    edgecolor: Any,
) -> None:
    points = np.column_stack([x, y])
    if len(points) == 2:
        ax.plot(x, y, color=edgecolor, linewidth=5.5, alpha=0.20, solid_capstyle="round", zorder=1)
        ax.plot(x, y, color=edgecolor, linewidth=1.2, alpha=0.42, solid_capstyle="round", zorder=2)
        return

    try:
        hull = ConvexHull(points)
    except QhullError:
        ax.scatter(x, y, s=430, color=facecolor, alpha=0.18, edgecolor=edgecolor, linewidth=1.0, zorder=1)
        return

    hull_points = points[hull.vertices]
    center = hull_points.mean(axis=0)
    padded = center + (hull_points - center) * 1.08
    patch = Polygon(
        padded,
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.1,
        alpha=0.22,
        zorder=1,
    )
    ax.add_patch(patch)


def _add_hyperparameter_group_legend(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    group_order: Sequence[str],
    group_palette: Mapping[str, Any],
) -> None:
    handles = []
    labels = []
    for group in group_order:
        group_df = plot_df[plot_df["base_config"] == group]
        if group_df.empty:
            continue
        handles.append(
            Patch(
                facecolor=group_palette[group],
                edgecolor=group_palette[group],
                alpha=0.28,
            )
        )
        labels.append(str(group_df["group_label"].iloc[0]))
    if not handles:
        return

    ax.legend(
        handles,
        labels,
        title="Model + label mode",
        loc="upper left",
        bbox_to_anchor=(0.012, 0.985),
        ncol=1,
        frameon=True,
        borderaxespad=0.0,
        borderpad=0.45,
        labelspacing=0.42,
        handlelength=1.15,
        handletextpad=0.45,
        fontsize=8,
        title_fontsize=8,
    )


def _hyperparameter_display_label(config: str) -> str:
    parts = config.split("__")
    if len(parts) >= 3:
        base = " / ".join(parts[:2]).replace("_", " ")
        tag = " / ".join(parts[2:]).replace("_", " ")
        return f"{base} | {tag}"
    return config.replace("__", " / ").replace("_", " ")


def _parse_hyperparameter_config(config: str) -> dict[str, str]:
    parts = config.split("__")
    if len(parts) < 3:
        raise PlotPreparationError(
            "phase 02 hyperparameter config ids must follow "
            "'<model>__<label_mode>__<param_tag>': "
            f"{config}"
        )
    model_type = parts[0]
    label = _format_label_mode("__".join(parts[:2]))
    param_tag = "__".join(parts[2:])
    return {
        "model_type": model_type,
        "label": label,
        "base_config": "__".join(parts[:2]),
        "group_label": f"{model_type} | {label}",
        "param_tag": param_tag,
        "param_label": _compact_param_label(model_type, param_tag),
        "point_label": _point_param_label(model_type, param_tag),
    }


def _format_label_mode(base_config: str) -> str:
    _, label = _parse_screening_config(base_config)
    return label


def _compact_param_label(model_type: str, param_tag: str) -> str:
    values = _parse_param_tag_values(param_tag)
    if model_type == "LGBMRanker":
        return f"NL{values.get('nl', '?')} M{values.get('mcs', '?')} | eta={values.get('lr', '?')}"
    if model_type == "XGBRanker":
        return (
            f"D{values.get('md', '?')} S{values.get('ss', '?')} | "
            f"W{values.get('mcw', '?')} eta={values.get('lr', '?')}"
        )
    if model_type == "CatBoostRanker":
        return f"D{values.get('d', '?')} L2={values.get('l2', '?')} | eta={values.get('lr', '?')}"
    return param_tag


def _point_param_label(model_type: str, param_tag: str) -> str:
    values = _parse_param_tag_values(param_tag)
    lr = _short_decimal_label(values.get("lr", "?"))
    if model_type == "LGBMRanker":
        return f"L={values.get('nl', '?')}|M={values.get('mcs', '?')}|r={lr}"
    if model_type == "XGBRanker":
        return (
            f"D={values.get('md', '?')}|S={_short_decimal_label(values.get('ss', '?'))}|"
            f"W={values.get('mcw', '?')}|r={lr}"
        )
    if model_type == "CatBoostRanker":
        return f"D={values.get('d', '?')}|L={values.get('l2', '?')}|r={lr}"
    return param_tag.replace("_", "|")


def _parse_param_tag_values(param_tag: str) -> dict[str, str]:
    tokens = param_tag.split("_")
    values: dict[str, str] = {}
    for token in tokens:
        if token.startswith("mcs"):
            values["mcs"] = token.removeprefix("mcs")
        elif token.startswith("mcw"):
            values["mcw"] = token.removeprefix("mcw")
        elif token.startswith("md"):
            values["md"] = token.removeprefix("md")
        elif token.startswith("nl"):
            values["nl"] = token.removeprefix("nl")
        elif token.startswith("ss"):
            values["ss"] = _tag_float_label(token.removeprefix("ss"))
        elif token.startswith("lr"):
            values["lr"] = _tag_float_label(token.removeprefix("lr"))
        elif token.startswith("l2"):
            values["l2"] = token.removeprefix("l2")
        elif token.startswith("d") and token[1:].isdigit():
            values["d"] = token.removeprefix("d")
    return values


def _short_decimal_label(value: str) -> str:
    if value.startswith("0."):
        return value.removeprefix("0")
    if value == "1.0":
        return "1"
    return value


def _tag_float_label(value: str) -> str:
    if value == "10":
        return "1.0"
    if value.startswith("0") and len(value) > 1:
        return f"0.{value[1:]}"
    if value.startswith("1") and len(value) > 1:
        return f"1.{value[1:]}"
    return value


def _format_p_value(value: Any) -> str:
    if pd.isna(value):
        return "winner"
    numeric = float(value)
    if numeric < 0.001:
        return "p<0.001"
    return f"p={numeric:.3f}"


def _optional_plot_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _p_value_significance(value: Any) -> str:
    if pd.isna(value):
        return "winner"
    return "significant" if float(value) < 0.05 else "not_significant"


def _p_value_color(value: Any) -> str:
    significance = _p_value_significance(value)
    if significance == "significant":
        return "#D55E00"
    if significance == "not_significant":
        return "#0072B2"
    return "#111111"


def _feature_set_display_label(feature_set: str) -> str:
    labels = {
        "full": "Full AMIGA",
        "objectives_only": "Objectives only",
        "technique_weights_only": "Technique weights only",
        "expression_only": "Expression only",
        "network_only": "Network only",
        "no_objectives": "No objectives",
        "no_technique_weights": "No technique weights",
        "no_expression": "No expression",
        "no_network": "No network",
    }
    return labels.get(feature_set, feature_set.replace("_", " ").title())


def _baseline_display_label(config: str) -> str:
    if config == "AMIGA_final":
        return "AMIGA"
    if config.startswith("objective__"):
        objective = config.removeprefix("objective__").replace("_", " ")
        objective_labels = {
            "degreedistribution": "degree distribution",
            "dynamicity": "dynamicity",
            "metricdistribution": "metric distribution",
            "motifs": "motifs",
            "quality": "quality",
            "reducenonessentialsinteractions": "reduce nonessential interactions",
        }
        objective = objective_labels.get(objective, objective)
        return f"Single objective: {objective}"
    labels = {
        "objective_mean_rank": "Objective mean rank",
        "objective_normalized_mean": "Normalized objective mean",
        "objective_ideal_l2": "Ideal L2 distance",
        "objective_topsis": "TOPSIS",
        "objective_vikor": "VIKOR",
        "objective_augmented_tchebycheff": "Augmented Tchebycheff",
        "objective_tchebycheff": "Objective Tchebycheff",
    }
    return labels.get(config, config.replace("_", " ").title())


def _parse_screening_config(config: str) -> tuple[str, str]:
    parts = config.split("__", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise PlotPreparationError(
            "phase 01 model-screening config ids must follow '<model>__<label_mode>': "
            f"{config}"
        )
    model, raw_label = parts
    if raw_label.startswith("quantiles_q"):
        label = raw_label.replace("_q", " q", 1)
    else:
        label = raw_label.replace("_", " ")
    return model, label


def _ordered_unique(df: pd.DataFrame, column: str, *, order_map: Mapping[str, int]) -> list[str]:
    values = list(dict.fromkeys(str(value) for value in df[column].dropna()))
    return sorted(values, key=lambda value: (order_map.get(value, 999), value))


def _plot_manifest_payload(
    context: CaseContext,
    *,
    phase: str,
    plots_dir: Path,
    primary_rank_table: Path,
    statistical_tests: Path | None,
    primary_table: pd.DataFrame,
    include_secondary: bool,
    planned_outputs: Mapping[str, Mapping[str, Path]],
    generated_outputs: Mapping[str, Mapping[str, Path]],
) -> dict[str, Any]:
    has_generated_outputs = any(generated_outputs.values())
    return {
        **common_manifest_fields(context),
        "manifest_type": "plot_phase",
        "phase": phase,
        "status": "generated" if has_generated_outputs else "prepared",
        "plot_generation_status": "generated" if has_generated_outputs else "phase_specific_plots_pending",
        "plots_dir": repo_relative(plots_dir),
        "primary_rank_table": repo_relative(primary_rank_table),
        "supporting_inputs": {
            "statistical_tests": repo_relative(statistical_tests)
        } if statistical_tests is not None else {},
        "primary_rank_table_rows": int(len(primary_table)),
        "primary_rank_table_columns": list(primary_table.columns),
        "secondary_outputs_policy": {
            "included": bool(include_secondary),
            "role": SUPPLEMENTARY_POLICY,
            "output_dir": repo_relative(plots_dir / "supplementary"),
        },
        "planned_outputs": {
            group: {
                file_type: repo_relative(path)
                for file_type, path in outputs.items()
            }
            for group, outputs in planned_outputs.items()
        },
        "generated_outputs": {
            group: {
                file_type: repo_relative(path)
                for file_type, path in outputs.items()
            }
            for group, outputs in generated_outputs.items()
        },
    }
