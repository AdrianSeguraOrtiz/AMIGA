"""Shared drawing helpers for the benchmark figures."""
from __future__ import annotations
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

HYPERPARAMETER_GROUP_COLORS = (
    "#FF6B6B",
    "#35B6E8",
    "#FDBA45",
    "#7A6FF0",
    "#8CC152",
    "#C77CFF",
)

def _hyperparameter_group_palette(n_colors: int) -> list[Any]:
    if n_colors <= len(HYPERPARAMETER_GROUP_COLORS):
        return list(sns.color_palette(HYPERPARAMETER_GROUP_COLORS[:n_colors]))
    return list(sns.color_palette(HYPERPARAMETER_GROUP_COLORS, n_colors=n_colors))


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


def _ordered_unique(df: pd.DataFrame, column: str, *, order_map: Mapping[str, int]) -> list[str]:
    values = list(dict.fromkeys(str(value) for value in df[column].dropna()))
    return sorted(values, key=lambda value: (order_map.get(value, 999), value))
