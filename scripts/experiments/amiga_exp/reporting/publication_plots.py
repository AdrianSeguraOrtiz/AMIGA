"""Refresh the original heatmap/scatter/lollipop designs with current results."""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import fill

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

from scripts.experiments.amiga_exp.plots import (
    _add_group_density_blob, _add_hyperparameter_parameter_legend,
    _add_hyperparameter_parameter_marginals, _hyperparameter_group_palette,
    _match_normalized_marginal_axis_lengths, _parameter_display_label, _parameter_short_label,
    _baseline_display_label,
)
from .figure_data import FAMILIES, LABELS
from .supervised import LABELS as METHOD_LABELS

GREEN, GRAY, BLUE, ORANGE = "#009E73", "#7D848C", "#0072B2", "#D55E00"
FAMILY_COLORS = dict(zip(FAMILIES, _hyperparameter_group_palette(3)))
PREFIXES = {"BIO-INSIGHT": "bio", "MO-GENECI": "mogeneci"}


def style():
    sns.set_theme(style="whitegrid", context="paper", rc={
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
        "axes.titlesize": 13, "axes.labelsize": 11, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "pdf.fonttype": 42, "ps.fonttype": 42})


def save(fig, prefix: Path):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    try:
        for extension in ("pdf", "png"):
            fig.savefig(prefix.with_suffix("."+extension), dpi=240, bbox_inches="tight")
    finally:
        plt.close(fig)


def screening(table, prefix, case):
    """Original blue heatmap; replace obsolete p-values with selection frequency."""
    style()
    values = table.pivot(index="family", columns="label", values="mean_regret5").loc[list(FAMILIES), list(LABELS)]
    counts = table.pivot(index="family", columns="label", values="selected_folds").loc[list(FAMILIES), list(LABELS)]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    cmap = sns.light_palette("#2A6FBB", as_cmap=True, reverse=True)
    ordinary = values.iloc[:, :6]
    sns.heatmap(values, cmap=cmap, vmin=ordinary.min().min(), vmax=ordinary.max().max(),
                linewidths=.7, linecolor="white", ax=ax,
                cbar_kws=dict(label="Mean inner Regret@5", extend="max", shrink=.8))
    for y in range(3):
        for x in range(8):
            count = int(counts.iloc[y, x])
            ax.text(x+.5, y+.38, f"{values.iloc[y,x]:.4f}", ha="center", va="center",
                    color="#111111", fontsize=11, weight="bold" if count else "normal")
            ax.text(x+.5, y+.68, f"selected {count}/5", ha="center", va="center", fontsize=9,
                    weight="bold" if count else "normal", color="#111111" if count else "#666666",
                    bbox=dict(facecolor="white", alpha=.65, edgecolor="none", pad=1))
    ax.axvline(6, color="#343434", linewidth=1.35)
    ax.set_xticklabels([x.replace("_q", " q").replace("_", " ") for x in LABELS], rotation=30, ha="right")
    ax.set(xlabel="Relevance label (last two columns: controls)", ylabel="Model family",
           title=f"{case} · Phase 1: relevance labels\nInner validation; equal-weight mean over five outer training complements")
    fig.tight_layout()
    save(fig, prefix)


def parameter_traces(table):
    keys = {"LightGBM": {"num_leaves":"nl", "min_child_samples":"mcs", "learning_rate":"lr"},
            "XGBoost": {"max_depth":"md", "subsample":"ss", "min_child_weight":"mcw", "learning_rate":"lr"},
            "CatBoost": {"depth":"d", "l2_leaf_reg":"l2", "learning_rate":"lr"}}
    rows = []
    for row in table.itertuples():
        params = json.loads(row.parameters)
        for key, short in keys[row.family].items():
            rows.append(dict(config=row.config_id, base_config=row.family, group_label=row.family,
                             parameter=short, parameter_label=_parameter_display_label(short),
                             parameter_short_label=_parameter_short_label(short), parameter_value=params[key],
                             mean_regret5=row.mean_regret5, std_regret5=row.std_regret5))
    trace = pd.DataFrame(rows)
    for (_, _), part in trace.groupby(["base_config", "parameter"]):
        low, high = part.parameter_value.min(), part.parameter_value.max()
        trace.loc[part.index, "parameter_norm"] = 0 if low == high else (part.parameter_value-low)/(high-low)
    return trace


def tuning(table, prefix, case, arm):
    """Reuse original density clouds, normalized parameter marginals and palette."""
    style()
    fig = plt.figure(figsize=(12.5, 8.4))
    grid = fig.add_gridspec(2, 4, width_ratios=[.055, .10, 1., .30], height_ratios=[.24, 1.], wspace=.10, hspace=.06)
    cax, top, ax, right, legend = (fig.add_subplot(grid[1,0]), fig.add_subplot(grid[0,2]),
                                   fig.add_subplot(grid[1,2]), fig.add_subplot(grid[1,3]), fig.add_subplot(grid[0,3]))
    top.sharex(ax)
    right.sharey(ax)
    for family in FAMILIES:
        _add_group_density_blob(ax, table[table.family == family], facecolor=FAMILY_COLORS[family], edgecolor=FAMILY_COLORS[family])
    points = ax.scatter(table.mean_regret5, table.std_regret5, c=table.diagnostic_rank,
                        cmap=sns.light_palette("#6A3D9A", as_cmap=True, reverse=True),
                        s=60, edgecolors="white", linewidth=.75, zorder=4)
    retained = table[table.selected_folds > 0]
    ax.scatter(retained.mean_regret5, retained.std_regret5, s=105, facecolors="none", edgecolors="black", linewidth=1.6, zorder=5)
    trace = parameter_traces(table)
    _add_hyperparameter_parameter_marginals(top, right, trace, group_order=FAMILIES, group_palette=FAMILY_COLORS)
    _add_hyperparameter_parameter_legend(legend, trace, group_order=FAMILIES, group_palette=FAMILY_COLORS)
    cbar = fig.colorbar(points, cax=cax)
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.set_title("Mean candidate\nrank (diagnostic)", fontsize=8, pad=8)
    handles = [Patch(facecolor=c, edgecolor=c, alpha=.3, label=f) for f,c in FAMILY_COLORS.items()]
    handles.append(Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor="black", linestyle="none", label="Selected in ≥1 outer fold"))
    ax.legend(handles=handles, fontsize=9, loc="best")
    for axis in (ax, top, right):
        axis.grid(alpha=.16)
    ax.set(xlabel="Mean inner Regret@5", ylabel="Mean within-complement SD of Regret@5")
    ax.set_xlim(max(0, table.mean_regret5.min()-.15*np.ptp(table.mean_regret5)), table.mean_regret5.max()+.15*np.ptp(table.mean_regret5))
    ax.set_ylim(max(0, table.std_regret5.min()-.15*np.ptp(table.std_regret5)), table.std_regret5.max()+.15*np.ptp(table.std_regret5))
    plt.setp(top.get_xticklabels(), visible=False)
    plt.setp(right.get_yticklabels(), visible=False)
    fig.suptitle(f"{case} · Phase 2: {METHOD_LABELS[arm]}\n99 configurations, averaged over five outer training complements; no significance coding", fontsize=12, y=1.025)
    fig.subplots_adjust(left=.08, right=.98, bottom=.10, top=.98)
    _match_normalized_marginal_axis_lengths(fig, top, right)
    save(fig, prefix)


def feature_matrix(importance, stability, *, count=None, extremes=False):
    values = importance.pivot(index="feature", columns="family", values="relative_centered_shap").loc[:, list(FAMILIES)]
    inclusion = stability.pivot(index="feature", columns="family", values="selected_frequency").loc[:, list(FAMILIES)]
    order = values.mean(axis=1).sort_values(ascending=False, kind="stable").index
    if count is not None:
        size = min(count, len(order))
        if extremes:
            highest, lowest = (size+1)//2, size//2
            order = order[:highest].append(order[-lowest:]) if lowest else order[:highest]
        else:
            order = order[:size]
    return 100*values.loc[order], 100*inclusion.loc[order]


def matrix_axes(fig, grid, importance, stability, *, count=None, annotate=True, extremes=False):
    values, inclusion = feature_matrix(importance, stability, count=count, extremes=extremes)
    first, second = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    for axis, data, cmap, title in (
        (first, values, sns.light_palette("#2A6FBB", as_cmap=True), "Relative centered\nSHAP (%)"),
        (second, inclusion, sns.light_palette(GREEN, as_cmap=True), "Selected-budget\ninclusion (%)")):
        sns.heatmap(data, ax=axis, cmap=cmap, vmin=0, vmax=100 if axis is second else None,
                    annot=annotate, fmt=(".2f" if extremes else ".1f") if axis is first else ".0f", annot_kws=dict(size=9 if count else 7),
                    cbar=False, linewidths=.5, linecolor="white", yticklabels=axis is first)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_xticklabels(["LGBM", "XGB", "Cat"], rotation=0)
        axis.tick_params(axis="y", labelsize=8.5 if count else 6.5, length=0)
        if axis is first:
            axis.set_yticklabels(axis.get_yticklabels(), rotation=0)
        if extremes and len(values) > 1:
            axis.axhline((len(values)+1)//2, color="#343434", linewidth=1.3, linestyle="--")
    return values, inclusion


def feature_selection(curves, fold_rows, importance, stability, prefix, case, arm, *, top_features=20):
    """Replace block ablation with the actual training-only column selection."""
    style()
    shown = min(top_features, importance.feature.nunique())
    fig = plt.figure(figsize=(10.8, max(7.7, .29*shown+1.8)))
    grid = fig.add_gridspec(2, 4, width_ratios=[1.05, .70, .57, .57],
                            height_ratios=[.49, .12], wspace=.10, hspace=.50)
    ax = fig.add_subplot(grid[0,0])
    for family in FAMILIES:
        color = FAMILY_COLORS[family]
        for _, part in fold_rows[fold_rows.family == family].groupby("outer_fold"):
            part = part.sort_values("fraction")
            ax.plot(100*part.fraction, part.mean_regret5, color=color, alpha=.45, linewidth=1.5)
        part = curves[curves.family == family].sort_values("fraction")
        ax.plot(100*part.fraction, part.mean_regret5, "-o", color=color, label=family, linewidth=2.8, markersize=5)
    sizes = curves.groupby("fraction").n_features.first().sort_index()
    ax.set_xticks(100*sizes.index.to_numpy(), [f"{int(100*f)}%\n({n})" for f,n in sizes.items()])
    ax.set(xlabel="Retained predictors (% and column count)", ylabel="Mean inner Regret@5",
           title="Recursive training-only selection")
    ax.grid(alpha=.18)
    ax.legend(fontsize=9, loc="best")
    counts_ax = fig.add_subplot(grid[1,0])
    counts = curves.pivot(index="family", columns="fraction", values="selected_folds").loc[list(FAMILIES), sizes.index]
    labels = counts.map(lambda value:f"{int(value)}/5")
    sns.heatmap(counts, ax=counts_ax, annot=labels, fmt="", cbar=False, vmin=0, vmax=5,
                cmap=sns.light_palette(GREEN, as_cmap=True), annot_kws=dict(size=9),
                linewidths=.7, linecolor="white")
    counts_ax.set_title("Within-family selection frequency", fontsize=10, pad=10)
    counts_ax.set(xlabel="", ylabel="")
    counts_ax.set_xticklabels([f"{int(100*f)}%" for f in sizes.index], rotation=0)
    counts_ax.set_yticklabels(list(FAMILIES), rotation=0)
    counts_ax.tick_params(axis="both", labelsize=9, length=0)
    counts_ax.xaxis.tick_top()
    matrix_axes(fig, [grid[:,2], grid[:,3]], importance, stability, count=top_features, extremes=True)
    fig.suptitle(f"{case} · Phase 3: {METHOD_LABELS[arm]}", fontsize=13)
    fig.text(.5, .02, "Lighter curves: five outer training complements; thick curves: their mean. Table: original choices out of five.\n"
             f"Right: {((shown+1)//2)} highest + {shown//2} lowest of {importance.feature.nunique()} columns by training SHAP; dashed line separates groups.\n"
             "Importance and inclusion average 15 overlapping inner-training fits per family; descriptive, not causal.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.07, right=.985, bottom=.13, top=.88)
    save(fig, prefix)


def full_feature_matrix(importance, stability, prefix, case, arm):
    style()
    n = importance.feature.nunique()
    fig = plt.figure(figsize=(10, max(7, .22*n+1.8)))
    grid = fig.add_gridspec(1, 2, wspace=.04)
    matrix_axes(fig, [grid[0], grid[1]], importance, stability)
    fig.suptitle(f"{case} · {METHOD_LABELS[arm]}\nAll {n} predictors; 15 overlapping inner-training fits per family", fontsize=12, y=1.)
    fig.subplots_adjust(left=.40, right=.985, bottom=.04, top=.97)
    save(fig, prefix)


def method_label(method):
    if method in METHOD_LABELS:
        return METHOD_LABELS[method]
    return {"objective_knee":"Knee (trade-off worthiness)", "random_uniform":"Uniform random",
            "objective__reducenonessentialsinteractions":"Single objective: nonessential edges"}.get(method, _baseline_display_label(method))


def decision_comparison(panels, prefix, case):
    """Side-by-side primary-metric panels with separate Friedman/Holm families."""
    style()
    height = max(6.8, .38*max(len(data) for _,data in panels)+1.8)
    fig = plt.figure(figsize=(12 if len(panels) > 1 else 8.2, height))
    widths = [1.12 if scope == "objectives" else 1 for scope,_ in panels]
    outer = fig.add_gridspec(1, len(panels), width_ratios=widths, wspace=.12)
    panel_titles = []
    for index, (scope, rows) in enumerate(panels):
        data = rows.sort_values(["Regret@5_mean_rank", "method"], kind="stable").reset_index(drop=True)
        label_width = 1.35 if scope == "objectives" else 1.0
        grid = outer[0,index].subgridspec(1, 3, width_ratios=[label_width, 1., .45], wspace=.04)
        label_ax, ax, p_ax = [fig.add_subplot(grid[i]) for i in range(3)]
        y = np.arange(len(data))
        for axis in (label_ax, ax, p_ax):
            axis.set_ylim(len(data)-.4, -.6)
        ranks = data["Regret@5_mean_rank"]
        reference = data.loc[data.method == "ranking", "Regret@5_mean_rank"].item()
        span = max(float(np.ptp(ranks)), .35)
        ax.axvline(reference, color=GREEN, linestyle=(0,(4,3)), alpha=.7, linewidth=1.1)
        label_ax.set_xlim(0,1)
        label_ax.axis("off")
        for row, position in zip(data.to_dict("records"), y, strict=True):
            rank, mean = row["Regret@5_mean_rank"], row["Regret@5_mean"]
            is_amiga = row["method"] == "ranking"
            label_ax.text(.98, position, fill(method_label(row["method"]), width=31),
                          ha="right", va="center", fontsize=10, linespacing=.95,
                          weight="bold" if is_amiga else "normal")
            ax.hlines(position, min(reference, rank), max(reference, rank), color="#B8BDC3", linewidth=1.5)
            ax.scatter(rank, position, marker="D" if is_amiga else "o", color=GREEN if is_amiga else GRAY,
                       edgecolor="#111111" if is_amiga else "white", s=95 if is_amiga else 55, zorder=3)
            ax.annotate(f"{rank:.2f}", (rank, position), xytext=(6,5.5), textcoords="offset points",
                        fontsize=10, va="center", weight="bold" if is_amiga else "normal")
            ax.annotate(f"μ={mean:.5f}", (rank, position), xytext=(6,-5.5), textcoords="offset points",
                        fontsize=9.5, color="#555555", va="center")
        ax.set_xlim(max(0, ranks.min()-.12*span), ranks.max()+.85*span)
        ax.set_yticks(y, [])
        ax.grid(axis="x", alpha=.20)
        ax.grid(axis="y", alpha=.08)
        ax.set_xlabel("Mean rank (lower is better)", fontsize=10)
        ax.set_title("Regret@5", fontsize=12, pad=12)
        p_ax.set_xlim(0,1)
        p_ax.set_title("Holm p", fontsize=10, pad=12)
        p_ax.axis("off")
        for position, row in zip(y, data.itertuples(), strict=True):
            label = "control" if row.method == "ranking" else "p<0.001" if row.p_holm < .001 else f"p={row.p_holm:.3f}"
            color = "#111111" if row.method == "ranking" else ORANGE if row.significant else BLUE
            p_ax.text(.05, position, label, va="center", fontsize=10, color=color,
                      weight="bold" if row.method == "ranking" else "normal")
        title = {"supervised":"Supervised formulations", "objectives":"Objective selectors and random", "all":"All deployable selectors"}[scope]
        panel_titles.append((label_ax, f"{chr(65+index)}. {title}\nFriedman p={data.friedman_p.iloc[0]:.4g}; "
                             f"N={int(data.n_topologies.iloc[0])}, k={int(data.n_methods.iloc[0])}"))
    handles = [Line2D([],[],marker="D",color="none",markerfacecolor=GREEN,markeredgecolor="black",label="AMIGA (fixed control)"),
               Line2D([],[],marker="o",color="none",markerfacecolor=GRAY,label="Comparator"),
               Line2D([],[],color=ORANGE,label="Omnibus + Holm p < 0.05"),
               Line2D([],[],color=BLUE,label="No rejection")]
    fig.suptitle(f"{case} · Phase 4: held-out topology comparison", x=.5, y=.998, fontsize=13)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5,.945), ncol=4, frameon=True, fontsize=9.5)
    fig.text(.5,.025,"Regret@5 only · μ: mean Regret@5 · Ranks and Holm correction computed separately within each panel.\n"
             "87 topology means after seed/condition averaging; exploratory inference on reused benchmarks.", ha="center", fontsize=9.5)
    fig.subplots_adjust(left=.025, right=.985, top=.76, bottom=.15)
    for index, (axis, title) in enumerate(panel_titles):
        position = axis.get_position()
        fig.text(position.x0, position.y1+.48/fig.get_figheight(), title,
                 fontsize=11, weight="bold", va="bottom", gid=f"panel-title-{index}")
    save(fig, prefix)
