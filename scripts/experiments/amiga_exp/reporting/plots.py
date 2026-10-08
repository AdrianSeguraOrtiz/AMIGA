"""Compact mean-rank tables from already computed reporting tables."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .supervised import LABELS, MAIN_METRICS, METHODS


def plot_tables(tables: dict, output: Path) -> None:
    omnibus = tables["friedman_omnibus.csv"]
    fig, axes = plt.subplots(len(omnibus), 1, figsize=(11.5, 2.95 * len(omnibus)), squeeze=False)
    try:
        for ax, (_, result) in zip(axes[:, 0], omnibus.iterrows(), strict=True):
            case = result["case"]
            table = tables[f"{case}-publication-table.csv"]
            ps = table["p_Holm_Regret@5_vs_AMIGA"]
            cells = [[LABELS[m]] + [f"{table.loc[m, v]:.3f}" for v in MAIN_METRICS]
                     + ["—" if m == "ranking" else f"{ps[m]:.4f}"] for m in METHODS]
            ax.axis("off")
            artist = ax.table(cellText=cells, colLabels=["Method", *MAIN_METRICS, "Holm p (Regret@5)"],
                              colWidths=[.32, .13, .13, .11, .11, .20], loc="center", cellLoc="center")
            artist.auto_set_font_size(False)
            artist.set_fontsize(10)
            artist.scale(1, 1.65)
            for (row, col), cell in artist.get_celld().items():
                cell.set_edgecolor("#AAAAAA")
                cell.set_linewidth(.4)
                if row == 0:
                    cell.set_facecolor("#EEEEEE")
                    cell.get_text().set_weight("bold")
                if col == 0:
                    cell.get_text().set_ha("left")
            for col, metric in enumerate(MAIN_METRICS, 1):
                for row, method in enumerate(METHODS, 1):
                    if np.isclose(table.loc[method, metric], table[metric].min(), rtol=0, atol=1e-12):
                        artist[row, col].get_text().set_weight("bold")
            ax.set_title(f"{case} — Friedman p = {result['p_value']:.4f}; {result['n_topologies']} topologies",
                         fontsize=12, pad=8)
        fig.text(.5, .015, "Lower mean rank is better; seed and condition metrics averaged first.\n"
                 "Two-sided mean-rank z comparisons vs AMIGA; Holm over 4 contrasts per case. Exploratory analysis.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .065, 1, 1))
        for extension in ("pdf", "png"):
            fig.savefig(output / f"mean_rank_tables.{extension}", dpi=180, bbox_inches="tight")
    finally:
        plt.close(fig)
