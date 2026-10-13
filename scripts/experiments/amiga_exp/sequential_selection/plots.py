"""Adapt earlier screening/scatter designs to topology-weighted inner selection."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scripts.experiments.amiga_exp.plots import _hyperparameter_group_palette
from .models import FAMILIES, LABELS


def save(fig, prefix):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(prefix.with_suffix('.png'), dpi=180, bbox_inches='tight')
    fig.savefig(prefix.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def screening(table, prefix, title):
    """Keep the original label-by-family heatmap, replace obsolete inference."""
    pivot = table.pivot(index='family', columns='label', values='mean_regret5').reindex(
        index=FAMILIES, columns=LABELS)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    sns.heatmap(pivot, ax=ax, annot=True, fmt='.4f', cmap='YlGnBu', linewidths=.6,
                cbar_kws={'label': 'Inner mean Regret@5 (lower is better)'})
    for row in table[table['selected_within_family']].itertuples():
        ax.text(LABELS.index(row.label)+.88, FAMILIES.index(row.family)+.23, '*',
                color='black', size=16, weight='bold', bbox=dict(facecolor='white', alpha=.8, pad=0))
    ax.axvline(6, color='black', lw=1)
    ax.set_title(title + '\nPhase 1 · * selected label within family')
    ax.set_xlabel('Label mode (last two columns: controls)')
    ax.set_ylabel('Model family')
    ax.tick_params(axis='x', rotation=30)
    save(fig, prefix)


def tuning(table, prefix, title):
    """Reuse the mean/dispersion view, marking all three retained families."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.3), squeeze=False)
    palette = dict(zip(FAMILIES, _hyperparameter_group_palette(3)))
    for ax, (arm, data) in zip(axes[0], table.groupby('arm', sort=True)):
        for family in FAMILIES:
            part = data[data['family'] == family]
            ax.scatter(part['mean_regret5'], part['std_regret5'], color=palette[family],
                       label=family, s=28, alpha=.65)
            selected = part[part['selected_within_family']]
            ax.scatter(selected['mean_regret5'], selected['std_regret5'], marker='*',
                       color=palette[family], edgecolor='black', s=180, zorder=5)
        ax.set_title(arm)
        ax.set_xlabel('Inner mean Regret@5')
        ax.set_ylabel('SD across validation topologies')
        ax.ticklabel_format(style='plain', useOffset=False)
    axes[0, -1].legend(fontsize=8)
    fig.suptitle(title + '\nPhase 2 · * selected parameters within family; all families continue')
    save(fig, prefix)


def feature_curves(table, prefix, title):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.3), squeeze=False)
    palette = dict(zip(FAMILIES, _hyperparameter_group_palette(3)))
    for ax, (arm, data) in zip(axes[0], table.groupby('arm', sort=True)):
        for family in FAMILIES:
            part = data[data['family'] == family].sort_values('n_features')
            ax.plot(part['n_features'], part['mean_regret5'], '-o', color=palette[family], label=family)
            selected = part[part['selected_within_family']]
            ax.scatter(selected['n_features'], selected['mean_regret5'], marker='*',
                       color=palette[family], edgecolor='black', s=180, zorder=5)
        ax.set_title(arm)
        ax.set_xlabel('Retained columns')
        ax.set_ylabel('Inner mean Regret@5')
        ax.ticklabel_format(style='plain', useOffset=False)
    axes[0, -1].legend(fontsize=8)
    fig.suptitle(title + '\nPhase 3 · parameters fixed; * selected feature budget within family')
    save(fig, prefix)


def stability_plot(table, prefix, case):
    # All columns are shown: correlated alternatives need not have high frequency.
    data = table.assign(procedure=table['family'] + ' / ' + table['arm'])
    pivot = data.pivot(index='feature', columns='procedure', values='selected_frequency')
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False, kind='stable').index]
    fig, ax = plt.subplots(figsize=(13, max(8, .19*len(pivot))))
    sns.heatmap(pivot, ax=ax, cmap='Blues', vmin=0, vmax=1,
                cbar_kws={'label': 'Fraction of 15 overlapping inner-training fits'})
    ax.set_title(case + '\nPhase 3 · column inclusion at the internally selected feature budget')
    ax.set_ylabel('Predictor column')
    ax.set_xlabel('Family / formulation')
    ax.tick_params(axis='y', labelsize=7)
    save(fig, prefix)


def make_plots(table, stability, output):
    output = Path(output)
    sns.set_theme(style='whitegrid', context='paper')
    for (case, outer), data in table.groupby(['case', 'outer_fold']):
        directory = output / case / f'outer-{outer}'
        title = f'{case} · outer fold {outer} training complement · inner validation only'
        for stage, name, render in [('phase1', 'label_screening', screening),
                                    ('phase2', 'hyperparameters', tuning),
                                    ('phase3', 'feature_curves', feature_curves)]:
            part = data[data['stage'] == stage]
            directory.mkdir(parents=True, exist_ok=True)
            part.to_csv(directory / f'{name}.csv', index=False)
            render(part, directory / name, title)
    for case, data in stability.groupby('case'):
        stability_plot(data, output / case / 'column_stability', case)
