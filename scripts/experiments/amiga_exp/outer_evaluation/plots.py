"""Export the complete method comparison and paired conditional intervals."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def make_plots(tables, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    central = tables['central_summary.csv']
    paired = tables['paired_differences.csv']
    names = {'ranking': 'AMIGA (selected ranker)', 'reg_aupr': 'AUPR regression',
             'reg_normalized': 'Normalized AUPR regression', 'clf_top20': 'Top-20% classification'}
    for case in sorted(central['case'].unique()):
        directory = output / case
        directory.mkdir()
        data = central.loc[(central['case'] == case) & (central['aggregation'] == 'topology_macro')]
        data = data.sort_values('Regret@5', ascending=False).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh([names.get(m, m) for m in data['method']], data['Regret@5'],
                color=['#c44e52' if m == 'ranking' else '#4c72b0' for m in data['method']])
        ax.set(xlabel='Mean Regret@5 (lower is better)', title=f'{case}: held-out topology evaluation')
        ax.grid(axis='x', alpha=.2)
        fig.tight_layout()
        for extension in ('png', 'pdf'):
            fig.savefig(directory / f'method_comparison.{extension}', dpi=180)
        plt.close(fig)
        for metric in ('Regret@5', 'Regret@1'):
            data = paired.loc[(paired['case'] == case) & (paired['metric'] == metric)]
            data = data.sort_values('mean_difference').reset_index(drop=True)
            fig, ax = plt.subplots(figsize=(9, 7))
            positions = np.arange(len(data))
            ax.hlines(positions, data['ci_low'], data['ci_high'], color='#4c72b0')
            ax.plot(data['mean_difference'], positions, 'o', color='#4c72b0')
            ax.set_yticks(positions, [names.get(m, m) for m in data['comparator']])
            ax.axvline(0, color='black', linewidth=.8)
            ax.set(xlabel=f'Ranking − comparator: {metric} (negative favors ranking)',
                   title=f'{case}: mean paired differences\n95% conditional topology bootstrap intervals')
            ax.grid(axis='x', alpha=.2)
            fig.tight_layout()
            for extension in ('png', 'pdf'):
                fig.savefig(directory / f'paired_{metric.lower().replace("@", "_")}.{extension}', dpi=180)
            plt.close(fig)
