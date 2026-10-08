"""Export classification-target sensitivity with its fixed ranking reference."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .models import ARM


def make_plots(tables,destination):
    destination=Path(destination)
    metrics=('Regret@5','Regret@1','Hit@1','Hit@5')
    order=['ranking','clf_top05',ARM,'clf_top20','reg_normalized','reg_aupr']
    labels=['Selected ranking','Classification top 5%','Classification top 10%','Classification top 20%',
            'Normalized AUPR regression','Direct AUPR regression']
    central=tables['central_summary.csv']
    for case in sorted(central['case'].unique()):
        target=destination/case
        target.mkdir(parents=True)
        data=central.loc[(central['case']==case)&(central['aggregation']=='topology_macro')].set_index('method')
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for ax,metric in zip(axes.flat,metrics):
            ax.barh(labels,data.loc[order,metric],color=['#888888','#E1812C','#3274A1','#A56CC1','#6AAA64','#B15F6F'])
            ax.set_xlabel(metric)
            ax.invert_yaxis()
        fig.suptitle(f'{case}: five-seed, topology-averaged selection quality')
        fig.tight_layout()
        for extension in ('png','pdf'):
            fig.savefig(target/f'method_comparison.{extension}',dpi=180,bbox_inches='tight')
        plt.close(fig)
        pairs=tables['paired_differences.csv']
        pairs=pairs.loc[pairs['case']==case]
        fig,axes=plt.subplots(2,2,figsize=(11,7))
        for ax,metric in zip(axes.flat,metrics):
            selected=pairs.loc[pairs['metric']==metric].set_index('comparator').loc[['ranking','clf_top05','clf_top20']]
            for y,row in enumerate(selected.itertuples()):
                ax.plot([row.ci_low,row.ci_high],[y,y],color='#3274A1')
                ax.scatter(row.mean_difference,y,color='#3274A1')
            ax.axvline(0,color='black',linewidth=.8)
            ax.set_yticks([0,1,2],['Top 10% minus ranking','Top 10% minus top 5%','Top 10% minus top 20%'])
            ax.set_ylim(-.6,2.6)
            ax.set_xlabel(metric+' difference')
        fig.suptitle(f'{case}: descriptive paired differences\nConditional 95% topology bootstrap intervals')
        fig.tight_layout()
        for extension in ('png','pdf'):
            fig.savefig(target/f'paired_differences.{extension}',dpi=180,bbox_inches='tight')
        plt.close(fig)
