# Supervised comparison reports

`amiga-exp report supervised` regenerates the five-formulation statistical
presentation from a completed evaluation summary. It fits no models, changes no
predictions and leaves the complete source method set intact.

## Command

From the repository root, after the top-10% pipeline completes:

```bash
scripts/experiments/amiga-exp report supervised \
  --summary experiments/top10-classification/full-001/outer-summary \
  --audit-manifest experiments/sequential-selection/summaries/full-001/summary_manifest.json \
  --audit-manifest experiments/outer-evaluation/summaries/evaluation-001/manifest.json \
  --audit-manifest experiments/top5-classification/full-001/outer-summary/manifest.json \
  --output experiments/reports/supervised-001
```

The source summary must contain `topology_metrics.csv` and a complete
`manifest.json` whose artifact hashes cover that CSV. All artifacts listed in
each supplied manifest are verified before reporting. Additional audit manifests
are repeatable; they verify saved selection/evaluation summaries without fitting
or reselecting anything. These file checks supplement the pipelines' original
audits of prediction coverage and metric recomputation.

The destination must be new and outside the source summary. Use `--no-figures`
for CSV/LaTeX tables only. The benchmark default is 87 paired topologies per case;
`--expected-topologies N` permits an explicit alternative count for another input.
Missing methods, missing metric values, duplicate observations, invalid ranges
and incomplete paired coverage stop reporting.

## Statistical specification

The fixed method set is `ranking`, `reg_aupr`, `clf_top05`, `clf_top10` and
`clf_top20`. AMIGA (`ranking`) is the control even when another method has the
best mean rank. Analyze generators separately. The input already averages
seed-specific metrics within fronts and conditions within topology; this command
does not reconstruct those averages from scores or treat seeds as independent.

Rank each metric within topology, giving average ranks to ties after rounding to
12 decimal places. Minimize Regret; maximize BestAUPR and Hit. Smaller ranks are
better for every metric. Compute raw metric means before rounding.

For Regret@5, compute SciPy's tie-corrected asymptotic Friedman chi-square test.
For each of the four alternatives, use the two-sided normal approximation

```text
z = (mean_rank_alternative - mean_rank_AMIGA) / sqrt(k * (k + 1) / (6 * N))
```

Here `k=5` and `N` is the paired topology count. Adjust the four p-values with
Holm separately within each generator. Also save joint adjustment over all
generators' control comparisons (eight for the two benchmark cases). A rejection
flag requires both omnibus and adjusted p-values below 0.05. If all methods tie
on every topology, record statistic 0 and p-value 1.

Secondary metric ranks are descriptive. The analysis is exploratory and
additional to the frozen Wilcoxon-Holm tests; it does not overwrite them. A first
mean rank does not demonstrate superiority. Benchmark reuse, overlapping
training sets and the exploratory classifier thresholds limit inference.

## Outputs

| File | Content |
| --- | --- |
| `mean_ranks_all_metrics.csv` | Mean ranks of Regret, BestAUPR and Hit at k=1,3,5,10 |
| `raw_metric_means.csv` | Corresponding unrounded raw means |
| `friedman_omnibus.csv` | Per-case Regret@5 statistics and p-values |
| `regret5_posthoc_holm.csv` | Rank differences, z, raw p, within-case and joint Holm p-values |
| `<case>-regret5.csv`, `<case>-regret5-ranks.csv` | Paired values and ranks used by the primary test |
| `<case>-publication-table.csv`, `.tex` | Regret@5, Regret@1, Hit@1, Hit@5 ranks and primary Holm p-values |
| `mean_rank_tables.pdf`, `.png` | Compact comparison tables |
| `manifest.json` | Workflow/library versions, input hashes, audit scope, implementation hashes and output hashes |

The manifest records the selected method set and numerical policy explicitly.
Source summaries continue to contain every method evaluated in their frozen
contracts. Regenerating a report in a new directory never changes those results.

For the original four phase designs, including separate supervised and
objective-selector comparison panels, use [`report figures`](figures.md).
It displays these five-method ranks and Holm values in panel A, together with
the original metric means, and generates tuning and column-selection figures
for every reported formulation.
