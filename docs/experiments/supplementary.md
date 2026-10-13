# Shared learning-curve and deployment execution

After benchmark phases 1–4 are complete, the fixed learning curves and one
real-data deployment fit share an execution bundle. This is orchestration of
phase 5 and the real application, rather than an additional scientific phase.

```bash
scripts/experiments/amiga-exp prepare \
  --output experiments/supplementary/fixed-contract.json
scripts/experiments/amiga-exp run \
  --contract experiments/supplementary/fixed-contract.json \
  --output experiments/supplementary/fixed-001 --jobs 16 --threads 4
scripts/experiments/amiga-exp status --run experiments/supplementary/fixed-001
```

Use a new destination. The bundle comprises 90 AMIGA learning-curve jobs,
with five seeds each, and one deployment fit: 451 fits in total. It does not
repeat label, parameter, feature or family selection. The full-size endpoints
reuse the phase-4 predictions.

The pipeline verifies frozen dependencies and source/data identities, runs the
fits, generates learning-curve summaries, applies the deployment model to the
real front and measures prediction/sorting time. `--resume` resumes an unchanged
contract; an explicit technical retry preserves previous attempts.

| Local directory | Contents |
| --- | --- |
| `run/` | Contract, job plan, state, model files and job receipts |
| `summary/` | Learning-curve metrics, conditional intervals and figures |
| `application/` | Real ranking, selected networks, evidence support and prediction costs |

Inspect these analyses independently with `benchmark learning`, `real` and
`costs`. `organize` creates relative links into a new local results index.
Generated outputs remain Git-ignored; they are not deposited in the repository.
