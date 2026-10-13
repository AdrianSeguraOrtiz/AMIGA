# Computational cost of prediction

This section measures scoring and ordering of an already prepared front with a
saved AMIGA model. It is separate from Study 1's phases and Study 2's biological
support. Model loading, matrix preparation, feature generation, training,
network reconstruction and upstream inference are excluded.

The completed measurement uses 300 candidates, 104 predictor columns and four
threads. Five repetitions follow one warm-up. Record prediction seconds,
sorting seconds and their sum, alongside row/feature counts and thread count.
Do not describe these values as end-to-end computational cost or scalability.

## Read the existing measurements

```bash
scripts/experiments/amiga-exp costs report
```

The default source is `experiments/supplementary/fixed-001/application`.
`--source` accepts another directory with `prediction_costs.csv` and a verified
`manifest.json`. Reading the report does not predict or fit models.
After `organize`, the same CSV is available at
`experiments/studies/costs/prediction_costs.csv`.

## Repeat timing with a saved model

```bash
scripts/experiments/amiga-exp costs predict \
  --model experiments/studies/real/model/ranking-seed-1201.model.json \
  --data experiments/BIO-INSIGHT/real-world/tcga_brca/amiga/data_real.csv \
  --output experiments/costs/prediction-new --threads 4
scripts/experiments/amiga-exp costs report \
  --source experiments/costs/prediction-new
```

The model option identifies JSON metadata next to the native model file.
The prepared CSV supplies `front_id`, `item_id` and the exact recorded predictor
columns. Target labels are not read. The output destination must be new.
These commands do not perform model selection, training or feature extraction.
Measure on an otherwise idle machine and retain the hardware/thread information.
