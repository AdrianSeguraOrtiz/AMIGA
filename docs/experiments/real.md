# Study 2: real TCGA-BRCA application

Study 2 is separate from the five labelled-benchmark phases. It applies the
updated AMIGA model to the existing prepared TCGA-BRCA front and retains the
original five-selector source-support analysis.

## Deployment model

Reuse the BIO-INSIGHT phase-4 recipe with the lowest recorded inner-validation
Regret@5, breaking a tie by outer-fold index. Fix its family, relevance labels,
parameters and exact predictor columns, then fit on all 104 benchmark fronts
(87 topologies) with seed 1201. Outer-test metrics and real-case regulatory
support do not choose the recipe. There is no additional parameter search.

The joint [`run`](supplementary.md) performs this one deployment fit
alongside phase 5. Native model bytes, schema, seed and integrity metadata are
preserved. `real apply` uses this saved model; it never trains one.

## Real front and source support

Reuse the existing processed expression matrix/front and original base networks.
No inference, evolutionary optimization or patient-data download is repeated.
The prepared matrix used 1,106 TCGA-BRCA primary-tumour samples and 500 variable
genes; preprocessing and source identities remain in the recorded case files.

The original five selectors are AMIGA, ReduceNEI, mean objective rank, TOPSIS and
metric distribution. Reconstruct their Top1 networks and use the same fixed
CollecTRI, DoRothEA, TRRUSTv2 and JASPAR evidence at top-250, and Cistrome BRCA-COR
at top-5000. Report TF-target and TF-source support. Coverage is incomplete;
absent pairs are not verified negatives, and support is not gold-standard accuracy.

## Commands

```bash
scripts/experiments/amiga-exp real status
scripts/experiments/amiga-exp real report
```

Both commands read the completed `experiments/supplementary/fixed-001` bundle by
default. To regenerate ranking and support in a new destination with the saved
model, without fitting:

```bash
scripts/experiments/amiga-exp real apply \
  --output experiments/real/tcga-brca-new
```

Use `--bundle` for another complete joint execution. Application regeneration
checks the recorded fitting environment, repository context and fixed scientific
inputs; it is intended for that execution context. After restoring evidence in
another workspace, `real report` inspects it without those fitting prerequisites.
Native models can also be used directly by `costs predict`. Fresh full
recomputation uses a newly frozen joint execution in its own workspace.
Regeneration also records the original five prediction-timing repetitions; the
[cost section](prediction-costs.md) describes their scope.

After `organize`, the organized paths are
`experiments/studies/real/tcga-brca/` for results and
`experiments/studies/real/model/` for the native model and metadata.
