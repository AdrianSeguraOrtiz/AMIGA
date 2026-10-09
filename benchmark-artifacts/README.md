# Versioned benchmark artifacts

This directory distributes checksummed benchmark evidence for **amiga-exp
0.3.0**, independently of the `amiga-grn` PyPI package. The source tag is
`amiga-exp-v0.3.0`. Each deposit has a JSON inventory with SHA-256 and sizes for
every compressed container and restored file. Individual containers stay below
90 MiB and can be downloaded through GitHub without Git LFS.

| Deposit | Scope | Status |
| --- | --- | --- |
| `comparison-v0.3.0/` | Processed benchmark inputs, phase 1–4 figure inputs/results, saved five-formulation and objective-selector evaluation predictions | Prepared for publication with the source release |
| `supplementary-v0.3.0/` | Learning curves, native deployment models, TCGA contextual support and cost measurements | Available after the full supplementary pipeline completes |

The supplementary execution is independent of the completed benchmark phases.
A source release or a running process does not establish completed evidence.
The [workflow guide](../docs/experiments/supplementary.md) specifies selection,
seeds, metrics, intervals, monitoring and technical resumption.

## Verify and restore

After cloning this repository and installing experimental dependencies:

```bash
scripts/experiments/amiga-exp supplement verify-archive \
  --archive benchmark-artifacts/comparison-v0.3.0
scripts/experiments/amiga-exp supplement restore-archive \
  --archive benchmark-artifacts/comparison-v0.3.0 --destination .
```

Restoration preflights all files, rejects altered containers and unsafe paths,
preserves identical existing files and refuses to overwrite different ones.
Restore into another checkout or empty directory if local results differ.
The archives preserve the original `experiments/` layout. Figure inputs and
summaries can be inspected or plotted without repeating model fitting. Exact
original source manifests retain their recorded method sets and provenance;
the current supervised presentation uses five formulations. Full inner
candidate predictions are regenerable intermediates, not included in this
compact distribution.

The supplementary deposit preserves execution definitions and completion
receipts as provenance. It omits detailed inner predictions and is therefore
not a resumable working directory. Native models and their JSON metadata can
be loaded using `supplementary.deployment.load_native`; feature names and model
hashes are recorded with each estimator.

For a new full execution, use a separate source checkout with only the two
processed input CSVs restored at their documented locations; existing completed
run destinations must not be reused as fresh runs. Follow the installation and
phase order in [reproducibility.md](../docs/experiments/reproducibility.md).

The supplementary deposit omits patient-level expression matrices and original
unfiltered external resource downloads. Derived evidence has incomplete
coverage and represents contextual support. Source attribution and original
data terms remain applicable; the software license does not relicense external
data. Feature-profile receipts identify the omitted matrix by hash.

## Create a deposit

```bash
scripts/experiments/amiga-exp supplement archive-comparison \
  --output benchmark-artifacts/comparison-v0.3.0
scripts/experiments/amiga-exp supplement archive \
  --run experiments/supplementary/full-001 \
  --output benchmark-artifacts/supplementary-v0.3.0
```

Creation requires new destinations and verifies all included source identities.
Deposits are complete only when their manifests say `complete`; source and
artifact tags are immutable references, never moved after publication.
