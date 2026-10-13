"""Verified navigation by study without relocating recorded artifacts."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.experiments.amiga_exp.runtime import sha256, write_json
from scripts.experiments.amiga_exp.reporting.supervised import verify_manifest

REPO = Path(__file__).resolve().parents[4]
BUNDLE = Path('experiments/supplementary/fixed-001')
FIGURES = Path('experiments/reports/figures-002')


def application_model(bundle):
    """Locate the one native model used to produce the recorded real-case report."""
    bundle = Path(bundle)
    application, _ = verify_manifest(bundle / 'application/manifest.json')
    parent = bundle / 'run/jobs/BIO-INSIGHT/deployment/final'
    found = []
    for metadata in parent.glob('attempt-*/ranking-seed-1201.model.json'):
        result = json.loads((metadata.parent / 'result.json').read_text())
        if result.get('status') != 'complete':
            continue
        info = json.loads(metadata.read_text())
        if result['artifacts'].get(metadata.name) != sha256(metadata):
            raise ValueError('Deployment metadata identity differs')
        model = metadata.parent / info['model_file']
        if (sha256(model) != info['model_sha256']
                or result['artifacts'].get(model.name) != info['model_sha256']
                or application['model_sha256'] != info['model_sha256']):
            raise ValueError('Deployment model differs from the application report')
        found.append(metadata)
    if len(found) != 1:
        raise ValueError('Expected one completed deployment model')
    return found[0]


def create_view(output, *, root=REPO, bundle=BUNDLE, figures=FIGURES):
    root = Path(root).resolve()
    output = Path(output).resolve()
    bundle, figures = (root / bundle).resolve(), (root / figures).resolve()
    if output.exists():
        raise ValueError('Use a new study-index destination')
    manifests = [figures / 'manifest.json', bundle / 'summary/manifest.json',
                 bundle / 'application/manifest.json']
    for manifest in manifests:
        document, _ = verify_manifest(manifest)
        if document.get('status') != 'complete':
            raise ValueError('Study index requires completed, verified reports')
    model = application_model(bundle)
    targets = {
        'benchmark/phases-1-4': figures,
        'benchmark/learning': bundle / 'summary',
        'real/tcga-brca': bundle / 'application',
        'real/model': model.parent,
        'costs/prediction_costs.csv': bundle / 'application/prediction_costs.csv',
    }
    if any(output == p or output.is_relative_to(p) for p in targets.values()):
        raise ValueError('Study index must be outside its source artifacts')
    before = {p: sha256(p) for p in manifests}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.study-index-', dir=output.parent) as temporary:
        staged = Path(temporary) / 'index'
        staged.mkdir()
        for name, source in targets.items():
            destination = staged / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Calculate against the final location, rather than the staging folder.
            relative = os.path.relpath(source, (output / name).parent)
            destination.symlink_to(relative, target_is_directory=source.is_dir())
        result = dict(schema_version=1, status='complete', kind='study_navigation',
            links={name: os.path.relpath(source, output) for name, source in targets.items()},
            source_manifests={os.path.relpath(p, output): digest for p, digest in before.items()},
            scientific_artifacts_relocated=False, model_fits=0)
        write_json(staged / 'manifest.json', result)
        (staged / 'README.md').write_text(
            '# Experimental studies\n\n'
            '- `benchmark/phases-1-4/`: completed benchmark comparisons and feature analysis.\n'
            '- `benchmark/learning/`: fixed-configuration AMIGA learning curves.\n'
            '- `real/tcga-brca/`: updated real-case ranking and source support.\n'
            '- `real/model/`: the native deployment model and its recorded schema.\n'
            '- `costs/prediction_costs.csv`: prepared-front prediction measurements.\n\n'
            'These relative links retain the original artifact locations and manifests.\n'
            'Generated inputs, models and reports remain local and Git-ignored.\n')
        if any(sha256(p) != digest for p, digest in before.items()):
            raise ValueError('Source report changed while creating study index')
        staged.rename(output)
    for name, source in targets.items():
        if (output / name).resolve() != source:
            raise ValueError('Study-index link does not resolve to its recorded source')
    return result
