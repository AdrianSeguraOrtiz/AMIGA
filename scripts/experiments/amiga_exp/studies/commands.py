"""Commands organized into benchmark phases, real application and prediction costs."""
import json
from pathlib import Path

import typer

from scripts.experiments.amiga_exp.sequential_selection.commands import app as selection_app
from scripts.experiments.amiga_exp.outer_evaluation.commands import app as evaluation_app
from scripts.experiments.amiga_exp.top5_classification.commands import app as top5_app
from scripts.experiments.amiga_exp.top10_classification.commands import app as top10_app
from scripts.experiments.amiga_exp.reporting import commands as reporting
from .organization import BUNDLE, FIGURES

benchmark_app = typer.Typer(no_args_is_help=True, help='Labelled benchmarks: selection, comparison and learning curves.')
learning_app = typer.Typer(no_args_is_help=True, help='Phase 5: fixed-AMIGA learning curves.')
real_app = typer.Typer(no_args_is_help=True, help='Real TCGA-BRCA application and contextual source support.')
costs_app = typer.Typer(no_args_is_help=True, help='Computational cost of prepared-front prediction and ranking.')

benchmark_app.add_typer(selection_app, name='selection', help='Phases 1–3: relevance labels, parameters, columns and family.')
benchmark_app.add_typer(evaluation_app, name='comparison', help='Phase 4: held-out selector comparison.')
evaluation_app.add_typer(top5_app, name='top5', help='Top-5% classification sensitivity analysis.')
evaluation_app.add_typer(top10_app, name='top10', help='Top-10% classification sensitivity analysis.')
benchmark_app.add_typer(learning_app, name='learning', help='Phase 5: fixed-AMIGA learning curves.')
benchmark_app.command('figures')(reporting.figures)
benchmark_app.command('statistics')(reporting.supervised)


def _report(bundle, name):
    from scripts.experiments.amiga_exp.reporting.supervised import verify_manifest
    folder = Path(bundle).resolve() / name
    manifest, _ = verify_manifest(folder / 'manifest.json')
    if manifest.get('status') != 'complete':
        raise ValueError('Expected a complete report')
    return folder, manifest


@learning_app.command('status')
def phase5_status(bundle: Path = typer.Option(BUNDLE, exists=True, file_okay=False)):
    from scripts.experiments.amiga_exp.supplementary.runner import read_run
    from scripts.experiments.amiga_exp.supplementary.execution import completed_job
    _, _, jobs = read_run(bundle / 'run', verify_environment=False)
    learning = [job for job in jobs if job['context_id'].startswith('learning/')]
    done = [job for job in learning if completed_job(bundle / 'run', job)]
    summary = bundle / 'summary/manifest.json'
    report_status = _report(bundle, 'summary')[1]['status'] if summary.exists() else 'pending'
    typer.echo(json.dumps(dict(study=1, phase=5, completed_jobs=len(done), total_jobs=len(learning),
        completed_fits=sum(j['planned_fits'] for j in done), planned_fits=sum(j['planned_fits'] for j in learning),
        report_status=report_status, results=str(bundle / 'summary')), indent=2))


@learning_app.command('report')
def phase5_report(bundle: Path = typer.Option(BUNDLE, exists=True, file_okay=False),
                  output: Path | None = typer.Option(None, help='Optional new directory to regenerate metrics and figures without fitting.')):
    """Inspect saved curves, or render their verified metrics in a new directory."""
    import pandas as pd
    source, source_manifest = _report(bundle, 'summary')
    if output is not None:
        from tempfile import TemporaryDirectory
        import shutil
        from scripts.experiments.amiga_exp.supplementary.summary import plot_curves
        from scripts.experiments.amiga_exp.runtime import sha256, write_json
        from . import __version__
        output = output.resolve()
        if output.exists() or output.is_relative_to(source):
            raise ValueError('Use a new phase-5 report directory outside its source')
        source_digest = sha256(source / 'manifest.json')
        curves = pd.read_csv(source / 'learning_curves.csv')
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix='.phase5-report-', dir=output.parent) as temporary:
            staged = Path(temporary) / 'report'
            staged.mkdir()
            for path in source.iterdir():
                if path.is_file() and path.suffix in ('.csv', '.json') and path.name != 'manifest.json':
                    shutil.copyfile(path, staged / path.name)
            plot_curves(curves, staged)
            _report(bundle, 'summary')
            if sha256(source / 'manifest.json') != source_digest:
                raise ValueError('Source summary changed while rendering')
            write_json(staged / 'manifest.json', dict(status='complete', workflow_version=__version__,
                study=1, phase=5, no_model_fits=True, metrics_recomputed=False,
                source_manifest_sha256=source_digest, scope=source_manifest.get('scope'),
                artifacts={p.name: sha256(p) for p in staged.iterdir() if p.is_file()}))
            staged.rename(output)
        folder = output
    else:
        folder = source
    curves = pd.read_csv(folder / 'learning_curves.csv')
    typer.echo(curves[curves.metric.isin(['Regret@5', 'Hit@5'])].to_json(orient='records', indent=2))


@real_app.command('status')
def study2_status(bundle: Path = typer.Option(BUNDLE, exists=True, file_okay=False)):
    from scripts.experiments.amiga_exp.supplementary.runner import read_run
    from scripts.experiments.amiga_exp.supplementary.execution import completed_job
    _, _, jobs = read_run(bundle / 'run', verify_environment=False)
    job = next(j for j in jobs if j['context_id'] == 'deployment')
    completed = completed_job(bundle / 'run', job)
    path = bundle / 'application/manifest.json'
    status = _report(bundle, 'application')[1]['status'] if path.exists() else 'pending'
    typer.echo(json.dumps(dict(study=2, model_status='complete' if completed else 'pending',
                              application_status=status, results=str(bundle / 'application')), indent=2))


@real_app.command('report')
def study2_report(bundle: Path = typer.Option(BUNDLE, exists=True, file_okay=False)):
    """Inspect the original five-selector real-case analysis."""
    import pandas as pd
    folder, _ = _report(bundle, 'application')
    typer.echo(pd.read_csv(folder / 'real_world_source_support_top1.csv').to_json(orient='records', indent=2))


@real_app.command('apply')
def study2_apply(output: Path = typer.Option(..., help='New destination; existing evidence is preserved.'),
                 bundle: Path = typer.Option(BUNDLE, exists=True, file_okay=False),
                 case_dir: Path = typer.Option(Path('experiments/BIO-INSIGHT/real-world/tcga_brca'), exists=True, file_okay=False)):
    """Apply the saved deployment model and regenerate support; never train."""
    from scripts.experiments.amiga_exp.supplementary.deployment import apply
    result = apply(bundle / 'run', case_dir, output)
    typer.echo(json.dumps(dict(status=result['status'], output=str(output), model_fits=0,
                              selectors=result['selectors']), indent=2))


@costs_app.command('report')
def costs_report(source: Path = typer.Option(BUNDLE / 'application', exists=True, file_okay=False)):
    """Summarize saved measurements without repeating predictions."""
    import pandas as pd
    from scripts.experiments.amiga_exp.reporting.supervised import verify_manifest
    manifest, _ = verify_manifest(source / 'manifest.json')
    if manifest.get('status') != 'complete':
        raise ValueError('Expected complete prediction measurements')
    frame = pd.read_csv(source / 'prediction_costs.csv')
    columns = ['prediction_seconds', 'ranking_seconds', 'total_seconds']
    if frame.empty or (frame[columns] < 0).any().any() or frame[columns].isna().any().any():
        raise ValueError('Invalid prediction-cost measurements')
    typer.echo(json.dumps(dict(scope='prepared-front prediction and ranking only',
        repeats=len(frame), rows=int(frame.rows.iloc[0]), features=int(frame.features.iloc[0]),
        threads=int(frame.threads.iloc[0]), seconds=frame[columns].agg(['mean', 'median', 'min', 'max']).to_dict()), indent=2))


@costs_app.command('predict')
def costs_predict(model: Path = typer.Option(..., exists=True, dir_okay=False, help='Saved ranking-seed-1201.model.json metadata.'),
                  data: Path = typer.Option(..., exists=True, dir_okay=False, help='Prepared candidate-front CSV.'),
                  output: Path = typer.Option(..., help='New output directory.'),
                  threads: int = typer.Option(4, min=1, max=8)):
    """Measure only prediction/sorting with an existing native AMIGA model."""
    import pandas as pd
    from scripts.experiments.amiga_exp.supplementary.deployment import load_native, prediction_costs
    from scripts.experiments.amiga_exp.runtime import sha256, write_json
    if output.exists():
        raise ValueError('Use a new prediction-cost destination')
    inputs = {str(p.resolve()): sha256(p) for p in (model, data)}
    info = json.loads(model.read_text())
    frame = pd.read_csv(data, usecols=['front_id', 'item_id', *info['feature_columns']])
    if frame.empty or frame.front_id.nunique() != 1 or frame.duplicated(['front_id', 'item_id']).any():
        raise ValueError('Expected one nonempty prepared front with unique candidate IDs')
    native = load_native(model.parent, info)
    _, timings = prediction_costs(native, info['procedure']['family'],
                                  frame[info['feature_columns']].to_numpy(), frame.item_id.to_numpy(), threads)
    if any(sha256(Path(p)) != digest for p, digest in inputs.items()):
        raise ValueError('Prediction inputs changed during measurement')
    output.mkdir(parents=True)
    timings.to_csv(output / 'prediction_costs.csv', index=False)
    write_json(output / 'manifest.json', dict(status='complete', scope='prepared-front prediction and ranking only',
        model_fits=0, model_sha256=info['model_sha256'], inputs_sha256=inputs,
        artifacts={'prediction_costs.csv': sha256(output / 'prediction_costs.csv')}))
    typer.echo(json.dumps(dict(status='complete', output=str(output), model_fits=0, repeats=len(timings)), indent=2))


def organize(output: Path = typer.Option(Path('experiments/studies'), help='New directory of relative links grouped by study.'),
             bundle: Path = typer.Option(BUNDLE, exists=True, file_okay=False),
             figures: Path = typer.Option(FIGURES, exists=True, file_okay=False)):
    """Create an index of verified results without moving or recomputing them."""
    from .organization import create_view
    result = create_view(output, bundle=bundle, figures=figures)
    typer.echo(json.dumps(dict(status=result['status'], output=str(output), links=len(result['links']), model_fits=0), indent=2))
