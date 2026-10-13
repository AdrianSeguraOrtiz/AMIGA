"""Study grouping, read-only reports, timing isolation and verified navigation."""
import hashlib
import io
import os
import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from scripts.experiments.amiga_exp.cli import app
from scripts.experiments.amiga_exp.studies.organization import BUNDLE, FIGURES, create_view


def test_public_commands_share_the_project_version():
    import amiga
    from scripts.experiments.amiga_exp.version import __version__
    assert __version__ == amiga.__version__
    result = CliRunner().invoke(app, ['--version'])
    assert result.exit_code == 0 and result.stdout.strip() == f'AMIGA {amiga.__version__}'


def signed(folder, files, **extra):
    folder.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, payload in files.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        artifacts[name] = hashlib.sha256(payload).hexdigest()
    (folder / 'manifest.json').write_text(json.dumps(dict(status='complete', artifacts=artifacts, **extra)))


def cost_csv():
    return pd.DataFrame([dict(method='AMIGA', repeat=i, rows=300, features=104, threads=4,
        prediction_seconds=.006, ranking_seconds=.00002, total_seconds=.00602) for i in range(5)]).to_csv(index=False).encode()


@pytest.mark.parametrize('command,words', [
    (['benchmark', '--help'], ['selection', 'comparison', 'learning', 'figures', 'statistics']),
    (['benchmark', 'learning', '--help'], ['status', 'report']),
    (['real', '--help'], ['status', 'apply', 'report']),
    (['costs', '--help'], ['predict', 'report']),
])
def test_study_command_help(command, words):
    result = CliRunner().invoke(app, command)
    assert result.exit_code == 0, result.output
    assert all(word in result.stdout for word in words)


def test_run_routes_to_joint_engine(tmp_path, monkeypatch):
    from scripts.experiments.amiga_exp.supplementary import pipeline
    contract = tmp_path / 'contract.json'
    contract.write_text('{}')
    calls = []
    monkeypatch.setattr(pipeline, 'run_all', lambda *args, **kwargs: calls.append((args, kwargs)))
    output = tmp_path / 'new-run'
    result = CliRunner().invoke(app, ['run', '--contract', str(contract),
        '--output', str(output), '--jobs', '4', '--threads', '2'])
    assert result.exit_code == 0, result.output
    assert calls == [((contract, output), dict(jobs=4, threads=2, resume=False, retry_failed=False))]
    assert not output.exists()


def test_phase5_status_counts_only_learning_jobs(tmp_path, monkeypatch):
    from scripts.experiments.amiga_exp.supplementary import runner, execution
    jobs = [dict(context_id='learning/a', planned_fits=5),
            dict(context_id='learning/b', planned_fits=5), dict(context_id='deployment', planned_fits=1)]
    monkeypatch.setattr(runner, 'read_run', lambda *args, **kwargs: ({}, {}, jobs))
    monkeypatch.setattr(execution, 'completed_job', lambda run, job: ({}, tmp_path))
    signed(tmp_path / 'summary', {'learning_curves.csv': b'case,metric,mean\nBIO-INSIGHT,Regret@5,0.1\n'})
    result = CliRunner().invoke(app, ['benchmark', 'learning', 'status', '--bundle', str(tmp_path)])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data['total_jobs'] == data['completed_jobs'] == 2
    assert data['planned_fits'] == data['completed_fits'] == 10


def test_reading_reports_does_not_refit_or_change_evidence(tmp_path, monkeypatch):
    from scripts.experiments.amiga_exp.sequential_selection import models
    monkeypatch.setattr(models, 'fit', lambda *args, **kwargs: pytest.fail('Report must not train'))
    curves = b'case,method,training_size,metric,mean\nBIO-INSIGHT,ranking,10,Regret@5,0.1\nBIO-INSIGHT,ranking,10,Hit@5,0.3\n'
    signed(tmp_path / 'summary', {'learning_curves.csv': curves})
    signed(tmp_path / 'application', {'prediction_costs.csv': cost_csv(),
        'real_world_source_support_top1.csv': b'method,candidate\nAMIGA,147\n'})
    before = {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    for command in [['benchmark', 'learning', 'report', '--bundle', str(tmp_path)],
                    ['real', 'report', '--bundle', str(tmp_path)],
                    ['costs', 'report', '--source', str(tmp_path / 'application')]]:
        result = CliRunner().invoke(app, command)
        assert result.exit_code == 0, result.output
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    (tmp_path / 'application/prediction_costs.csv').write_bytes(b'changed')
    assert CliRunner().invoke(app, ['costs', 'report', '--source', str(tmp_path / 'application')]).exit_code != 0


def test_cost_prediction_uses_saved_columns_and_does_not_train(tmp_path, monkeypatch):
    from scripts.experiments.amiga_exp.supplementary import deployment
    from scripts.experiments.amiga_exp.sequential_selection import models
    monkeypatch.setattr(models, 'fit', lambda *args, **kwargs: pytest.fail('Timing must not train'))
    metadata = tmp_path / 'model.json'
    metadata.write_text(json.dumps(dict(procedure=dict(family='XGBoost'), feature_columns=['x'], model_sha256='recorded')))
    data = tmp_path / 'data.csv'
    pd.DataFrame(dict(front_id=[1, 1], item_id=[2, 3], x=[.3, .7], AUPR=[.9, .1])).to_csv(data, index=False)
    native = object()
    monkeypatch.setattr(deployment, 'load_native', lambda directory, info: native)
    calls = []
    def predict(model, family, X, ids, threads):
        calls.append((model, family, X.tolist(), ids.tolist(), threads))
        return X[:, 0], pd.read_csv(io.BytesIO(cost_csv()))
    monkeypatch.setattr(deployment, 'prediction_costs', predict)
    output = tmp_path / 'timing'
    result = CliRunner().invoke(app, ['costs', 'predict', '--model', str(metadata), '--data', str(data), '--output', str(output)])
    assert result.exit_code == 0, result.output
    assert calls == [(native, 'XGBoost', [[.3], [.7]], [2, 3], 4)]
    assert json.loads((output / 'manifest.json').read_text())['model_fits'] == 0
    assert CliRunner().invoke(app, ['costs', 'predict', '--model', str(metadata), '--data', str(data), '--output', str(output)]).exit_code != 0


def test_study_index_retains_artifacts_and_relative_links_after_move(tmp_path):
    root = tmp_path / 'workspace'
    bundle = root / BUNDLE
    signed(root / FIGURES, {'phase4.pdf': b'vector figure'})
    signed(bundle / 'summary', {'curve.pdf': b'learning curve'})
    native = b'saved native model'
    digest = hashlib.sha256(native).hexdigest()
    signed(bundle / 'application', {'prediction_costs.csv': cost_csv()}, model_sha256=digest)
    model = bundle / 'run/jobs/BIO-INSIGHT/deployment/final/attempt-001'
    model.mkdir(parents=True)
    (model / 'ranker.cbm').write_bytes(native)
    metadata = model / 'ranking-seed-1201.model.json'
    metadata.write_text(json.dumps(dict(model_file='ranker.cbm', model_sha256=digest)))
    (model / 'result.json').write_text(json.dumps(dict(status='complete', artifacts={
        'ranker.cbm': digest, metadata.name: hashlib.sha256(metadata.read_bytes()).hexdigest()})))
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    output = root / 'experiments/studies'
    result = create_view(output, root=root)
    assert result['model_fits'] == 0 and not result['scientific_artifacts_relocated']
    assert (output / 'benchmark/learning/curve.pdf').read_bytes() == b'learning curve'
    assert (output / 'real/model/ranker.cbm').read_bytes() == native
    assert all((root / name).read_bytes() == data for name, data in before.items())
    assert all(not Path(os.readlink(p)).is_absolute() for p in output.rglob('*') if p.is_symlink())
    moved = tmp_path / 'relocated'
    root.rename(moved)
    assert (moved / 'experiments/studies/costs/prediction_costs.csv').read_bytes() == cost_csv()
    with pytest.raises(ValueError, match='new study-index'):
        create_view(moved / 'experiments/studies', root=moved)


def test_phase5_render_copies_verified_metrics_without_refitting(tmp_path, monkeypatch):
    from scripts.experiments.amiga_exp.supplementary import summary
    from scripts.experiments.amiga_exp.sequential_selection import models
    monkeypatch.setattr(models, 'fit', lambda *args, **kwargs: pytest.fail('Rendering must not fit'))
    monkeypatch.setattr(summary, 'summarize', lambda *args, **kwargs: pytest.fail('Rendering must not recompute metrics'))
    payload = b'case,method,training_size,metric,mean\nBIO-INSIGHT,ranking,10,Regret@5,0.1\n'
    signed(tmp_path / 'bundle/summary', {'learning_curves.csv': payload}, scope='conditional fixed recipe')
    calls = []
    def plot(curves, output):
        calls.append(curves.to_csv(index=False))
        (output / 'curve.pdf').write_bytes(b'vector PDF')
    monkeypatch.setattr(summary, 'plot_curves', plot)
    output = tmp_path / 'rendered'
    result = CliRunner().invoke(app, ['benchmark', 'learning', 'report', '--bundle', str(tmp_path / 'bundle'), '--output', str(output)])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert (output / 'learning_curves.csv').read_bytes() == payload
    report = json.loads((output / 'manifest.json').read_text())
    assert report['no_model_fits'] and not report['metrics_recomputed']
    assert report['scope'] == 'conditional fixed recipe'
    assert not list(tmp_path.glob('.phase5-report-*'))
