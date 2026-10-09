"""Updated AMIGA real-case ranking, original five selectors and prediction costs."""
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.experiments.amiga_exp.grouped_validation.pilot import sha256, write_json
from scripts.experiments.amiga_exp.real_world_validation import (
    REPORTED_SOURCES, infer_grn_dir, reconstruct_selected_networks,
    build_reported_source_support_table, select_reported_candidates,
    write_selected_candidates, build_reported_markdown, validate_ranked_frame,
)
from scripts.experiments.amiga_exp.sequential_selection import models
from .execution import completed_job
from .runner import read_run, verify_sources


def load_native(directory, info):
    from catboost import CatBoostRanker
    from lightgbm import Booster
    from xgboost import XGBRanker
    path = Path(directory) / info['model_file']
    if sha256(path) != info['model_sha256']:
        raise ValueError('Native model identity differs')
    family = info['procedure']['family']
    if family == 'LightGBM':
        return SimpleNamespace(booster_=Booster(model_file=str(path)))
    if family not in ('CatBoost', 'XGBoost'):
        raise ValueError('Unknown AMIGA model family')
    model = CatBoostRanker() if family == 'CatBoost' else XGBRanker()
    model.load_model(str(path))
    return model


def prediction_costs(model, family, X, item_ids, threads):
    """Time an already prepared matrix; data loading and model loading are excluded."""
    if family == 'XGBoost':
        model.set_params(n_jobs=threads)
    values = models.scores(model, family, 'ranking', X, threads)
    timings = []
    for repeat in range(5):
        start = time.perf_counter()
        again = models.scores(model, family, 'ranking', X, threads)
        predicted = time.perf_counter()
        order = np.lexsort((item_ids, -again))
        ranked = time.perf_counter()
        np.testing.assert_allclose(values, again, rtol=1e-12, atol=1e-12)
        if len(order) != len(X):
            raise ValueError('Incomplete ranked front')
        timings.append(dict(method='AMIGA', repeat=repeat, rows=len(X), features=X.shape[1],
            threads=threads, prediction_seconds=predicted-start, ranking_seconds=ranked-predicted,
            total_seconds=ranked-start))
    return values, pd.DataFrame(timings)


def apply(run, case_dir, output):
    run, case_dir, output = map(lambda p: Path(p).resolve(), (run, case_dir, output))
    manifest, c, jobs = read_run(run)
    root = Path(manifest['repo_root'])
    verify_sources(root, c)
    if output.exists():
        raise ValueError('Use a new application output directory')
    job = next(j for j in jobs if j['context_id'] == 'deployment')
    complete = completed_job(run, job)
    if complete is None:
        raise ValueError('AMIGA deployment fitting is incomplete')
    _, directory = complete
    info = json.loads((directory / 'ranking-seed-1201.model.json').read_text())
    source = case_dir / 'amiga/data_real.csv'
    evidence_path = case_dir / 'validation/amiga_exp_reported/reported_external_tf_target_evidence.csv'
    inputs = {p: sha256(p) for p in [source, evidence_path, *sorted(infer_grn_dir(case_dir).glob('GRN_*.csv'))]}
    frame = pd.read_csv(source).drop(columns=['AUPR'], errors='ignore').sort_values(['front_id', 'item_id']).reset_index(drop=True)
    if frame.empty or frame.front_id.nunique() != 1 or frame.duplicated(['front_id', 'item_id']).any():
        raise ValueError('Expected one nonempty real front with unique candidate identifiers')
    features = info['feature_columns']
    X = frame[features].to_numpy(dtype=float)
    model = load_native(directory, info)
    values, timing = prediction_costs(model, info['procedure']['family'], X,
                                      frame.item_id.to_numpy(), manifest['threads'])
    output.mkdir(parents=True)
    timing.to_csv(output / 'prediction_costs.csv', index=False)
    ranked = frame.assign(score=values).sort_values(['score', 'item_id'], ascending=[False, True], kind='mergesort').reset_index(drop=True)
    ranked['rank_in_front'] = np.arange(1, len(ranked)+1)
    validate_ranked_frame(ranked)
    ranked.to_csv(output / 'ranked_real.csv', index=False)
    # Exactly the original real-case analysis: AMIGA and four objective selectors.
    selectors = select_reported_candidates(ranked)
    write_selected_candidates(ranked=ranked, selectors=selectors, output_path=output / 'selected_candidates.csv')
    networks_dir = output / 'selected_networks'
    networks_dir.mkdir()
    network_paths = reconstruct_selected_networks(ranked=ranked, selectors=selectors,
                        grn_dir=infer_grn_dir(case_dir), networks_dir=networks_dir)
    evidence = pd.read_csv(evidence_path)
    if (set(evidence.resource) != {s['resource'] for s in REPORTED_SOURCES}
            or evidence[['source', 'target', 'resource']].isna().any().any()):
        raise ValueError('Incomplete frozen evidence resources')
    table = build_reported_source_support_table(selectors=selectors, network_paths=network_paths, evidence=evidence)
    table.to_csv(output / 'real_world_source_support_top1.csv', index=False)
    (output / 'real_world_source_support_top1.md').write_text(build_reported_markdown(table))
    evidence.to_csv(output / 'source_evidence_snapshot.csv', index=False)
    if any(sha256(p) != digest for p, digest in inputs.items()):
        raise ValueError('Application inputs changed during scoring')
    result = dict(status='complete', quality_labels_used=False, model_selection=c['deployment_policy'],
        seed=c['deployment_seed'], model_sha256=info['model_sha256'], selectors=[s.selector_id for s in selectors],
        prediction_cost_scope=c['prediction_cost_scope'], timing_repeats=5, warmup_repeats=1,
        evidence_scope='Original fixed resource snapshot; incomplete contextual support, not biological accuracy',
        resource_cutoffs=list(REPORTED_SOURCES), inputs_sha256={str(p.relative_to(root)): d for p, d in inputs.items()},
        artifacts={p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob('*')) if p.is_file()})
    write_json(output / 'manifest.json', result)
    return result
