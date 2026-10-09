"""Reproducible publication figures from completed scientific summaries."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import scipy
import statsmodels

from scripts.experiments.amiga_exp.version import __version__
from .figure_data import (audit_selection, comparison_sets, feature_curve_data,
                          load_importance, rank_comparison, screening_data,
                          tuning_data, validate_feature_summary, validate_selection)
from .supervised import METHODS, sha256, verify_manifest


def generate_figures(selection_summary: Path, selection_run: Path,
                     top5_summary: Path, top5_run: Path,
                     top10_summary: Path, top10_run: Path,
                     outer_summary: Path, output: Path, *, layout="separate", top_features=20) -> dict:
    """Preserve frozen sources and artifacts; write figures into a new directory."""
    from . import publication_plots as plots
    implementation = {p:sha256(p) for p in Path(__file__).parent.glob("*.py")}
    style_source = Path(plots.__file__).parents[1]/"plots.py"
    style_digest = sha256(style_source)
    output = Path(output).resolve()
    inputs = [Path(p).resolve() for p in (selection_summary, selection_run, top5_summary,
                                          top5_run, top10_summary, top10_run, outer_summary)]
    if output.exists() or any(output.is_relative_to(p) for p in inputs):
        raise ValueError("Use a new figure directory outside all source directories")
    if layout not in ("separate", "joint") or not 5 <= top_features <= 40:
        raise ValueError("Invalid comparison layout or displayed feature count")
    selection_summary, selection_run, top5_summary, top5_run, top10_summary, top10_run, outer_summary = inputs
    candidates, stability, identities, selections, audits = [], [], {}, {}, []
    groups = [(selection_summary, selection_run, ("ranking", "reg_aupr", "clf_top20")),
              (top5_summary, top5_run, ("clf_top05",)), (top10_summary, top10_run, ("clf_top10",))]
    for summary, run, arms in groups:
        table, inclusion, manifest, verified = audit_selection(summary)
        identities.update(verified)
        candidates.append(table[table.arm.isin(arms)])
        stability.append(inclusion[inclusion.arm.isin(arms)])
        selections.update({arm: (run, manifest) for arm in arms})
        audits.append(dict(manifest=str(summary/"summary_manifest.json"), verified_artifacts=len(manifest["artifacts"])))
    table, stability = pd.concat(candidates, ignore_index=True), pd.concat(stability, ignore_index=True)
    validate_selection(table)
    importance, verified = load_importance(table, selections)
    validate_feature_summary(importance, stability)
    identities.update(verified)
    outer, verified = verify_manifest(outer_summary/"manifest.json")
    if outer_summary/"topology_metrics.csv" not in verified:
        raise ValueError("Outer summary must cover topology_metrics.csv")
    identities.update(verified)
    audits.append(dict(manifest=str(outer_summary/"manifest.json"), verified_artifacts=len(outer["artifacts"])))
    metrics = pd.read_csv(outer_summary/"topology_metrics.csv", dtype={"topology_id":str})
    if set(metrics["case"]) != set(table["case"]):
        raise ValueError("Selection and final-comparison cases differ")
    screening, tuning, curves = screening_data(table), tuning_data(table), feature_curve_data(table)
    if not (tuning.n_outer_folds == 5).all() or not (curves.n_outer_folds == 5).all():
        raise ValueError("All five outer training complements must contribute to each diagnostic")
    comparisons, omnibus = [], []
    for case, data in metrics.groupby("case", sort=True):
        for scope, methods in comparison_sets(data, layout).items():
            ranked, tested = rank_comparison(data, methods)
            comparisons.append(ranked.assign(scope=scope))
            omnibus.append(tested.assign(scope=scope))
    comparisons, omnibus = pd.concat(comparisons, ignore_index=True), pd.concat(omnibus, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".amiga-figures-", dir=output.parent) as temporary:
        destination = Path(temporary)/"figures"
        destination.mkdir()
        tables = {"selection_candidates.csv":table, "screening.csv":screening, "tuning.csv":tuning,
                  "feature_curves.csv":curves, "relative_training_shap.csv":importance,
                  "feature_stability.csv":stability, "comparison_ranks.csv":comparisons,
                  "friedman_omnibus.csv":omnibus}
        for name, frame in tables.items():
            frame.to_csv(destination/name, index=False)
        for case in sorted(table.case.unique()):
            prefix = plots.PREFIXES.get(case, case.lower())
            plots.screening(screening[screening.case == case], destination/f"{prefix}_phase01_model_screening", case)
            for arm in METHODS:
                candidate = tuning[(tuning.case == case) & (tuning.arm == arm)]
                feature = curves[(curves.case == case) & (curves.arm == arm)]
                fold_rows = table[(table.case == case) & (table.arm == arm) & (table.stage == "phase3")]
                shap = importance[(importance.case == case) & (importance.arm == arm)]
                inclusion = stability[(stability.case == case) & (stability.arm == arm)]
                parent = destination if arm == "ranking" else destination/"supplementary"/case/arm
                stem = prefix if arm == "ranking" else arm
                plots.tuning(candidate, parent/f"{stem}_phase02_hyperparameter_tuning", case, arm)
                plots.feature_selection(feature, fold_rows, shap, inclusion,
                                        parent/f"{stem}_phase03_feature_selection", case, arm, top_features=top_features)
                plots.full_feature_matrix(shap, inclusion, destination/"supplementary"/case/arm/"all_columns", case, arm)
            panels = [(scope, comparisons[(comparisons.case == case) & (comparisons.scope == scope)])
                      for scope in comparison_sets(metrics[metrics.case == case], layout)]
            plots.decision_comparison(panels, destination/f"{prefix}_phase04_decision_baselines", case)
        for path, digest in identities.items():
            if sha256(path) != digest:
                raise ValueError(f"Source artifact changed during plotting: {path}")
        for path, digest in {**implementation, style_source:style_digest}.items():
            if sha256(path) != digest:
                raise ValueError(f"Plotting implementation changed during rendering: {path}")
        artifacts = {p.relative_to(destination).as_posix():sha256(p) for p in sorted(destination.rglob("*")) if p.is_file()}
        manifest = dict(schema_version=1, status="complete", workflow="publication_figures", workflow_version=__version__,
                        created_at_utc=datetime.now(timezone.utc).isoformat(), no_model_fits=True,
                        comparison_layout=layout, comparison_orientation="horizontal",
                        supervised_methods=list(METHODS), display_metrics=["Regret@5"],
                        saved_comparison_metrics=["Regret@5","Hit@1","Hit@5"],
                        displayed_feature_count=top_features, feature_display="highest and lowest mean relative training SHAP",
                        feature_budget_display="within-family original selection counts out of five; no stars",
                        artifact_audit=audits,
                        inputs_sha256={str(p):digest for p,digest in identities.items()},
                        implementation_sha256={p.name:digest for p,digest in implementation.items()},
                        reused_style_sha256=style_digest,
                        selection_scope="equal-weight average over five overlapping outer training complements; descriptive inner diagnostics",
                        selection_markers="frequency of the original within-family winners; no pooled re-selection",
                        shap_scope="full-feature centered training SHAP normalized within each inner fit, averaged over 15 overlapping fits per family",
                        inference_scope="Regret@5 only; fixed AMIGA control, tie-corrected Friedman, Holm within each case/panel; exploratory",
                        numerical_ties="round to 12 decimal places before average tied ranks; raw means unrounded",
                        unit="equally weighted topology means of seed-averaged, then condition-averaged metrics",
                        posthoc="two-sided normal z=(comparator rank-AMIGA rank)/sqrt(k*(k+1)/(6*N))",
                        secondary_scope="Hit@1 and Hit@5 mean ranks and raw hit rates saved in comparison_ranks.csv for textual reporting; not plotted or tested",
                        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                                      scipy=scipy.__version__, statsmodels=statsmodels.__version__,
                                      matplotlib=plots.matplotlib.__version__, seaborn=plots.sns.__version__),
                        artifacts=artifacts)
        (destination/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        destination.rename(output)
    return manifest
