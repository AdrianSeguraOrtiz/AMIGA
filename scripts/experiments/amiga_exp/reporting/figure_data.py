"""Audited plotting data from the completed topology-grouped experiments."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, norm
from statsmodels.stats.multitest import multipletests

from .supervised import METHODS, sha256, verify_manifest

FAMILIES = ("LightGBM", "XGBoost", "CatBoost")
LABELS = ("rank_dense", "rank_avg", "quantiles_q5", "quantiles_q10",
          "quantiles_q15", "continuous", "reversed", "shuffled")
DISPLAY_METRICS = ("Regret@5", "Hit@1", "Hit@5")
FRACTIONS = (.25, .5, .75, 1.)


def audit_selection(summary: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Read immutable selection summaries; no predictions or models are recomputed."""
    summary = Path(summary).resolve()
    manifest, identities = verify_manifest(summary / "summary_manifest.json")
    for name in ("selection_candidates.csv", "feature_stability.csv"):
        if summary / name not in identities:
            raise ValueError(f"Selection manifest does not cover {name}")
    candidates = pd.read_csv(summary / "selection_candidates.csv")
    required = {"case", "outer_fold", "stage", "family", "arm", "label", "config_id",
                "parameters", "fraction", "n_features", "candidate", "job", "mean_regret5",
                "std_regret5", "selected_within_family"}
    if not required.issubset(candidates) or candidates.empty or candidates[list(required)].isna().any().any():
        raise ValueError("Incomplete selection candidate data")
    if candidates.duplicated(["case", "candidate"]).any():
        raise ValueError("Duplicate selection candidates")
    if not set(candidates.selected_within_family).issubset({True, False}):
        raise ValueError("Invalid selection markers")
    values = candidates[["mean_regret5", "std_regret5"]].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Selection metrics must be finite and nonnegative")
    return candidates, pd.read_csv(summary / "feature_stability.csv"), manifest, identities


def validate_selection(table: pd.DataFrame) -> None:
    """Require every fold, family, original grid and recursive feature budget."""
    if table.empty or set(table.arm) != set(METHODS):
        raise ValueError("Expected all five selected formulations")
    if not set(table.stage).issubset({"phase1", "phase2", "phase3"}):
        raise ValueError("Unexpected selection stage")
    if table.duplicated(["case", "candidate"]).any():
        raise ValueError("Duplicate selection candidates")
    for case, data in table.groupby("case"):
        if not isinstance(case, str) or not case.replace("-", "").isalnum():
            raise ValueError("Case must be a portable identifier")
        for arm in METHODS:
            part = data[data.arm == arm]
            stages = ("phase1", "phase2", "phase3") if arm == "ranking" else ("phase2", "phase3")
            if set(part.stage) != set(stages):
                raise ValueError("Missing or unexpected formulation stages")
            for stage in stages:
                for fold in range(5):
                    fold_data = part[(part.stage == stage) & (part.outer_fold == fold)]
                    if set(fold_data.family) != set(FAMILIES):
                        raise ValueError("Missing outer fold or model family")
                    for family in FAMILIES:
                        rows = fold_data[fold_data.family == family]
                        expected = 8 if stage == "phase1" else (27 if family == "LightGBM" else 36) if stage == "phase2" else 4
                        if len(rows) != expected or int(rows.selected_within_family.sum()) != 1:
                            raise ValueError("Incomplete grid, labels, fractions or selection markers")
                        if stage == "phase1" and set(rows.label) != set(LABELS):
                            raise ValueError("Relevance-label coverage differs")
                        if stage == "phase2" and rows.config_id.nunique() != expected:
                            raise ValueError("Duplicate parameter configurations")
                        if stage == "phase3" and set(rows.fraction) != set(FRACTIONS):
                            raise ValueError("Recursive feature coverage differs")
            if set(part.outer_fold) != set(range(5)):
                raise ValueError("Unexpected outer fold")


def screening_data(table: pd.DataFrame) -> pd.DataFrame:
    rows = table[(table.stage == "phase1") & (table.arm == "ranking")]
    return rows.groupby(["case", "family", "label"], as_index=False).agg(
        mean_regret5=("mean_regret5", "mean"), selected_folds=("selected_within_family", "sum"),
        n_outer_folds=("outer_fold", "nunique"))


def tuning_data(table: pd.DataFrame) -> pd.DataFrame:
    rows = table[table.stage == "phase2"].copy()
    rows["rounded_regret5"] = rows.mean_regret5.round(12)
    rows["diagnostic_rank"] = rows.groupby(["case", "arm", "outer_fold"])["rounded_regret5"].rank(method="average")
    # Configuration IDs denote the same parameter settings across complements;
    # ranking relevance labels can differ and are never treated as a single label.
    return rows.groupby(["case", "arm", "family", "config_id", "parameters"], as_index=False).agg(
        mean_regret5=("mean_regret5", "mean"), std_regret5=("std_regret5", "mean"),
        diagnostic_rank=("diagnostic_rank", "mean"), selected_folds=("selected_within_family", "sum"),
        n_outer_folds=("outer_fold", "nunique"), label_modes=("label", lambda s: ", ".join(sorted(set(s)))))


def feature_curve_data(table: pd.DataFrame) -> pd.DataFrame:
    rows = table[table.stage == "phase3"]
    return rows.groupby(["case", "arm", "family", "fraction", "n_features"], as_index=False).agg(
        mean_regret5=("mean_regret5", "mean"), selected_folds=("selected_within_family", "sum"),
        n_outer_folds=("outer_fold", "nunique"))


def normalize_importance(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize centered training SHAP within each full-feature inner fit."""
    required = {"feature", "inner_fold", "fraction", "centered_mean_abs_shap"}
    if not required.issubset(frame) or frame[list(required)].isna().any().any():
        raise ValueError("Incomplete feature importance")
    data = frame.loc[frame.fraction == 1.].copy()
    if data.empty or data.duplicated(["inner_fold", "feature"]).any():
        raise ValueError("Missing or duplicate full-feature importance")
    values = data.centered_mean_abs_shap.to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid centered SHAP contributions")
    total = data.groupby("inner_fold").centered_mean_abs_shap.transform("sum")
    data["relative_centered_shap"] = np.divide(values, total, out=np.zeros_like(values), where=total.to_numpy() > 0)
    return data


def load_importance(table: pd.DataFrame, selections: dict) -> tuple[pd.DataFrame, dict]:
    """Bind each SHAP CSV to the checksummed result identity in its summary."""
    rows, identities = [], {}
    for job in table.loc[table.stage == "phase3"].drop_duplicates("job").itertuples():
        run, manifest = selections[job.arm]
        result_root = Path(run) / "jobs" / job.job
        expected = manifest["dependency_result_hashes"][job.job]
        matches = [p for p in result_root.glob("attempt-*/result.json") if sha256(p) == expected]
        if len(matches) != 1:
            raise ValueError(f"Missing or ambiguous feature result: {job.job}")
        result_path = matches[0]
        result = json.loads(result_path.read_text())
        if result.get("status") != "complete" or result.get("job", {}).get("id") != job.job:
            raise ValueError("Feature result identity differs from selection")
        importance_path = result_path.parent / "feature_importance.csv"
        if sha256(importance_path) != result["artifacts"]["feature_importance.csv"]:
            raise ValueError("Feature importance SHA-256 mismatch")
        identities[result_path] = expected
        identities[importance_path] = result["artifacts"]["feature_importance.csv"]
        data = normalize_importance(pd.read_csv(importance_path))
        if set(data.inner_fold) != {0, 1, 2}:
            raise ValueError("Incomplete SHAP inner-fold coverage")
        full = table[(table.job == job.job) & (table.fraction == 1.)].n_features.item()
        for inner, part in data.groupby("inner_fold"):
            if len(part) != full:
                raise ValueError("Full-feature SHAP predictor coverage differs")
        rows.append(data.assign(case=job.case, arm=job.arm, family=job.family, outer_fold=job.outer_fold))
    all_rows = pd.concat(rows, ignore_index=True)
    summary = all_rows.groupby(["case", "arm", "family", "feature"], as_index=False).agg(
        relative_centered_shap=("relative_centered_shap", "mean"), n_inner_fits=("inner_fold", "size"))
    if not (summary.n_inner_fits == 15).all():
        raise ValueError("Feature importance must include all 15 overlapping training fits")
    return summary, identities


def validate_feature_summary(importance: pd.DataFrame, stability: pd.DataFrame) -> None:
    """Every displayed importance must have a matching 15-fit inclusion rate."""
    keys = ["case", "arm", "family", "feature"]
    required = {*keys, "selected_frequency", "number_of_fits"}
    if (not required.issubset(stability) or stability[list(required)].isna().any().any()
            or stability.duplicated(keys).any()):
        raise ValueError("Incomplete or duplicate column inclusion summary")
    values = stability.selected_frequency.to_numpy(float)
    if (not np.isfinite(values).all() or (values < 0).any() or (values > 1).any()
            or not (stability.number_of_fits == 15).all()):
        raise ValueError("Column inclusion must cover 15 fits with rates in [0, 1]")
    if set(map(tuple, importance[keys].to_numpy())) != set(map(tuple, stability[keys].to_numpy())):
        raise ValueError("SHAP and column inclusion predictor coverage differ")


def rank_comparison(frame: pd.DataFrame, methods: tuple[str, ...], *, expected_topologies: int = 87) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fixed AMIGA control, paired topology ranks, primary-only omnibus/Holm."""
    if len(set(methods)) != len(methods) or len(methods) < 3 or methods[0] != "ranking":
        raise ValueError("Use at least three unique methods and the fixed ranking control first")
    required = {"case", "topology_id", "method", *DISPLAY_METRICS}
    if not required.issubset(frame) or frame.empty or expected_topologies < 2:
        raise ValueError("Missing paired topology metrics or invalid topology count")
    if frame[list(required)].isna().any().any():
        raise ValueError("Missing metric values or topology identifiers")
    output, tests = [], []
    for case, data in frame.groupby("case", sort=True):
        selected = data[data.method.isin(methods)]
        if selected.duplicated(["topology_id", "method"]).any():
            raise ValueError("Duplicate paired topology observations")
        n, k = selected.topology_id.nunique(), len(methods)
        if n != expected_topologies or len(selected) != n*k or set(selected.method) != set(methods):
            raise ValueError("Incomplete paired comparison coverage")
        values = selected[list(DISPLAY_METRICS)].to_numpy(float)
        if not np.isfinite(values).all() or (values < -1e-12).any() or (values > 1+1e-12).any():
            raise ValueError("Metrics must be finite and within [0, 1]")
        matrices, ranks, means = {}, {}, {}
        for metric in DISPLAY_METRICS:
            raw = selected.pivot(index="topology_id", columns="method", values=metric).reindex(columns=methods)
            if not np.isfinite(raw.to_numpy(float)).all():
                raise ValueError("Incomplete or nonfinite paired metric matrix")
            matrices[metric] = raw.round(12)
            ranks[metric] = matrices[metric].rank(axis=1, method="average", ascending=metric.startswith("Regret")).mean()
            means[metric] = raw.mean()
        primary = matrices["Regret@5"]
        if (primary.nunique(axis=1) == 1).all():
            statistic, p = 0., 1.
        else:
            statistic, p = friedmanchisquare(*[primary[m].to_numpy() for m in methods])
        se = np.sqrt(k*(k+1)/(6*n))
        z = [(ranks["Regret@5"][m] - ranks["Regret@5"]["ranking"])/se for m in methods[1:]]
        raw_p = 2*norm.sf(np.abs(z))
        adjusted = multipletests(raw_p, method="holm")[1]
        tests.append(dict(case=case, metric="Regret@5", statistic=statistic, p_value=p,
                          n_topologies=n, n_methods=k, control="ranking", n_holm_comparisons=k-1))
        for i, method in enumerate(methods):
            record = dict(case=case, method=method, n_topologies=n, n_methods=k,
                          control="ranking", standard_error=se,
                          mean_rank_delta=ranks["Regret@5"][method]-ranks["Regret@5"]["ranking"],
                          friedman_p=p, p_holm=np.nan if i == 0 else adjusted[i-1],
                          p_raw=np.nan if i == 0 else raw_p[i-1],
                          significant=bool(i and p < .05 and adjusted[i-1] < .05))
            for metric in DISPLAY_METRICS:
                record[metric+"_mean_rank"] = ranks[metric][method]
                record[metric+"_mean"] = means[metric][method]
            output.append(record)
    return pd.DataFrame(output), pd.DataFrame(tests)


def comparison_sets(frame: pd.DataFrame, layout: str) -> dict[str, tuple[str, ...]]:
    """Never include the oracle or the unreported normalized-regression arm."""
    if layout not in ("separate", "joint"):
        raise ValueError("Comparison layout must be separate or joint")
    methods = sorted(set(frame.method) - set(METHODS) - {"oracle", "reg_normalized"})
    if any(not m.startswith("objective_") and m != "random_uniform" for m in methods):
        raise ValueError("Unrecognized objective-only comparator")
    if layout == "joint":
        return {"all": (*METHODS, *methods)}
    return {"supervised": METHODS, "objectives": ("ranking", *methods)}
