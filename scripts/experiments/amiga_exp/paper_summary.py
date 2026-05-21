"""Paper-level summaries for AMIGA experimental results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from amiga.analysis.stat_tests import (
    build_front_metric_matrix_from_frames,
    friedman_with_holm,
    is_lower_better,
    rank_matrix,
)
from scripts.experiments.amiga_exp.context import CaseContext, REPO_ROOT
from scripts.experiments.amiga_exp.manifests import WriteOptions, repo_relative
from scripts.experiments.amiga_exp.phases.phase_04_decision_baselines import AMIGA_REFERENCE_ID


DEFAULT_PRIMARY_METRIC = "Regret@5"
SUMMARY_FILENAMES = {
    "final_test": "final_test_comparison.csv",
    "ablation": "ablation_comparison.csv",
    "baseline": "baseline_comparison.csv",
    "statistical_tests_csv": "statistical_tests.csv",
    "statistical_tests_json": "statistical_tests.json",
}
EXCLUDED_METRIC_COLUMNS = {"fold", "front_id", "n_items"}


class PaperSummaryError(ValueError):
    """Raised when paper summaries cannot be built safely."""


@dataclass(frozen=True)
class PaperSummaryResult:
    """Artifacts produced by paper-level summarization."""

    summary_dir: Path
    final_test_comparison: Path
    ablation_comparison: Path
    baseline_comparison: Path
    statistical_tests_csv: Path
    statistical_tests_json: Path
    status: str

    @property
    def outputs(self) -> dict[str, Path]:
        return {
            "final_test_comparison": self.final_test_comparison,
            "ablation_comparison": self.ablation_comparison,
            "baseline_comparison": self.baseline_comparison,
            "statistical_tests_csv": self.statistical_tests_csv,
            "statistical_tests_json": self.statistical_tests_json,
        }


def summarize_paper(
    context: CaseContext,
    *,
    options: WriteOptions | None = None,
    primary_metric: str = DEFAULT_PRIMARY_METRIC,
) -> PaperSummaryResult:
    """Build paper-level comparison tables and paired statistical tests."""
    options = options or WriteOptions()
    options.validate()
    summary_dir = context.results_root / "summaries"
    outputs = _summary_output_paths(summary_dir)

    final_metrics = _load_final_test_metrics(context)
    ablation_metrics = _load_ablation_metrics(context)
    baseline_metrics = _load_baseline_metrics(context)

    _require_metric("final_test", final_metrics, primary_metric)
    _require_metric("ablation", ablation_metrics, primary_metric)
    _require_metric("baseline", baseline_metrics, primary_metric)

    final_test_comparison = build_comparison_table(
        {AMIGA_REFERENCE_ID: final_metrics},
        comparison_type="final_test",
        primary_metric=primary_metric,
    )
    ablation_comparison = build_comparison_table(
        {AMIGA_REFERENCE_ID: final_metrics, **ablation_metrics},
        comparison_type="ablation",
        primary_metric=primary_metric,
    )
    baseline_comparison = build_comparison_table(
        baseline_metrics,
        comparison_type="baseline",
        primary_metric=primary_metric,
    )
    statistical_tests = build_statistical_tests(
        {
            "ablation": {AMIGA_REFERENCE_ID: final_metrics, **ablation_metrics},
            "baseline": baseline_metrics,
        },
        primary_metric=primary_metric,
    )

    if options.dry_run:
        return _result(summary_dir, outputs, status="dry_run")

    summary_dir.mkdir(parents=True, exist_ok=True)
    statuses = [
        _write_csv(outputs["final_test"], final_test_comparison, options),
        _write_csv(outputs["ablation"], ablation_comparison, options),
        _write_csv(outputs["baseline"], baseline_comparison, options),
        _write_csv(outputs["statistical_tests_csv"], statistical_tests, options),
        _write_json(
            outputs["statistical_tests_json"],
            _statistical_tests_payload(
                context,
                primary_metric=primary_metric,
                tests=statistical_tests,
                comparisons={
                    "final_test": final_test_comparison,
                    "ablation": ablation_comparison,
                    "baseline": baseline_comparison,
                },
            ),
            options,
        ),
    ]
    status = "skipped" if all(item == "skipped" for item in statuses) else "written"
    return _result(summary_dir, outputs, status=status)


def build_comparison_table(
    metrics_by_method: Mapping[str, pd.DataFrame],
    *,
    comparison_type: str,
    primary_metric: str = DEFAULT_PRIMARY_METRIC,
) -> pd.DataFrame:
    """Aggregate per-front method metrics over common fronts only."""
    common_fronts = _common_fronts(metrics_by_method)
    metric_names = _metric_names(metrics_by_method, common_fronts)
    if primary_metric not in metric_names:
        raise PaperSummaryError(
            f"comparison '{comparison_type}' is missing primary metric '{primary_metric}'"
        )

    rows: list[dict[str, Any]] = []
    for metric in metric_names:
        matrix = _metric_matrix(metrics_by_method, metric, common_fronts)
        if matrix.empty:
            continue
        lower_is_better = is_lower_better(metric)
        ranks = rank_matrix(matrix, lower_is_better=lower_is_better).mean(axis=0)
        for method in matrix.columns:
            values = pd.to_numeric(matrix[method], errors="coerce").dropna()
            rows.append(
                {
                    "comparison_type": comparison_type,
                    "method": method,
                    "metric": metric,
                    "is_primary_metric": metric == primary_metric,
                    "lower_is_better": lower_is_better,
                    "mean": float(values.mean()) if not values.empty else np.nan,
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "n_fronts": int(len(values)),
                    "n_fronts_common": int(len(common_fronts)),
                    "rank": float(ranks[method]),
                }
            )

    if not rows:
        raise PaperSummaryError(f"comparison '{comparison_type}' produced no metric rows")
    return pd.DataFrame(rows).sort_values(
        ["is_primary_metric", "metric", "rank", "method"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)


def build_statistical_tests(
    comparison_metrics: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    primary_metric: str = DEFAULT_PRIMARY_METRIC,
) -> pd.DataFrame:
    """Build paired Friedman/Holm tests over common fronts for each comparison."""
    rows: list[pd.DataFrame] = []
    for comparison_type, metrics_by_method in comparison_metrics.items():
        if len(metrics_by_method) < 2:
            continue
        matrix = build_front_metric_matrix_from_frames(metrics_by_method, primary_metric)
        matrix = matrix.dropna(axis=0, how="any")
        if matrix.empty:
            raise PaperSummaryError(
                f"comparison '{comparison_type}' has no common fronts for '{primary_metric}'"
            )
        ranked = rank_matrix(matrix, lower_is_better=is_lower_better(primary_metric))
        stats_df, friedman = friedman_with_holm(ranked)
        if stats_df.empty:
            continue
        stats_df = stats_df.rename(columns={"model": "method"})
        stats_df.insert(0, "comparison_type", comparison_type)
        stats_df.insert(1, "metric", primary_metric)
        stats_df["n_methods"] = int(matrix.shape[1])
        stats_df["n_fronts_common"] = int(matrix.shape[0])
        stats_df["friedman_stat"] = friedman["friedman_stat"]
        stats_df["friedman_p"] = friedman["friedman_p"]
        rows.append(stats_df)

    columns = [
        "comparison_type",
        "metric",
        "method",
        "avg_rank",
        "holm_p_adj",
        "is_winner",
        "significantly_worse",
        "friedman_stat",
        "friedman_p",
        "n_fronts",
        "n_methods",
        "n_fronts_common",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.concat(rows, ignore_index=True).loc[:, columns]


def load_report_groups(
    report_path: Path,
    *,
    method: str,
    front_col: str = "front_id",
) -> pd.DataFrame:
    """Load cv_report groups as one per-front metric frame for a method."""
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PaperSummaryError(f"missing cv_report.json: {report_path}") from exc
    except json.JSONDecodeError as exc:
        raise PaperSummaryError(f"invalid cv_report.json: {report_path}") from exc
    if not isinstance(payload, list):
        raise PaperSummaryError(f"{report_path}: expected a list of report entries")

    rows: list[dict[str, Any]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            raise PaperSummaryError(f"{report_path}: report entries must be objects")
        groups = entry.get("groups")
        if not isinstance(groups, list):
            raise PaperSummaryError(f"{report_path}: report entry is missing per-front groups")
        fold = int(entry.get("fold", 1))
        for group in groups:
            if not isinstance(group, dict):
                raise PaperSummaryError(f"{report_path}: group entries must be objects")
            row = dict(group)
            if front_col not in row:
                raise PaperSummaryError(f"{report_path}: group entry is missing '{front_col}'")
            row["fold"] = fold
            rows.append(row)

    if not rows:
        raise PaperSummaryError(f"{report_path}: no per-front groups found")
    df = pd.DataFrame(rows)
    df[front_col] = pd.to_numeric(df[front_col], errors="raise").astype(int)
    numeric_columns = [
        column
        for column in df.columns
        if column != front_col and _has_numeric_values(df[column])
    ]
    metrics_df = df[[front_col, *numeric_columns]].copy()
    for column in numeric_columns:
        metrics_df[column] = pd.to_numeric(metrics_df[column], errors="coerce")
    metrics_df = metrics_df.groupby(front_col, as_index=False).mean(numeric_only=True)
    metrics_df["method"] = method
    return metrics_df


def _load_final_test_metrics(context: CaseContext) -> pd.DataFrame:
    path = context.results_root / "02_hyperparameter_tuning" / "final_test" / "cv_report.json"
    return load_report_groups(path, method=AMIGA_REFERENCE_ID)


def _load_ablation_metrics(context: CaseContext) -> dict[str, pd.DataFrame]:
    root = context.results_root / "03_ablation" / "final_test"
    paths = sorted(root.glob("*/cv_report.json"))
    if not paths:
        raise PaperSummaryError(f"no ablation final-test reports found under: {root}")
    return {
        path.parent.name: load_report_groups(path, method=path.parent.name)
        for path in paths
    }


def _load_baseline_metrics(context: CaseContext) -> dict[str, pd.DataFrame]:
    manifest_path = context.results_root / "04_decision_baselines" / "baseline_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PaperSummaryError(f"decision-baseline manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise PaperSummaryError(f"invalid decision-baseline manifest JSON: {manifest_path}") from exc

    baselines = manifest.get("baselines") if isinstance(manifest, dict) else None
    if not isinstance(baselines, list) or not baselines:
        raise PaperSummaryError(f"decision-baseline manifest has no non-empty 'baselines' list: {manifest_path}")

    metrics = {}
    for baseline in baselines:
        if not isinstance(baseline, dict):
            raise PaperSummaryError(f"invalid baseline entry in manifest: {manifest_path}")
        baseline_id = str(baseline.get("baseline_id") or "")
        outputs = baseline.get("outputs")
        cv_report = outputs.get("cv_report") if isinstance(outputs, dict) else None
        if not baseline_id or not cv_report:
            raise PaperSummaryError(f"baseline entry is missing baseline_id or outputs.cv_report: {manifest_path}")
        path = _resolve_manifest_path(cv_report)
        if not path.exists():
            raise PaperSummaryError(f"decision-baseline report not found for '{baseline_id}': {path}")
        metrics[baseline_id] = load_report_groups(path, method=baseline_id)

    if AMIGA_REFERENCE_ID not in metrics:
        raise PaperSummaryError(f"decision-baseline reports must include '{AMIGA_REFERENCE_ID}'")
    return metrics


def _resolve_manifest_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _common_fronts(metrics_by_method: Mapping[str, pd.DataFrame]) -> list[int]:
    if not metrics_by_method:
        raise PaperSummaryError("at least one method is required")
    front_sets = []
    for method, metrics_df in metrics_by_method.items():
        if "front_id" not in metrics_df.columns:
            raise PaperSummaryError(f"method '{method}' has no front_id column")
        fronts = set(pd.to_numeric(metrics_df["front_id"], errors="raise").astype(int))
        if not fronts:
            raise PaperSummaryError(f"method '{method}' has no fronts")
        front_sets.append(fronts)
    common = sorted(set.intersection(*front_sets))
    if not common:
        raise PaperSummaryError("methods have no common fronts")
    return common


def _metric_names(metrics_by_method: Mapping[str, pd.DataFrame], common_fronts: Sequence[int]) -> list[str]:
    metric_sets = []
    for metrics_df in metrics_by_method.values():
        filtered = metrics_df[metrics_df["front_id"].isin(common_fronts)]
        metric_sets.append(
            {
                column
                for column in filtered.columns
                if column not in EXCLUDED_METRIC_COLUMNS
                and column != "method"
                and _has_numeric_values(filtered[column])
            }
        )
    return sorted(set.intersection(*metric_sets))


def _metric_matrix(
    metrics_by_method: Mapping[str, pd.DataFrame],
    metric: str,
    common_fronts: Sequence[int],
) -> pd.DataFrame:
    series = []
    for method, metrics_df in metrics_by_method.items():
        filtered = metrics_df[metrics_df["front_id"].isin(common_fronts)]
        metric_series = (
            filtered[["front_id", metric]]
            .rename(columns={metric: method})
            .set_index("front_id")
        )
        series.append(metric_series)
    return pd.concat(series, axis=1, join="inner").sort_index()


def _require_metric(
    comparison_type: str,
    metrics_by_method: Mapping[str, pd.DataFrame] | pd.DataFrame,
    metric: str,
) -> None:
    if isinstance(metrics_by_method, pd.DataFrame):
        frames = {comparison_type: metrics_by_method}
    else:
        frames = metrics_by_method
    missing = [
        method
        for method, metrics_df in frames.items()
        if metric not in metrics_df.columns
    ]
    if missing:
        raise PaperSummaryError(
            f"primary metric '{metric}' is missing for {comparison_type} method(s): {missing}"
        )


def _has_numeric_values(values: pd.Series) -> bool:
    numeric = pd.to_numeric(values, errors="coerce")
    return bool(numeric.notna().any())


def _summary_output_paths(summary_dir: Path) -> dict[str, Path]:
    return {name: summary_dir / filename for name, filename in SUMMARY_FILENAMES.items()}


def _write_csv(path: Path, df: pd.DataFrame, options: WriteOptions) -> str:
    if options.dry_run:
        return "dry_run"
    if path.exists():
        if options.skip_existing:
            return "skipped"
        if not options.force:
            raise PaperSummaryError(f"refusing to overwrite existing summary without --force: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return "written"


def _write_json(path: Path, payload: dict[str, Any], options: WriteOptions) -> str:
    if options.dry_run:
        return "dry_run"
    if path.exists():
        if options.skip_existing:
            return "skipped"
        if not options.force:
            raise PaperSummaryError(f"refusing to overwrite existing summary without --force: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return "written"


def _statistical_tests_payload(
    context: CaseContext,
    *,
    primary_metric: str,
    tests: pd.DataFrame,
    comparisons: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    return {
        "manifest_type": "paper_summary_statistical_tests",
        "case": context.case_name,
        "results_root": repo_relative(context.results_root),
        "primary_metric": primary_metric,
        "paired_by": "front_id",
        "comparisons": {
            name: {
                "methods": sorted(table["method"].unique().tolist()),
                "n_methods": int(table["method"].nunique()),
                "n_fronts_common": int(table["n_fronts_common"].max()) if not table.empty else 0,
            }
            for name, table in comparisons.items()
        },
        "tests": tests.to_dict(orient="records"),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return None if np.isnan(value) else value
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    return value


def _result(summary_dir: Path, outputs: Mapping[str, Path], *, status: str) -> PaperSummaryResult:
    return PaperSummaryResult(
        summary_dir=summary_dir,
        final_test_comparison=outputs["final_test"],
        ablation_comparison=outputs["ablation"],
        baseline_comparison=outputs["baseline"],
        statistical_tests_csv=outputs["statistical_tests_csv"],
        statistical_tests_json=outputs["statistical_tests_json"],
        status=status,
    )
