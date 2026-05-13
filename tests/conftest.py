from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _write_report(path: Path, payload: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def sample_cv_reports(tmp_path: Path) -> dict[str, Path]:
    report_a = [
        {
            "fold": 1,
            "agg": {"Regret@1": 0.10, "Regret@3": 0.05, "Hit@1": 1.0, "Hit@3": 1.0, "NDCG@1": 0.92, "Spearman": 0.50, "KendallTau": 0.33, "n_items": 3},
            "groups": [
                {"front_id": 101, "Regret@1": 0.00, "Regret@3": 0.00, "Hit@1": 1.0, "Hit@3": 1.0, "NDCG@1": 1.00, "Spearman": 1.0, "KendallTau": 1.0, "n_items": 3},
                {"front_id": 102, "Regret@1": 0.20, "Regret@3": 0.10, "Hit@1": 0.0, "Hit@3": 1.0, "NDCG@1": 0.84, "Spearman": 0.0, "KendallTau": 0.0, "n_items": 3},
            ],
            "label_mode": "continuous",
            "label_quantiles": None,
            "meta": {"model": "ModelA", "label_mode": "continuous", "label_quantiles": None},
        },
        {
            "fold": 2,
            "agg": {"Regret@1": 0.12, "Regret@3": 0.07, "Hit@1": 1.0, "Hit@3": 1.0, "NDCG@1": 0.89, "Spearman": 0.42, "KendallTau": 0.25, "n_items": 3},
            "groups": [
                {"front_id": 201, "Regret@1": 0.05, "Regret@3": 0.00, "Hit@1": 1.0, "Hit@3": 1.0, "NDCG@1": 0.94, "Spearman": 0.5, "KendallTau": 0.33, "n_items": 3},
                {"front_id": 202, "Regret@1": 0.19, "Regret@3": 0.14, "Hit@1": 0.0, "Hit@3": 1.0, "NDCG@1": 0.78, "Spearman": -0.1, "KendallTau": 0.0, "n_items": 3},
            ],
            "label_mode": "continuous",
            "label_quantiles": None,
            "meta": {"model": "ModelA", "label_mode": "continuous", "label_quantiles": None},
        },
    ]
    report_b = [
        {
            "fold": 1,
            "agg": {"Regret@1": 0.25, "Regret@3": 0.12, "Hit@1": 0.0, "Hit@3": 1.0, "NDCG@1": 0.71, "Spearman": 0.10, "KendallTau": 0.0, "n_items": 3},
            "groups": [
                {"front_id": 101, "Regret@1": 0.30, "Regret@3": 0.12, "Hit@1": 0.0, "Hit@3": 1.0, "NDCG@1": 0.66, "Spearman": 0.0, "KendallTau": 0.0, "n_items": 3},
                {"front_id": 102, "Regret@1": 0.20, "Regret@3": 0.12, "Hit@1": 0.0, "Hit@3": 1.0, "NDCG@1": 0.76, "Spearman": 0.2, "KendallTau": 0.0, "n_items": 3},
            ],
            "label_mode": "continuous",
            "label_quantiles": None,
            "meta": {"model": "ModelB", "label_mode": "continuous", "label_quantiles": None},
        },
        {
            "fold": 2,
            "agg": {"Regret@1": 0.18, "Regret@3": 0.09, "Hit@1": 0.0, "Hit@3": 1.0, "NDCG@1": 0.74, "Spearman": 0.12, "KendallTau": 0.0, "n_items": 3},
            "groups": [
                {"front_id": 201, "Regret@1": 0.22, "Regret@3": 0.11, "Hit@1": 0.0, "Hit@3": 1.0, "NDCG@1": 0.69, "Spearman": 0.0, "KendallTau": 0.0, "n_items": 3},
                {"front_id": 202, "Regret@1": 0.14, "Regret@3": 0.07, "Hit@1": 0.0, "Hit@3": 1.0, "NDCG@1": 0.79, "Spearman": 0.25, "KendallTau": 0.0, "n_items": 3},
            ],
            "label_mode": "continuous",
            "label_quantiles": None,
            "meta": {"model": "ModelB", "label_mode": "continuous", "label_quantiles": None},
        },
    ]
    model_a = _write_report(tmp_path / "run_a" / "cv_report.json", report_a)
    model_b = _write_report(tmp_path / "run_b" / "cv_report.json", report_b)
    return {"ModelA": model_a, "ModelB": model_b}


@pytest.fixture
def sample_cv_report_without_groups(tmp_path: Path) -> Path:
    payload = [
        {
            "fold": 1,
            "agg": {"Regret@1": 0.1, "Hit@1": 1.0, "n_items": 3},
            "label_mode": "continuous",
            "label_quantiles": None,
            "meta": {"model": "BrokenModel", "label_mode": "continuous", "label_quantiles": None},
        }
    ]
    return _write_report(tmp_path / "broken" / "cv_report.json", payload)


@pytest.fixture
def tiny_expression_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "c1": [1.0, 2.0, 3.0],
            "c2": [2.0, 1.0, 4.0],
            "c3": [3.0, 2.0, 5.0],
        },
        index=["g1", "g2", "g3"],
    )
