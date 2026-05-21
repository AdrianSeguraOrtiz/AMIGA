from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.experiments.amiga_exp.context import CaseContextError, context_summary, load_case_context


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_case(
    tmp_path: Path,
    *,
    case_name: str = "CASE",
    folder_name: str = "CASE",
    assignments: list[dict] | None = None,
) -> tuple[Path, Path]:
    case_dir = tmp_path / folder_name
    data_dir = case_dir / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "data_1.csv").write_text(
        "front_id,item_id,AUPR,objective,feature\n"
        "1,1,0.8,0.1,1.0\n"
        "1,2,0.6,0.3,0.8\n"
        "2,1,0.4,0.2,0.5\n",
        encoding="utf-8",
    )
    split_manifest = _write_json(
        tmp_path / f"{case_name}_split_manifest.json",
        {
            "case": case_name,
            "assignments": assignments
            if assignments is not None
            else [
                {"front_id": 1, "front_name": "front_1", "split": "development"},
                {"front_id": 2, "front_name": "front_2", "split": "test"},
            ],
        },
    )
    feature_contract = _write_json(
        tmp_path / f"{case_name}_feature_columns.json",
        {
            "case": case_name,
            "target_column": "AUPR",
            "control_columns": ["front_id", "item_id"],
            "objective_columns": ["objective"],
            "objective_directions": {"objective": "minimize"},
            "feature_sets": {
                "full": ["objective", "feature"],
                "objectives_only": ["objective"],
            },
        },
    )
    config = _write_json(
        tmp_path / f"{case_name}.json",
        {
            "case": case_name,
            "data_csv": str(data_dir / "data_1.csv"),
            "split_manifest": str(split_manifest),
            "feature_contract": str(feature_contract),
            "group_column": "front_id",
            "item_column": "item_id",
            "target_column": "AUPR",
        },
    )
    return case_dir, config


def test_load_case_context_exposes_split_rows_and_contract(tmp_path):
    case_dir, config = _write_case(tmp_path)

    context = load_case_context(case_dir, config_path=config)
    summary = context_summary(context)

    assert context.case_name == "CASE"
    assert context.n_rows == 3
    assert context.n_fronts == 2
    assert context.n_development_rows == 2
    assert context.n_test_rows == 1
    assert context.n_development_fronts == 1
    assert context.n_test_fronts == 1
    assert context.target_column == "AUPR"
    assert context.control_columns == ["front_id", "item_id"]
    assert context.objective_columns == ["objective"]
    assert context.objective_directions == {"objective": "minimize"}
    assert context.feature_sets["full"] == ["objective", "feature"]
    assert context.results_root == case_dir / "results" / "amiga-exp"
    assert summary["development_rows"] == 2
    assert summary["test_rows"] == 1


def test_load_case_context_allows_case_name_override_for_folder_alias(tmp_path):
    case_dir, config = _write_case(tmp_path, case_name="CASE", folder_name="folder-alias")

    context = load_case_context(case_dir, case_name="CASE", config_path=config)

    assert context.case_name == "CASE"
    assert context.case_dir.name == "folder-alias"


def test_load_case_context_requires_all_fronts_to_have_split(tmp_path):
    case_dir, config = _write_case(
        tmp_path,
        assignments=[
            {"front_id": 1, "front_name": "front_1", "split": "development"},
        ],
    )

    with pytest.raises(CaseContextError, match="no split assignment"):
        load_case_context(case_dir, config_path=config)


def test_load_case_context_rejects_front_assigned_to_multiple_splits(tmp_path):
    case_dir, config = _write_case(
        tmp_path,
        assignments=[
            {"front_id": 1, "front_name": "front_1", "split": "development"},
            {"front_id": 1, "front_name": "front_1_duplicate", "split": "test"},
            {"front_id": 2, "front_name": "front_2", "split": "test"},
        ],
    )

    with pytest.raises(CaseContextError, match="multiple splits"):
        load_case_context(case_dir, config_path=config)
