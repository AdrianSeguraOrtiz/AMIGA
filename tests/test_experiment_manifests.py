from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.experiments.amiga_exp.context import load_case_context
from scripts.experiments.amiga_exp.manifests import (
    DEFAULT_PHASES,
    ManifestError,
    WriteOptions,
    initialize_results_layout,
    results_layout,
    run_manifest,
    write_manifest,
    write_phase_status_manifest,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_case(tmp_path: Path) -> tuple[Path, Path]:
    case_dir = tmp_path / "CASE"
    data_dir = case_dir / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "data_1.csv").write_text(
        "front_id,item_id,AUPR,objective,feature\n"
        "1,1,0.8,0.1,1.0\n"
        "2,1,0.4,0.2,0.5\n",
        encoding="utf-8",
    )
    split_manifest = _write_json(
        tmp_path / "CASE_split_manifest.json",
        {
            "case": "CASE",
            "assignments": [
                {"front_id": 1, "front_name": "front_1", "split": "development"},
                {"front_id": 2, "front_name": "front_2", "split": "test"},
            ],
        },
    )
    feature_contract = _write_json(
        tmp_path / "CASE_feature_columns.json",
        {
            "case": "CASE",
            "target_column": "AUPR",
            "control_columns": ["front_id", "item_id"],
            "objective_columns": ["objective"],
            "objective_directions": {"objective": "minimize"},
            "feature_sets": {"full": ["objective", "feature"]},
        },
    )
    config = _write_json(
        tmp_path / "CASE.json",
        {
            "case": "CASE",
            "data_csv": str(data_dir / "data_1.csv"),
            "split_manifest": str(split_manifest),
            "feature_contract": str(feature_contract),
            "group_column": "front_id",
            "item_column": "item_id",
            "target_column": "AUPR",
        },
    )
    return case_dir, config


def _context(tmp_path: Path):
    case_dir, config = _write_case(tmp_path)
    return load_case_context(case_dir, config_path=config)


def test_initialize_results_layout_creates_directories_and_basic_manifests(tmp_path):
    context = _context(tmp_path)

    result = initialize_results_layout(context, seed=123, options=WriteOptions())
    layout = result["layout"]

    for directory in layout.all_directories:
        assert directory.is_dir()
    environment = json.loads((layout.manifests / "environment_manifest.json").read_text())
    assert environment["manifest_type"] == "environment"
    assert environment["case"] == "CASE"
    assert environment["seed"] == 123
    for phase in DEFAULT_PHASES:
        phase_manifest = json.loads((layout.phases[phase] / "phase_manifest.json").read_text())
        assert phase_manifest["phase"] == phase
        assert phase_manifest["status"] == "pending"
    assert result["environment_status"] == "written"
    assert set(result["phase_statuses"]) == set(DEFAULT_PHASES)


def test_initialize_results_layout_dry_run_does_not_write(tmp_path):
    context = _context(tmp_path)
    layout = results_layout(context)

    result = initialize_results_layout(context, seed=123, options=WriteOptions(dry_run=True))

    assert result["environment_status"] == "dry_run"
    assert not layout.root.exists()


def test_write_phase_status_manifest_promotes_pending_without_force(tmp_path):
    context = _context(tmp_path)
    result = initialize_results_layout(context, seed=123, options=WriteOptions())
    layout = result["layout"]
    path = layout.phases["01_model_screening"] / "phase_manifest.json"

    status = write_phase_status_manifest(
        path,
        context,
        "01_model_screening",
        seed=123,
        status="completed",
        outputs={"screening_manifest": "screening_manifest.json"},
        options=WriteOptions(),
    )

    assert status == "written"
    phase_payload = json.loads(path.read_text(encoding="utf-8"))
    assert phase_payload["status"] == "completed"
    assert phase_payload["outputs"]["screening_manifest"] == "screening_manifest.json"


def test_write_manifest_refuses_overwrite_without_force(tmp_path):
    context = _context(tmp_path)
    layout = results_layout(context)
    path = layout.manifests / "run_manifest.json"
    payload = run_manifest(context, "01_model_screening", "run_a", seed=1)

    assert write_manifest(path, payload, WriteOptions()) == "written"
    with pytest.raises(ManifestError, match="refusing to overwrite"):
        write_manifest(path, payload, WriteOptions())


def test_write_manifest_skip_existing_and_force_controls(tmp_path):
    context = _context(tmp_path)
    layout = results_layout(context)
    path = layout.manifests / "run_manifest.json"
    payload = run_manifest(context, "01_model_screening", "run_a", seed=1)
    replacement = run_manifest(context, "01_model_screening", "run_b", seed=2)

    assert write_manifest(path, payload, WriteOptions()) == "written"
    assert write_manifest(path, replacement, WriteOptions(skip_existing=True)) == "skipped"
    assert json.loads(path.read_text())["run_id"] == "run_a"
    assert write_manifest(path, replacement, WriteOptions(force=True)) == "written"
    rewritten = json.loads(path.read_text())
    assert rewritten["run_id"] == "run_b"
    assert rewritten["seed"] == 2


def test_write_options_reject_force_and_skip_existing_together():
    with pytest.raises(ManifestError, match="cannot be used together"):
        WriteOptions(force=True, skip_existing=True).validate()
