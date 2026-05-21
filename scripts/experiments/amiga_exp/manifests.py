"""Result-layout and manifest helpers for AMIGA experimental phases."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from scripts.experiments.amiga_exp.context import CaseContext, REPO_ROOT


DEFAULT_PHASES = (
    "01_model_screening",
    "02_hyperparameter_tuning",
    "03_ablation",
    "04_decision_baselines",
)

ManifestWriteStatus = Literal["written", "skipped", "dry_run"]


class ManifestError(ValueError):
    """Raised when result layout or manifest writing cannot proceed safely."""


@dataclass(frozen=True)
class WriteOptions:
    """Common write controls for experimental artifacts."""

    force: bool = False
    skip_existing: bool = False
    dry_run: bool = False

    def validate(self) -> None:
        if self.force and self.skip_existing:
            raise ManifestError("--force and --skip-existing cannot be used together")


@dataclass(frozen=True)
class ResultsLayout:
    """Resolved output directories for one case."""

    root: Path
    manifests: Path
    summaries: Path
    phases: dict[str, Path]

    @property
    def all_directories(self) -> tuple[Path, ...]:
        return (
            self.root,
            self.manifests,
            *self.phases.values(),
            self.summaries,
        )


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def results_layout(context: CaseContext, phases: tuple[str, ...] = DEFAULT_PHASES) -> ResultsLayout:
    phase_dirs = {phase: context.results_root / phase for phase in phases}
    return ResultsLayout(
        root=context.results_root,
        manifests=context.results_root / "manifests",
        summaries=context.results_root / "summaries",
        phases=phase_dirs,
    )


def ensure_results_layout(layout: ResultsLayout, *, dry_run: bool = False) -> list[Path]:
    """Create all result directories unless dry-run is enabled."""
    directories = list(layout.all_directories)
    if not dry_run:
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    return directories


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_manifest(path: Path, payload: dict[str, Any], options: WriteOptions) -> ManifestWriteStatus:
    """Write one manifest with explicit overwrite controls."""
    options.validate()
    if options.dry_run:
        return "dry_run"
    if path.exists():
        if options.skip_existing:
            return "skipped"
        if not options.force:
            raise ManifestError(f"refusing to overwrite existing manifest without --force: {path}")
    _write_json(path, payload)
    return "written"


def write_phase_status_manifest(
    path: Path,
    context: CaseContext,
    phase: str,
    *,
    seed: int | None,
    status: str,
    outputs: dict[str, Any] | None,
    options: WriteOptions,
) -> ManifestWriteStatus:
    """Update the lifecycle manifest for a phase.

    `init-results` creates these manifests as `pending`. A phase run should be
    able to promote its own lifecycle manifest to `completed` without requiring
    callers to pass `--force` just because the pending file exists.
    """
    lifecycle_options = WriteOptions(force=True, dry_run=options.dry_run)
    return write_manifest(
        path,
        phase_manifest(
            context,
            phase,
            seed=seed,
            status=status,
            outputs=outputs,
        ),
        lifecycle_options,
    )


def common_manifest_fields(context: CaseContext, *, seed: int | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": utc_now_iso(),
        "git_commit": git_commit(),
        "case": context.case_name,
        "case_dir": repo_relative(context.case_dir),
        "data_csv": repo_relative(context.data_csv),
        "config_path": repo_relative(context.config_path),
        "split_manifest": repo_relative(context.split_manifest_path),
        "feature_contract": repo_relative(context.feature_contract_path),
        "results_root": repo_relative(context.results_root),
        "split": {
            "development_fronts": context.n_development_fronts,
            "development_rows": context.n_development_rows,
            "test_fronts": context.n_test_fronts,
            "test_rows": context.n_test_rows,
        },
    }
    if seed is not None:
        payload["seed"] = int(seed)
    return payload


def environment_manifest(context: CaseContext, *, seed: int | None = None) -> dict[str, Any]:
    return {
        **common_manifest_fields(context, seed=seed),
        "manifest_type": "environment",
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "command_environment": {
            "cwd": str(Path.cwd()),
        },
    }


def phase_manifest(
    context: CaseContext,
    phase: str,
    *,
    seed: int | None = None,
    status: str = "pending",
    outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        **common_manifest_fields(context, seed=seed),
        "manifest_type": "phase",
        "phase": phase,
        "status": status,
        "outputs": outputs or {},
    }


def run_manifest(
    context: CaseContext,
    phase: str,
    run_id: str,
    *,
    seed: int | None = None,
    status: str = "pending",
    parameters: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        **common_manifest_fields(context, seed=seed),
        "manifest_type": "run",
        "phase": phase,
        "run_id": run_id,
        "status": status,
        "parameters": parameters or {},
        "outputs": outputs or {},
    }


def initialize_results_layout(
    context: CaseContext,
    *,
    seed: int | None,
    options: WriteOptions,
    phases: tuple[str, ...] = DEFAULT_PHASES,
) -> dict[str, Any]:
    """Create the standard result layout and write basic manifests."""
    options.validate()
    layout = results_layout(context, phases=phases)
    directories = ensure_results_layout(layout, dry_run=options.dry_run)

    environment_path = layout.manifests / "environment_manifest.json"
    environment_status = write_manifest(
        environment_path,
        environment_manifest(context, seed=seed),
        options,
    )

    phase_statuses: dict[str, ManifestWriteStatus] = {}
    for phase, phase_dir in layout.phases.items():
        manifest_path = phase_dir / "phase_manifest.json"
        phase_statuses[phase] = write_manifest(
            manifest_path,
            phase_manifest(
                context,
                phase,
                seed=seed,
                status="pending",
                outputs={"phase_dir": repo_relative(phase_dir)},
            ),
            options,
        )

    return {
        "layout": layout,
        "directories": directories,
        "environment_manifest": environment_path,
        "environment_status": environment_status,
        "phase_statuses": phase_statuses,
    }
