#!/usr/bin/env python3
"""Create frozen development/test split manifests for AMIGA experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from inspect_case import (
    DEFAULT_CASES,
    REPO_ROOT,
    build_front_id_map,
    build_fronts_summary,
    family_test_quotas,
    git_commit,
    resolve_training_csv,
    select_diverse_test_fronts,
)


DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "experiments" / "splits"
ASSIGNMENT_COLUMNS = ["front_id", "front_name", "family", "source", "size", "condition", "n_items"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create reproducible held-out development/test split manifests for "
            "one or more AMIGA case studies."
        )
    )
    parser.add_argument(
        "cases",
        nargs="*",
        default=list(DEFAULT_CASES),
        help="Case-study names under experiments/. Defaults to BIO-INSIGHT and MO-GENECI.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory where split manifests and assignment CSV files will be written.",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.20,
        help="Target fraction of fronts assigned to the held-out test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic tie-breaking.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing manifest files.",
    )
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def load_fronts_for_case(case: str) -> tuple[Path, pd.DataFrame, int]:
    csv_path = resolve_training_csv(case)
    df = pd.read_csv(csv_path)
    front_map = build_front_id_map(case)
    fronts_df = build_fronts_summary(df, front_map)
    return csv_path, fronts_df[ASSIGNMENT_COLUMNS].copy(), len(df)


def assign_split(fronts_df: pd.DataFrame, test_frac: float, seed: int) -> tuple[pd.DataFrame, dict[str, int], int]:
    quotas = family_test_quotas(fronts_df, test_frac)
    target_test_fronts = int(round(len(fronts_df) * test_frac))
    target_test_fronts = min(target_test_fronts, max(len(fronts_df) - 1, 0))
    assigned = fronts_df.copy()
    assigned["split"] = "development"

    for family, family_df in assigned.groupby("family", dropna=False, sort=True):
        test_front_ids = select_diverse_test_fronts(family_df, quotas[str(family)], seed)
        assigned.loc[assigned["front_id"].isin(test_front_ids), "split"] = "test"

    return assigned.sort_values(["front_id"]).reset_index(drop=True), quotas, target_test_fronts


def split_summary(split_df: pd.DataFrame) -> dict[str, Any]:
    split_counts = split_df["split"].value_counts().sort_index().to_dict()
    family_split_counts = (
        split_df.groupby(["family", "split"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .sort_index()
        .to_dict(orient="index")
    )
    return {
        "split_counts": {str(key): int(value) for key, value in split_counts.items()},
        "family_split_counts": {
            str(family): {str(split): int(count) for split, count in counts.items()}
            for family, counts in family_split_counts.items()
        },
    }


def write_manifest(
    case: str,
    csv_path: Path,
    n_rows: int,
    split_df: pd.DataFrame,
    quotas: dict[str, int],
    target_test_fronts: int,
    *,
    out_dir: Path,
    test_frac: float,
    seed: int,
    force: bool,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{case}_split_manifest.json"
    csv_out_path = out_dir / f"{case}_split_assignments.csv"

    if not force:
        existing = [path for path in (manifest_path, csv_out_path) if path.exists()]
        if existing:
            names = ", ".join(str(path) for path in existing)
            raise FileExistsError(f"Refusing to overwrite existing split artifact(s): {names}")

    split_df.to_csv(csv_out_path, index=False)
    summary = split_summary(split_df)

    manifest = {
        "schema_version": 1,
        "case": case,
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "created_by": "scripts/experiments/create_split_manifest.py",
        "git_commit": git_commit(),
        "input": {
            "training_csv": csv_path.name,
            "training_csv_sha256": sha256_file(csv_path),
            "n_rows": int(n_rows),
            "n_fronts": int(len(split_df)),
        },
        "protocol": {
            "name": "held_out_development_test_by_front",
            "front_key": "front_name",
            "group_column": "front_id",
            "seed": int(seed),
            "test_fraction": float(test_frac),
            "target_test_fronts": int(target_test_fronts),
            "actual_test_fronts": int((split_df["split"] == "test").sum()),
            "actual_development_fronts": int((split_df["split"] == "development").sum()),
            "quota_level": "family",
            "family_test_quotas": quotas,
            "within_family_selection": (
                "deterministic greedy metadata diversity over size, condition, and source; "
                "stable hash tie-break by front_name"
            ),
            "target_column_used_for_assignment": False,
            "singleton_family_policy": "development",
        },
        "summary": summary,
        "assignments_csv": repo_relative(csv_out_path),
        "assignments": split_df.where(pd.notna(split_df), None).to_dict(orient="records"),
    }

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path, csv_out_path


def validate_shared_front_assignment(manifest_paths: list[Path]) -> None:
    if len(manifest_paths) < 2:
        return

    reference: dict[str, str] | None = None
    reference_case = ""
    for path in manifest_paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assignment = {row["front_name"]: row["split"] for row in manifest["assignments"]}
        if reference is None:
            reference = assignment
            reference_case = manifest["case"]
            continue
        if assignment != reference:
            raise ValueError(
                f"Split assignment for {manifest['case']} does not match {reference_case} by front_name."
            )


def main() -> None:
    args = build_parser().parse_args()
    written_manifests: list[Path] = []

    for case in args.cases:
        csv_path, fronts_df, n_rows = load_fronts_for_case(case)
        split_df, quotas, target_test_fronts = assign_split(fronts_df, args.test_frac, args.seed)
        manifest_path, csv_out_path = write_manifest(
            case,
            csv_path,
            n_rows,
            split_df,
            quotas,
            target_test_fronts,
            out_dir=args.out_dir,
            test_frac=args.test_frac,
            seed=args.seed,
            force=args.force,
        )
        written_manifests.append(manifest_path)
        counts = split_df["split"].value_counts().sort_index().to_dict()
        print(f"[{case}] wrote {manifest_path}")
        print(f"  assignments: {csv_out_path}")
        print(f"  split: {counts}")

    validate_shared_front_assignment(written_manifests)


if __name__ == "__main__":
    main()
