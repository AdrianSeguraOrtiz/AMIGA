#!/usr/bin/env python3
"""Validate AMIGA case-study data contracts against datasets and split manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from inspect_case import DEFAULT_CASES, REPO_ROOT, git_commit


CONFIG_DIR = REPO_ROOT / "docs" / "experiments" / "cases"
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "experiments" / "contracts"
VALID_OBJECTIVE_DIRECTIONS = {"minimize", "maximize"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate case-study configs, datasets, split manifests, feature blocks, "
            "and ablation feature sets."
        )
    )
    parser.add_argument(
        "cases",
        nargs="*",
        default=list(DEFAULT_CASES),
        help="Case-study names. Defaults to BIO-INSIGHT and MO-GENECI.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=CONFIG_DIR,
        help="Directory containing <CASE>.json case configs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory where validated manifests will be written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing manifest files.",
    )
    return parser


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def resolve_case_data_csv(case: str, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()

    repo_candidate = (REPO_ROOT / candidate).resolve()
    if repo_candidate.exists():
        return repo_candidate

    return (REPO_ROOT / "experiments" / case / "data" / candidate.name).resolve()


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def columns_by_prefix(columns: Iterable[str], prefixes: Iterable[str]) -> list[str]:
    prefixes_tuple = tuple(prefixes)
    return [column for column in columns if column.startswith(prefixes_tuple)]


def resolve_feature_blocks(config: dict[str, Any], df: pd.DataFrame) -> dict[str, list[str]]:
    blocks: dict[str, list[str]] = {}
    all_columns = list(df.columns)
    excluded_feature_columns = set(config.get("excluded_feature_columns", []))

    for block_name, block_config in config["feature_blocks"].items():
        columns = list(block_config.get("columns", []))
        columns.extend(columns_by_prefix(all_columns, block_config.get("prefixes", [])))
        columns = unique_preserving_order(columns)
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise ValueError(f"{config['case']} block {block_name} has missing columns: {missing}")
        blocks[block_name] = [column for column in columns if column not in excluded_feature_columns]

    return blocks


def resolve_feature_sets(config: dict[str, Any], blocks: dict[str, list[str]]) -> dict[str, list[str]]:
    feature_sets: dict[str, list[str]] = {}
    for feature_set, block_names in config["feature_sets"].items():
        unknown_blocks = [block_name for block_name in block_names if block_name not in blocks]
        if unknown_blocks:
            raise ValueError(f"{config['case']} feature set {feature_set} references unknown blocks: {unknown_blocks}")

        columns: list[str] = []
        for block_name in block_names:
            columns.extend(blocks[block_name])
        feature_sets[feature_set] = unique_preserving_order(columns)
    return feature_sets


def find_duplicate_block_columns(blocks: dict[str, list[str]]) -> dict[str, list[str]]:
    owners: dict[str, list[str]] = {}
    for block_name, columns in blocks.items():
        for column in columns:
            owners.setdefault(column, []).append(block_name)
    return {column: block_names for column, block_names in owners.items() if len(block_names) > 1}


def validate_objectives(config: dict[str, Any], errors: list[str]) -> None:
    objective_columns = list(config["objective_columns"])
    directions = dict(config["objective_directions"])

    if set(objective_columns) != set(directions):
        errors.append("objective_columns and objective_directions keys do not match.")

    invalid = {column: direction for column, direction in directions.items() if direction not in VALID_OBJECTIVE_DIRECTIONS}
    if invalid:
        errors.append(f"Invalid objective directions: {invalid}")

    non_minimized = {column: direction for column, direction in directions.items() if direction != "minimize"}
    if non_minimized:
        errors.append(f"All objectives must currently be minimization objectives, found: {non_minimized}")


def validate_columns(
    config: dict[str, Any],
    df: pd.DataFrame,
    blocks: dict[str, list[str]],
    feature_sets: dict[str, list[str]],
    errors: list[str],
) -> dict[str, Any]:
    group_column = config["group_column"]
    item_column = config["item_column"]
    target_column = config["target_column"]
    control_columns = [group_column, item_column]
    excluded_feature_columns = list(config.get("excluded_feature_columns", []))
    required = control_columns + [target_column] + list(config["objective_columns"])

    missing_required = [column for column in required if column not in df.columns]
    if missing_required:
        errors.append(f"Missing required columns: {missing_required}")

    missing_excluded = [column for column in excluded_feature_columns if column not in df.columns]
    if missing_excluded:
        errors.append(f"Excluded feature columns are not present in the dataset: {missing_excluded}")

    duplicate_block_columns = find_duplicate_block_columns(blocks)
    if duplicate_block_columns:
        errors.append(f"Columns assigned to multiple feature blocks: {duplicate_block_columns}")

    feature_columns = unique_preserving_order(column for columns in blocks.values() for column in columns)
    forbidden_features = [column for column in feature_columns if column in control_columns or column == target_column]
    if forbidden_features:
        errors.append(f"Control or target columns included as features: {forbidden_features}")

    expected_columns = set(control_columns + [target_column] + feature_columns + excluded_feature_columns)
    unassigned_columns = [column for column in df.columns if column not in expected_columns]
    if unassigned_columns and not config.get("allow_unassigned_columns", False):
        errors.append(f"Unassigned columns are not allowed: {unassigned_columns}")

    numeric_columns = feature_columns + [target_column]
    non_numeric_columns = [column for column in numeric_columns if column in df.columns and not pd.api.types.is_numeric_dtype(df[column])]
    if non_numeric_columns:
        errors.append(f"Non-numeric model/target columns: {non_numeric_columns}")

    missing_values = {
        column: int(df[column].isna().sum())
        for column in numeric_columns
        if column in df.columns and int(df[column].isna().sum()) > 0
    }
    if missing_values:
        errors.append(f"Missing values found in model/target columns: {missing_values}")

    constant_features = [
        column
        for column in feature_columns
        if column in df.columns and int(df[column].nunique(dropna=True)) <= 1
    ]
    if constant_features:
        errors.append(f"Constant feature columns found: {constant_features}")

    missing_full = [column for column in feature_columns if column not in feature_sets.get("full", [])]
    if missing_full:
        errors.append(f"Full feature set does not include every feature column: {missing_full}")

    return {
        "control_columns": control_columns,
        "excluded_feature_columns": excluded_feature_columns,
        "feature_columns": feature_columns,
        "unassigned_columns": unassigned_columns,
        "constant_features": constant_features,
        "n_columns_by_block": {block_name: len(columns) for block_name, columns in blocks.items()},
        "n_columns_by_feature_set": {name: len(columns) for name, columns in feature_sets.items()},
    }


def validate_split(
    config: dict[str, Any],
    df: pd.DataFrame,
    split_manifest: dict[str, Any],
    data_sha256: str,
    errors: list[str],
) -> dict[str, Any]:
    case = config["case"]
    group_column = config["group_column"]
    primary_k = int(config["primary_k"])

    if split_manifest.get("case") != case:
        errors.append(f"Split manifest case mismatch: {split_manifest.get('case')} != {case}")

    manifest_sha = split_manifest.get("input", {}).get("training_csv_sha256")
    if manifest_sha and manifest_sha != data_sha256:
        errors.append("Split manifest input SHA256 does not match current dataset.")

    assignments = pd.DataFrame(split_manifest.get("assignments", []))
    if assignments.empty:
        errors.append("Split manifest has no assignments.")
        return {}

    required_assignment_columns = {"front_id", "front_name", "split"}
    missing_assignment_columns = sorted(required_assignment_columns - set(assignments.columns))
    if missing_assignment_columns:
        errors.append(f"Split assignments missing columns: {missing_assignment_columns}")
        return {}

    invalid_split_values = sorted(set(assignments["split"]) - {"development", "test"})
    if invalid_split_values:
        errors.append(f"Invalid split values: {invalid_split_values}")

    duplicated_front_ids = assignments.loc[assignments["front_id"].duplicated(), "front_id"].tolist()
    if duplicated_front_ids:
        errors.append(f"Duplicated front IDs in split manifest: {duplicated_front_ids}")

    data_front_ids = set(df[group_column].dropna().astype(int).unique().tolist())
    manifest_front_ids = set(assignments["front_id"].dropna().astype(int).unique().tolist())
    if data_front_ids != manifest_front_ids:
        errors.append(
            "Dataset front IDs and split manifest front IDs differ: "
            f"missing_in_manifest={sorted(data_front_ids - manifest_front_ids)}, "
            f"missing_in_data={sorted(manifest_front_ids - data_front_ids)}"
        )

    front_counts = df.groupby(group_column).size()
    too_small = front_counts[front_counts < primary_k]
    if not too_small.empty:
        errors.append(f"Fronts with fewer than primary_k={primary_k} items: {too_small.to_dict()}")

    split_by_front = assignments[["front_id", "split"]].rename(columns={"front_id": group_column})
    merged = df[[group_column]].merge(split_by_front, on=group_column, how="left")
    missing_split_rows = int(merged["split"].isna().sum())
    if missing_split_rows:
        errors.append(f"Rows without split assignment: {missing_split_rows}")

    rows_by_split = merged["split"].value_counts().sort_index().to_dict()
    fronts_by_split = assignments["split"].value_counts().sort_index().to_dict()
    return {
        "fronts_by_split": {str(key): int(value) for key, value in fronts_by_split.items()},
        "rows_by_split": {str(key): int(value) for key, value in rows_by_split.items()},
        "min_items_per_front": int(front_counts.min()),
        "max_items_per_front": int(front_counts.max()),
    }


def target_summary(df: pd.DataFrame, target_column: str) -> dict[str, float]:
    target = pd.to_numeric(df[target_column], errors="coerce")
    return {
        "min": float(target.min()),
        "max": float(target.max()),
        "mean": float(target.mean()),
        "std": float(target.std(ddof=0)),
    }


def output_paths(case: str, out_dir: Path) -> dict[str, Path]:
    return {
        "case_manifest": out_dir / f"{case}_case_manifest.json",
        "data_manifest": out_dir / f"{case}_data_manifest.json",
        "feature_columns": out_dir / f"{case}_feature_columns.json",
        "validation_report": out_dir / f"{case}_validation_report.json",
    }


def ensure_writable(paths: dict[str, Path], force: bool) -> None:
    if force:
        return
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing contract artifact(s): {names}")


def validate_case(case: str, *, config_dir: Path, out_dir: Path, force: bool) -> dict[str, Path]:
    config_path = (config_dir / f"{case}.json").resolve()
    config = read_json(config_path)
    if config.get("case") != case:
        raise ValueError(f"Config case mismatch: {config.get('case')} != {case}")

    data_csv = resolve_case_data_csv(case, config["data_csv"])
    split_manifest_path = resolve_repo_path(config["split_manifest"])
    df = pd.read_csv(data_csv)
    split_manifest = read_json(split_manifest_path)
    data_sha256 = sha256_file(data_csv)

    errors: list[str] = []
    warnings: list[str] = []

    validate_objectives(config, errors)
    blocks = resolve_feature_blocks(config, df)
    feature_sets = resolve_feature_sets(config, blocks)
    column_report = validate_columns(config, df, blocks, feature_sets, errors)
    split_report = validate_split(config, df, split_manifest, data_sha256, errors)

    if column_report["unassigned_columns"] and config.get("allow_unassigned_columns", False):
        warnings.append(f"Unassigned columns allowed by config: {column_report['unassigned_columns']}")

    paths = output_paths(case, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_writable(paths, force)

    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    common = {
        "schema_version": 1,
        "case": case,
        "generated_at_utc": generated_at,
        "generated_by": "scripts/experiments/validate_case_contract.py",
        "git_commit": git_commit(),
        "config_path": repo_relative(config_path),
    }

    case_manifest = {
        **common,
        "data_manifest": repo_relative(paths["data_manifest"]),
        "feature_columns": repo_relative(paths["feature_columns"]),
        "validation_report": repo_relative(paths["validation_report"]),
        "split_manifest": repo_relative(split_manifest_path),
        "primary_metric": config["primary_metric"],
        "primary_k": int(config["primary_k"]),
        "target_column": config["target_column"],
        "group_column": config["group_column"],
        "item_column": config["item_column"],
        "objective_directions": config["objective_directions"],
    }

    data_manifest = {
        **common,
        "data_csv": data_csv.name,
        "data_csv_sha256": data_sha256,
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "n_fronts": int(df[config["group_column"]].nunique()),
        "columns": list(df.columns),
        "target_summary": target_summary(df, config["target_column"]),
        "split_summary": split_report,
    }

    feature_columns = {
        **common,
        "target_column": config["target_column"],
        "control_columns": column_report["control_columns"],
        "excluded_feature_columns": column_report["excluded_feature_columns"],
        "excluded_feature_column_reasons": config.get("excluded_feature_column_reasons", {}),
        "objective_columns": list(config["objective_columns"]),
        "objective_directions": config["objective_directions"],
        "feature_blocks": blocks,
        "feature_sets": feature_sets,
        "counts": {
            "feature_blocks": column_report["n_columns_by_block"],
            "feature_sets": column_report["n_columns_by_feature_set"],
            "all_features": len(column_report["feature_columns"]),
        },
    }

    validation_report = {
        **common,
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "objectives_all_minimize": not any(
                direction != "minimize" for direction in config["objective_directions"].values()
            ),
            "no_target_or_control_columns_in_features": not any(
                column in column_report["feature_columns"]
                for column in column_report["control_columns"] + [config["target_column"]]
            ),
            "no_unassigned_columns": not column_report["unassigned_columns"],
            "no_constant_features": not column_report["constant_features"],
            "split_manifest_matches_dataset": not errors,
        },
        "column_report": column_report,
        "split_report": split_report,
    }

    if errors:
        write_json(paths["validation_report"], validation_report)
        raise SystemExit(f"{case} contract validation failed; see {paths['validation_report']}")

    write_json(paths["case_manifest"], case_manifest)
    write_json(paths["data_manifest"], data_manifest)
    write_json(paths["feature_columns"], feature_columns)
    write_json(paths["validation_report"], validation_report)

    print(f"[{case}] contract validation passed")
    print(f"  case manifest: {paths['case_manifest']}")
    print(f"  feature columns: {paths['feature_columns']}")
    return paths


def main() -> None:
    args = build_parser().parse_args()
    for case in args.cases:
        validate_case(case, config_dir=args.config_dir, out_dir=args.out_dir, force=args.force)


if __name__ == "__main__":
    main()
