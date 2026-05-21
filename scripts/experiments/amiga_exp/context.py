"""Case-context loading for AMIGA experimental phases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASE_CONFIG_DIR = REPO_ROOT / "docs" / "experiments" / "cases"
DEFAULT_FEATURE_CONTRACT_DIR = REPO_ROOT / "docs" / "experiments" / "contracts"
RESULTS_ROOT_NAME = "amiga-exp"


class CaseContextError(ValueError):
    """Raised when an experimental case cannot be loaded or validated."""


@dataclass(frozen=True)
class BasicCaseInfo:
    """Minimal case-folder information."""

    case_dir: Path
    case_name: str
    data_dir: Path
    data_csvs: tuple[Path, ...]
    audit_dir: Path


@dataclass(frozen=True)
class CaseContext:
    """Resolved case context shared by all experimental phases."""

    case_dir: Path
    case_name: str
    data_dir: Path
    data_csv: Path
    config_path: Path
    config: dict[str, Any]
    split_manifest_path: Path
    split_manifest: dict[str, Any]
    feature_contract_path: Path
    feature_contract: dict[str, Any]
    df: pd.DataFrame
    split_assignments: pd.DataFrame
    development_df: pd.DataFrame
    test_df: pd.DataFrame
    group_column: str
    item_column: str
    target_column: str
    control_columns: list[str]
    objective_columns: list[str]
    objective_directions: dict[str, str]
    feature_sets: dict[str, list[str]]
    results_root: Path

    @property
    def n_rows(self) -> int:
        return int(len(self.df))

    @property
    def n_fronts(self) -> int:
        return int(self.df[self.group_column].nunique())

    @property
    def n_development_rows(self) -> int:
        return int(len(self.development_df))

    @property
    def n_test_rows(self) -> int:
        return int(len(self.test_df))

    @property
    def n_development_fronts(self) -> int:
        return int(self.development_df[self.group_column].nunique())

    @property
    def n_test_fronts(self) -> int:
        return int(self.test_df[self.group_column].nunique())


def inspect_basic_case(case_dir: Path, *, case_name: str | None = None) -> BasicCaseInfo:
    """Inspect a case folder without requiring versioned configs."""
    resolved = case_dir.expanduser().resolve()
    if not resolved.exists():
        raise CaseContextError(f"case directory does not exist: {resolved}")
    if not resolved.is_dir():
        raise CaseContextError(f"case path is not a directory: {resolved}")

    data_dir = resolved / "data"
    data_csvs = tuple(sorted(data_dir.glob("data_*.csv"))) if data_dir.exists() else ()
    return BasicCaseInfo(
        case_dir=resolved,
        case_name=case_name or resolved.name,
        data_dir=data_dir,
        data_csvs=data_csvs,
        audit_dir=data_dir / "audit",
    )


def require_basic_case_layout(case_info: BasicCaseInfo) -> None:
    """Validate the minimal folder layout expected for a case."""
    if not case_info.data_dir.exists():
        raise CaseContextError(f"case data directory does not exist: {case_info.data_dir}")
    if not case_info.data_dir.is_dir():
        raise CaseContextError(f"case data path is not a directory: {case_info.data_dir}")
    if not case_info.data_csvs:
        raise CaseContextError(f"no data_*.csv file found under: {case_info.data_dir}")


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CaseContextError(f"required JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CaseContextError(f"invalid JSON file: {path}") from exc


def resolve_path(path: str | Path, *, base_dir: Path = REPO_ROOT) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.expanduser().resolve()


def default_config_path(case_name: str) -> Path:
    return DEFAULT_CASE_CONFIG_DIR / f"{case_name}.json"


def default_feature_contract_path(case_name: str) -> Path:
    return DEFAULT_FEATURE_CONTRACT_DIR / f"{case_name}_feature_columns.json"


def select_data_csv(case_info: BasicCaseInfo, config: dict[str, Any]) -> Path:
    """Select the case data CSV, preferring the case folder over repo-root paths."""
    require_basic_case_layout(case_info)
    if len(case_info.data_csvs) == 1:
        return case_info.data_csvs[0]

    configured_name = Path(str(config.get("data_csv", ""))).name
    matches = [path for path in case_info.data_csvs if path.name == configured_name]
    if len(matches) == 1:
        return matches[0]

    names = ", ".join(path.name for path in case_info.data_csvs)
    raise CaseContextError(
        f"multiple data_*.csv files found under {case_info.data_dir}; "
        f"could not disambiguate using config data_csv. Candidates: {names}"
    )


def resolve_case_name(case_info: BasicCaseInfo, config: dict[str, Any], explicit_case_name: str | None) -> str:
    if explicit_case_name:
        case_name = explicit_case_name
    elif config.get("case"):
        case_name = str(config["case"])
    else:
        case_name = case_info.case_name

    config_case = config.get("case")
    if config_case and str(config_case) != case_name:
        raise CaseContextError(f"config case '{config_case}' does not match requested case '{case_name}'")
    return case_name


def load_config(case_info: BasicCaseInfo, *, case_name: str | None, config_path: Path | None) -> tuple[str, Path, dict[str, Any]]:
    initial_case_name = case_name or case_info.case_name
    resolved_config_path = resolve_path(config_path) if config_path is not None else default_config_path(initial_case_name)
    config = read_json(resolved_config_path)
    resolved_case_name = resolve_case_name(case_info, config, case_name)
    return resolved_case_name, resolved_config_path, config


def load_split_manifest(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    split_manifest_value = config.get("split_manifest")
    if not split_manifest_value:
        raise CaseContextError("case config is missing 'split_manifest'")
    split_manifest_path = resolve_path(split_manifest_value)
    return split_manifest_path, read_json(split_manifest_path)


def load_feature_contract(case_name: str, config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    feature_contract_value = config.get("feature_contract")
    feature_contract_path = (
        resolve_path(feature_contract_value)
        if feature_contract_value
        else default_feature_contract_path(case_name)
    )
    return feature_contract_path, read_json(feature_contract_path)


def validate_manifest_cases(case_name: str, split_manifest: dict[str, Any], feature_contract: dict[str, Any]) -> None:
    split_case = split_manifest.get("case")
    if split_case and split_case != case_name:
        raise CaseContextError(f"split manifest case '{split_case}' does not match case '{case_name}'")

    contract_case = feature_contract.get("case")
    if contract_case and contract_case != case_name:
        raise CaseContextError(f"feature contract case '{contract_case}' does not match case '{case_name}'")


def validate_feature_contract_columns(df: pd.DataFrame, feature_contract: dict[str, Any]) -> None:
    missing: list[str] = []
    for key in ("target_column",):
        column = feature_contract.get(key)
        if column and column not in df.columns:
            missing.append(str(column))

    for key in ("control_columns", "objective_columns"):
        for column in feature_contract.get(key, []):
            if column not in df.columns:
                missing.append(str(column))

    for feature_set, columns in feature_contract.get("feature_sets", {}).items():
        missing_columns = [str(column) for column in columns if column not in df.columns]
        if missing_columns:
            raise CaseContextError(f"feature set '{feature_set}' has missing columns: {missing_columns}")

    if missing:
        raise CaseContextError(f"feature contract references missing columns: {sorted(set(missing))}")


def build_split_assignments(split_manifest: dict[str, Any], group_column: str) -> pd.DataFrame:
    assignments = pd.DataFrame(split_manifest.get("assignments", []))
    if assignments.empty:
        raise CaseContextError("split manifest has no assignments")
    if "front_id" not in assignments.columns or "split" not in assignments.columns:
        raise CaseContextError("split manifest assignments must contain front_id and split")

    assignments = assignments[["front_id", "split"]].rename(columns={"front_id": group_column}).copy()
    assignments[group_column] = assignments[group_column].astype(int)
    assignments["split"] = assignments["split"].astype(str)
    invalid = sorted(set(assignments["split"]) - {"development", "test"})
    if invalid:
        raise CaseContextError(f"invalid split values in split manifest: {invalid}")

    split_counts_by_front = assignments.groupby(group_column)["split"].nunique()
    mixed_fronts = split_counts_by_front[split_counts_by_front > 1].index.tolist()
    if mixed_fronts:
        raise CaseContextError(f"fronts assigned to multiple splits: {mixed_fronts}")

    duplicated = assignments[assignments[group_column].duplicated()][group_column].tolist()
    if duplicated:
        raise CaseContextError(f"duplicated front IDs in split manifest: {duplicated}")

    return assignments


def attach_splits(df: pd.DataFrame, assignments: pd.DataFrame, group_column: str) -> pd.DataFrame:
    if group_column not in df.columns:
        raise CaseContextError(f"dataset is missing group column '{group_column}'")

    working = df.copy()
    working[group_column] = working[group_column].astype(int)
    merged = working.merge(assignments, on=group_column, how="left", validate="many_to_one")
    missing_count = int(merged["split"].isna().sum())
    if missing_count:
        missing_fronts = sorted(merged.loc[merged["split"].isna(), group_column].unique().tolist())
        raise CaseContextError(
            f"{missing_count} dataset rows have no split assignment; missing fronts: {missing_fronts}"
        )

    development_fronts = set(merged.loc[merged["split"] == "development", group_column].unique().tolist())
    test_fronts = set(merged.loc[merged["split"] == "test", group_column].unique().tolist())
    overlap = sorted(development_fronts & test_fronts)
    if overlap:
        raise CaseContextError(f"fronts appear in both development and test rows: {overlap}")

    return merged


def load_case_context(
    case_dir: Path,
    *,
    case_name: str | None = None,
    config_path: Path | None = None,
) -> CaseContext:
    """Load and validate the shared context for an experimental case."""
    basic = inspect_basic_case(case_dir, case_name=case_name)
    require_basic_case_layout(basic)
    resolved_case_name, resolved_config_path, config = load_config(
        basic,
        case_name=case_name,
        config_path=config_path,
    )
    basic = inspect_basic_case(case_dir, case_name=resolved_case_name)
    data_csv = select_data_csv(basic, config)
    split_manifest_path, split_manifest = load_split_manifest(config)
    feature_contract_path, feature_contract = load_feature_contract(resolved_case_name, config)
    validate_manifest_cases(resolved_case_name, split_manifest, feature_contract)

    df = pd.read_csv(data_csv)
    validate_feature_contract_columns(df, feature_contract)

    group_column = str(feature_contract.get("control_columns", [config.get("group_column", "front_id")])[0])
    item_column = str(config.get("item_column", "item_id"))
    target_column = str(feature_contract.get("target_column", config.get("target_column", "AUPR")))
    control_columns = [str(column) for column in feature_contract.get("control_columns", [])]
    objective_columns = [str(column) for column in feature_contract.get("objective_columns", [])]
    objective_directions = {str(key): str(value) for key, value in feature_contract.get("objective_directions", {}).items()}
    feature_sets = {
        str(name): [str(column) for column in columns]
        for name, columns in feature_contract.get("feature_sets", {}).items()
    }

    assignments = build_split_assignments(split_manifest, group_column)
    merged = attach_splits(df, assignments, group_column)
    development_df = merged[merged["split"] == "development"].drop(columns=["split"]).copy()
    test_df = merged[merged["split"] == "test"].drop(columns=["split"]).copy()

    if development_df.empty:
        raise CaseContextError("development split is empty")
    if test_df.empty:
        raise CaseContextError("test split is empty")

    return CaseContext(
        case_dir=basic.case_dir,
        case_name=resolved_case_name,
        data_dir=basic.data_dir,
        data_csv=data_csv,
        config_path=resolved_config_path,
        config=config,
        split_manifest_path=split_manifest_path,
        split_manifest=split_manifest,
        feature_contract_path=feature_contract_path,
        feature_contract=feature_contract,
        df=df,
        split_assignments=assignments,
        development_df=development_df,
        test_df=test_df,
        group_column=group_column,
        item_column=item_column,
        target_column=target_column,
        control_columns=control_columns,
        objective_columns=objective_columns,
        objective_directions=objective_directions,
        feature_sets=feature_sets,
        results_root=basic.case_dir / "results" / RESULTS_ROOT_NAME,
    )


def context_summary(context: CaseContext) -> dict[str, Any]:
    """Return a compact, print-friendly summary of a loaded case context."""
    return {
        "case_name": context.case_name,
        "case_dir": str(context.case_dir),
        "data_csv": str(context.data_csv),
        "config_path": str(context.config_path),
        "split_manifest": str(context.split_manifest_path),
        "feature_contract": str(context.feature_contract_path),
        "rows": context.n_rows,
        "fronts": context.n_fronts,
        "development_rows": context.n_development_rows,
        "development_fronts": context.n_development_fronts,
        "test_rows": context.n_test_rows,
        "test_fronts": context.n_test_fronts,
        "target_column": context.target_column,
        "control_columns": context.control_columns,
        "objective_columns": context.objective_columns,
        "objective_directions": context.objective_directions,
        "feature_sets": sorted(context.feature_sets),
        "results_root": str(context.results_root),
    }
