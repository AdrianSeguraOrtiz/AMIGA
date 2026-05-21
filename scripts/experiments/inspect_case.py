#!/usr/bin/env python3
"""Audit AMIGA experiment case-study datasets before building the final pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
DEFAULT_CASES = ("BIO-INSIGHT", "MO-GENECI")
CONTROL_COLUMNS = {"front_id", "item_id"}
TARGET_COLUMN = "AUPR"
FEATURE_PREFIX_BLOCKS = (
    ("technique_weights", ("GRN_",)),
    ("expression", ("expr_", "exp_", "ts_")),
    ("network", ("grn_", "net_", "entropy_")),
)


@dataclass(frozen=True)
class FrontMetadata:
    front_name: str
    family: str
    source: str
    size: int | None
    condition: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect one or more AMIGA experimental case-study datasets and write "
            "front, column, and split-audit summaries."
        )
    )
    parser.add_argument(
        "cases",
        nargs="*",
        default=list(DEFAULT_CASES),
        help="Case-study names under experiments/. Defaults to BIO-INSIGHT and MO-GENECI.",
    )
    parser.add_argument(
        "--training-csv",
        type=Path,
        default=None,
        help=(
            "Optional explicit data_*.csv path. Only valid when inspecting one case. "
            "By default, the first experiments/<CASE>/data/data_*.csv is used."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. Only valid when inspecting one case. "
            "By default, writes to experiments/<CASE>/data/audit/."
        ),
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.20,
        help="Approximate fraction of fronts assigned to the held-out test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for the deterministic candidate split.",
    )
    return parser


def resolve_training_csv(case: str, explicit_path: Path | None = None) -> Path:
    if explicit_path is not None:
        path = explicit_path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Training CSV not found: {path}")
        return path

    data_dir = EXPERIMENTS_DIR / case / "data"
    candidates = sorted(data_dir.glob("data_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No data_*.csv found in: {data_dir}")
    return candidates[0]


def infer_front_metadata(front_name: str) -> FrontMetadata:
    clean_name = front_name.removesuffix("_exp")

    match = re.match(r"^InSilicoSize(?P<size>\d+)-(?P<source>[A-Za-z]+)(?P<rep>\d+)-(?P<condition>.+)$", clean_name)
    if match:
        return FrontMetadata(
            front_name=front_name,
            family="InSilico",
            source=f"{match.group('source')}{match.group('rep')}",
            size=int(match.group("size")),
            condition=match.group("condition"),
        )

    match = re.match(r"^dream4_(?P<size>\d+)_(?P<rep>\d+)$", clean_name)
    if match:
        return FrontMetadata(
            front_name=front_name,
            family="dream4",
            source=f"dream4_{match.group('rep')}",
            size=int(match.group("size")),
            condition="unknown",
        )

    match = re.match(r"^(?P<family>gnw|rogers|syntren)(?P<size>\d+)$", clean_name)
    if match:
        return FrontMetadata(
            front_name=front_name,
            family=match.group("family"),
            source=match.group("family"),
            size=int(match.group("size")),
            condition="unknown",
        )

    match = re.match(r"^switch-(?P<condition>.+)$", clean_name)
    if match:
        return FrontMetadata(
            front_name=front_name,
            family="switch",
            source="switch",
            size=None,
            condition=match.group("condition"),
        )

    match = re.match(r"^sim_(?P<body>.+?)_(?P<condition>mixed|knockdown|knockout|overexpression)$", clean_name)
    if match:
        body = match.group("body")
        condition = match.group("condition")

        size_match = re.search(r"_size-(?P<size>\d+)", body)
        size = int(size_match.group("size")) if size_match else None

        if body.startswith("BioGrid_"):
            family = "BioGrid"
            source = body[len("BioGrid_") :]
        elif body.startswith("GRNdb_"):
            family = "GRNdb"
            source = body[len("GRNdb_") :]
        elif body.startswith("RegNetwork_"):
            family = "RegNetwork"
            source = body[len("RegNetwork_") :]
        elif body.startswith("TFLink_"):
            family = "TFLink"
            source = body[len("TFLink_") :]
        elif body.startswith("eipo-modular"):
            family = "eipo-modular"
            source = "eipo-modular"
        elif body.startswith("scale-free"):
            family = "scale-free"
            source = "scale-free"
        else:
            family = body.split("_", 1)[0]
            source = body

        return FrontMetadata(
            front_name=front_name,
            family=family,
            source=source,
            size=size,
            condition=condition,
        )

    return FrontMetadata(
        front_name=front_name,
        family=clean_name.split("_", 1)[0].split("-", 1)[0],
        source=clean_name,
        size=None,
        condition="unknown",
    )


def build_front_id_map(case: str) -> dict[int, FrontMetadata]:
    front_dir = EXPERIMENTS_DIR / case / "data" / "fronts"
    mapping: dict[int, FrontMetadata] = {}
    if not front_dir.exists():
        return mapping

    for data_csv in sorted(front_dir.glob("*/data.csv")):
        header = pd.read_csv(data_csv, usecols=["front_id"])
        unique_ids = sorted(header["front_id"].dropna().astype(int).unique().tolist())
        metadata = infer_front_metadata(data_csv.parent.name)
        for front_id in unique_ids:
            mapping[front_id] = metadata
    return mapping


def classify_column(column: str) -> str:
    if column in CONTROL_COLUMNS:
        return "control"
    if column == TARGET_COLUMN:
        return "target"
    for block_name, prefixes in FEATURE_PREFIX_BLOCKS:
        if column.startswith(prefixes):
            return block_name
    return "objective_or_metadata"


def numeric_summary(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    non_na = numeric.dropna()
    if non_na.empty:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "zero_fraction": None,
        }
    return {
        "min": float(non_na.min()),
        "max": float(non_na.max()),
        "mean": float(non_na.mean()),
        "std": float(non_na.std(ddof=0)),
        "zero_fraction": float((non_na == 0).mean()),
    }


def build_columns_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        series = df[column]
        missing_count = int(series.isna().sum())
        n_unique = int(series.nunique(dropna=True))
        summary = numeric_summary(series) if pd.api.types.is_numeric_dtype(series) else {}
        rows.append(
            {
                "column": column,
                "block": classify_column(column),
                "dtype": str(series.dtype),
                "non_null_count": int(series.notna().sum()),
                "missing_count": missing_count,
                "missing_fraction": float(missing_count / max(len(series), 1)),
                "n_unique": n_unique,
                "is_constant": bool(n_unique <= 1),
                **summary,
            }
        )
    return pd.DataFrame(rows)


def build_fronts_summary(df: pd.DataFrame, front_map: dict[int, FrontMetadata]) -> pd.DataFrame:
    rows = []
    for front_id, group in df.groupby("front_id", sort=True):
        metadata = front_map.get(
            int(front_id),
            FrontMetadata(
                front_name=f"front_{front_id}",
                family="unknown",
                source="unknown",
                size=None,
                condition="unknown",
            ),
        )
        target = pd.to_numeric(group[TARGET_COLUMN], errors="coerce")
        rows.append(
            {
                "front_id": int(front_id),
                **asdict(metadata),
                "n_items": int(len(group)),
                "target_min": float(target.min()),
                "target_max": float(target.max()),
                "target_mean": float(target.mean()),
                "target_std": float(target.std(ddof=0)),
                "target_range": float(target.max() - target.min()),
                "target_n_unique": int(target.nunique(dropna=True)),
                "best_item_id": int(group.loc[target.idxmax(), "item_id"]) if target.notna().any() else None,
            }
        )
    return pd.DataFrame(rows)


def stable_random_key(value: str, seed: int) -> int:
    payload = f"{seed}:{value}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def family_test_quotas(fronts_df: pd.DataFrame, test_frac: float) -> dict[str, int]:
    family_counts = fronts_df["family"].value_counts().sort_index()
    target_test_fronts = int(round(len(fronts_df) * test_frac))
    target_test_fronts = min(target_test_fronts, max(len(fronts_df) - 1, 0))

    quota_rows = []
    for family, count in family_counts.items():
        count = int(count)
        raw = count * target_test_fronts / max(len(fronts_df), 1)
        min_quota = 1 if count > 1 else 0
        max_quota = max(count - 1, 0)
        quota = int(raw)
        quota = min(max(quota, min_quota), max_quota)
        quota_rows.append(
            {
                "family": str(family),
                "raw": raw,
                "remainder": raw - int(raw),
                "min": min_quota,
                "max": max_quota,
                "quota": quota,
            }
        )

    quota_df = pd.DataFrame(quota_rows)

    while int(quota_df["quota"].sum()) < target_test_fronts:
        candidates = quota_df[quota_df["quota"] < quota_df["max"]].copy()
        if candidates.empty:
            break
        candidates["_tie"] = candidates["family"].map(lambda value: stable_random_key(str(value), 0))
        chosen_idx = candidates.sort_values(["remainder", "_tie"], ascending=[False, True]).index[0]
        quota_df.loc[chosen_idx, "quota"] += 1

    while int(quota_df["quota"].sum()) > target_test_fronts:
        candidates = quota_df[quota_df["quota"] > quota_df["min"]].copy()
        if candidates.empty:
            break
        candidates["_tie"] = candidates["family"].map(lambda value: stable_random_key(str(value), 0))
        chosen_idx = candidates.sort_values(["remainder", "_tie"], ascending=[True, False]).index[0]
        quota_df.loc[chosen_idx, "quota"] -= 1

    return {str(row.family): int(row.quota) for row in quota_df.itertuples(index=False)}


def metadata_value(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    return str(value)


def select_diverse_test_fronts(family_df: pd.DataFrame, n_test: int, seed: int) -> set[int]:
    if n_test <= 0:
        return set()

    remaining = family_df.copy()
    remaining["_split_key"] = remaining["front_name"].map(lambda value: stable_random_key(str(value), seed))
    selected_indices: list[int] = []

    for _ in range(n_test):
        if remaining.empty:
            break

        selected = family_df.loc[selected_indices] if selected_indices else pd.DataFrame(columns=family_df.columns)
        selected_sizes = {metadata_value(value) for value in selected["size"]} if not selected.empty else set()
        selected_conditions = {metadata_value(value) for value in selected["condition"]} if not selected.empty else set()
        selected_sources = {metadata_value(value) for value in selected["source"]} if not selected.empty else set()

        scored = remaining.copy()
        scored["_novelty_score"] = scored.apply(
            lambda row: (
                8 * (metadata_value(row["size"]) not in selected_sizes)
                + 4 * (metadata_value(row["condition"]) not in selected_conditions)
                + 2 * (metadata_value(row["source"]) not in selected_sources)
            ),
            axis=1,
        )
        chosen_idx = scored.sort_values(
            ["_novelty_score", "_split_key", "front_id"],
            ascending=[False, True, True],
        ).index[0]
        selected_indices.append(int(chosen_idx))
        remaining = remaining.drop(index=chosen_idx)

    return {int(family_df.loc[index, "front_id"]) for index in selected_indices}


def assign_candidate_split(fronts_df: pd.DataFrame, test_frac: float, seed: int) -> pd.DataFrame:
    if not 0.0 < test_frac < 1.0:
        raise ValueError("--test-frac must be between 0 and 1.")

    quotas = family_test_quotas(fronts_df, test_frac)
    split_df = fronts_df.copy()
    split_df["split"] = "development"

    for family, family_df in split_df.groupby("family", dropna=False, sort=True):
        test_front_ids = select_diverse_test_fronts(family_df, quotas[str(family)], seed)
        split_df.loc[split_df["front_id"].isin(test_front_ids), "split"] = "test"

    return split_df.sort_values("front_id").reset_index(drop=True)


def build_split_candidate_summary(split_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        split_df.groupby(["family", "split"], dropna=False)
        .agg(
            n_fronts=("front_id", "count"),
            n_items=("n_items", "sum"),
            min_size=("size", "min"),
            max_size=("size", "max"),
            mean_target_range=("target_range", "mean"),
        )
        .reset_index()
    )
    totals = (
        split_df.groupby("split")
        .agg(
            n_fronts=("front_id", "count"),
            n_items=("n_items", "sum"),
            min_size=("size", "min"),
            max_size=("size", "max"),
            mean_target_range=("target_range", "mean"),
        )
        .reset_index()
    )
    totals.insert(0, "family", "__TOTAL__")
    return pd.concat([totals, grouped], ignore_index=True)


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def inspect_case(case: str, *, training_csv: Path | None, out_dir: Path | None, test_frac: float, seed: int) -> Path:
    csv_path = resolve_training_csv(case, training_csv)
    output_dir = out_dir if out_dir is not None else EXPERIMENTS_DIR / case / "data" / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    required = CONTROL_COLUMNS | {TARGET_COLUMN}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing required column(s): {', '.join(missing)}")

    front_map = build_front_id_map(case)
    fronts_df = build_fronts_summary(df, front_map)
    columns_df = build_columns_summary(df)
    split_df = assign_candidate_split(fronts_df, test_frac=test_frac, seed=seed)
    split_summary_df = build_split_candidate_summary(split_df)

    fronts_path = output_dir / "fronts_summary.csv"
    columns_path = output_dir / "columns_summary.csv"
    split_assignments_path = output_dir / "split_candidate_assignments.csv"
    split_summary_path = output_dir / "split_candidate_summary.csv"
    case_summary_path = output_dir / "case_summary.json"

    fronts_df.to_csv(fronts_path, index=False)
    columns_df.to_csv(columns_path, index=False)
    split_df.to_csv(split_assignments_path, index=False)
    split_summary_df.to_csv(split_summary_path, index=False)

    block_counts = columns_df["block"].value_counts().sort_index().to_dict()
    split_counts = split_df["split"].value_counts().sort_index().to_dict()
    family_counts = fronts_df["family"].value_counts().sort_index().to_dict()

    summary = {
        "case": case,
        "training_csv": str(csv_path),
        "output_dir": str(output_dir),
        "git_commit": git_commit(),
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "n_fronts": int(df["front_id"].nunique()),
        "rows_per_front": {
            "min": int(fronts_df["n_items"].min()),
            "max": int(fronts_df["n_items"].max()),
            "mean": float(fronts_df["n_items"].mean()),
        },
        "column_blocks": {str(key): int(value) for key, value in block_counts.items()},
        "front_families": {str(key): int(value) for key, value in family_counts.items()},
        "candidate_split": {
            "seed": seed,
            "test_frac": test_frac,
            "counts": {str(key): int(value) for key, value in split_counts.items()},
        },
        "outputs": {
            "fronts_summary": str(fronts_path),
            "columns_summary": str(columns_path),
            "split_candidate_assignments": str(split_assignments_path),
            "split_candidate_summary": str(split_summary_path),
        },
    }
    write_json(case_summary_path, summary)

    print(f"[{case}] wrote audit to {output_dir}")
    print(f"  rows={summary['n_rows']} columns={summary['n_columns']} fronts={summary['n_fronts']}")
    print(f"  split={summary['candidate_split']['counts']}")
    return output_dir


def main() -> None:
    args = build_parser().parse_args()
    if args.training_csv is not None and len(args.cases) != 1:
        raise SystemExit("--training-csv can only be used with one case.")
    if args.out_dir is not None and len(args.cases) != 1:
        raise SystemExit("--out-dir can only be used with one case.")

    for case in args.cases:
        inspect_case(
            case,
            training_csv=args.training_csv,
            out_dir=args.out_dir,
            test_frac=args.test_frac,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
