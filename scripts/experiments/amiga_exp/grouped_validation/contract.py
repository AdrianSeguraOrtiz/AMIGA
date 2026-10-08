"""Deterministic contracts for topology-grouped benchmark validation.

Only public metadata, CSV identifiers/headers and file hashes are inspected.
AUPR values are not loaded or evaluated. Topology controls splitting while
front_id remains the ranking query. Runtime contracts use repository-relative
paths and are written separately from versioned source metadata.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

CASE_LABEL_MODES = {"BIO-INSIGHT": "rank_dense", "MO-GENECI": "rank_avg"}
ARM_IDS = ["ltr_catboost", "reg_aupr", "reg_normalized", "clf_top20"]
OUTER_SEED = 20260910
INNER_SEEDS = list(range(20260911, 20260916))
LEARNING_SEEDS = [1301, 1302, 1303]
FINAL_SEEDS = [1201, 1202, 1203, 1204, 1205]
STATUS = "pilot_ready_not_full_eval_frozen"
GROUP_METADATA_PATH = "docs/experiments/groups/topology_groups.json"
CONTRACT_SOURCE_PATH = "scripts/experiments/amiga_exp/grouped_validation/contract.py"
LABEL_SOURCE_PATH = "amiga/selection/learn2rank.py"
FORBIDDEN_FEATURES = {"front_id", "item_id", "topology_id", "AUPR", "AUROC", "Accuracy Mean", "score", "rank", "label", "target"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _source_hash_paths() -> set[str]:
    paths = {GROUP_METADATA_PATH, CONTRACT_SOURCE_PATH, LABEL_SOURCE_PATH}
    for case in CASE_LABEL_MODES:
        paths.update({
            f"docs/experiments/splits/{case}_split_manifest.json",
            f"docs/experiments/contracts/{case}_feature_columns.json",
            f"docs/experiments/contracts/{case}_data_manifest.json",
            f"experiments/{case}/data/data_104.csv",
        })
    return paths


def _validate_group_metadata(metadata: dict[str, Any]) -> None:
    _require(metadata.get("schema_version") == 1, "unsupported topology metadata schema")
    _require(metadata.get("cases") == list(CASE_LABEL_MODES), "topology metadata must describe both benchmark cases")
    rows = metadata.get("fronts", [])
    _require(len(rows) == 104, "topology metadata must contain 104 fronts")
    _require(len({row["front_id"] for row in rows}) == 104, "topology metadata front IDs must be unique")
    _require(len({row["front_name"] for row in rows}) == 104, "topology metadata front names must be unique")
    groups = {row["topology_sha256"] for row in rows}
    _require(len(groups) == 87 and all(_valid_sha256(group) for group in groups), "topology metadata must contain 87 valid groups")
    _require(metadata.get("counts", {}).get("fronts") == 104 and metadata.get("counts", {}).get("topology_groups") == 87, "topology metadata counts disagree")
    for row in rows:
        _require(isinstance(row["front_id"], int), "topology metadata front IDs must be integers")
        for field in ["family", "source", "condition"]:
            _require(isinstance(row.get(field), str) and bool(row[field]), f"missing topology provenance: {field}")
        for source in ["expression", "gold_standard"]:
            reference = row.get(source, {})
            path = reference.get("source_path", "")
            relative = PurePosixPath(path)
            _require(bool(path) and not relative.is_absolute() and ".." not in relative.parts and "\\" not in path and ":" not in path, "source collection paths must be portable and relative")
            _require(_valid_sha256(reference.get("sha256")), f"invalid {source} provenance hash")


def _partitions(topologies: list[str], seed: int, n_splits: int) -> list[list[str]]:
    ordered = np.asarray(sorted(topologies), dtype=str)
    shuffled = np.random.default_rng(seed).permutation(ordered)
    return [sorted(chunk.tolist()) for chunk in np.array_split(shuffled, n_splits)]


def _front_ids(mapping: dict[str, str], topologies: list[str]) -> list[int]:
    allowed = set(topologies)
    return sorted(int(front) for front, topology in mapping.items() if topology in allowed)


def _grid() -> list[dict[str, Any]]:
    return [
        {
            "id": f"d{depth}_lr{str(rate).replace('.', '')}",
            "depth": depth,
            "learning_rate": rate,
            "l2_leaf_reg": 3,
            "iterations": 3000,
            "use_best_model": False,
        }
        for depth in [4, 6, 8]
        for rate in [0.03, 0.1]
    ]


def _training_policy() -> dict[str, Any]:
    return {
        "same_features_and_training_fronts_for_all_arms": True,
        "flat_fronts": {
            "definition": "all training AUPR values exactly equal within front",
            "excluded_from_training": list(ARM_IDS),
            "retained_in_evaluation": True,
            "detected_using_training_labels_only": True,
        },
        "weights": {
            "regression_and_classification": {
                "raw_sample_weight": "1 / (usable_fronts_per_topology * candidates_per_front)",
                "normalization": "mean_one_over_training_rows",
            },
            "ranking": {
                "group_weight": "1 / usable_fronts_per_topology",
                "constant_within_front": True,
                "individual_sample_weights": False,
                "additional_normalization": False,
            },
        },
        "labels": {
            "ltr_catboost": "fixed per-case within-front labeling mode; YetiRank",
            "reg_aupr": "untransformed training AUPR; RMSE",
            "reg_normalized": "within-training-front min-max AUPR; RMSE; no inverse transformation or prediction clipping",
            "clf_top20": {
                "loss": "Logloss",
                "threshold": "AUPR at descending position ceil(0.20 * n_candidates)",
                "include_threshold_ties": True,
                "if_threshold_is_minimum_on_nonflat_front": "positive iff AUPR strictly exceeds minimum",
                "ranking_score": "positive-class probability, without a 0.5 cutoff",
            },
        },
        "early_stopping": False,
        "use_best_model": False,
    }


def _metric_policy() -> dict[str, Any]:
    return {
        "primary": "Regret@5",
        "secondary": "Regret@1",
        "ks": [1, 3, 5, 10],
        "effective_k": "min(k, n_candidates_in_front)",
        "predicted_score_ties": "exact_equal_scores; uniform ordering within tied score blocks",
        "top_k_tie_evaluation": "expected BestAUPR@k and Hit@k, not accidental input row order",
        "hit_optimality": {"comparison": "numpy.isclose(AUPR, front_max_AUPR)", "rtol": 1e-5, "atol": 1e-8},
        "aggregation_order": ["mean_metrics_across_final_seeds_within_front", "mean_front_metrics_within_topology", "mean_across_topologies"],
        "front_macro_mean_is_sensitivity_only": True,
        "configuration_selection": {
            "primary": "minimum topology-macro Regret@5 over inner validation predictions",
            "numeric_tie_decimals": 12,
            "tie_breakers": ["minimum topology-macro Regret@1", "stable configuration id"],
            "p_values_are_selection_gate": False,
        },
    }


def _learning_subsets(mapping: dict[str, str], training: list[str]) -> list[dict[str, Any]]:
    result = []
    for seed in LEARNING_SEEDS:
        ordered = np.random.default_rng(seed).permutation(np.asarray(sorted(training), dtype=str)).tolist()
        for size in [10, 20, 40]:
            selected = sorted(ordered[:size])
            result.append({"seed": seed, "size": size, "topology_ids": selected, "front_ids": _front_ids(mapping, selected)})
    # Full data are represented once: there is no sub-sampling seed for this row.
    result.append({"seed": None, "size": len(training), "topology_ids": sorted(training), "front_ids": _front_ids(mapping, training)})
    return result


def build_contract(repo_root: Path) -> dict[str, Any]:
    """Verify current source identity and create a deterministic pilot contract."""
    repo_root = Path(repo_root).resolve()
    audit_path = repo_root / GROUP_METADATA_PATH
    audit = _json(audit_path)
    _validate_group_metadata(audit)
    audit_fronts = audit["fronts"]
    _require(len(audit_fronts) == 104, "topology audit must map 104 fronts")
    topology_by_front = {str(row["front_id"]): row["topology_sha256"] for row in audit_fronts}
    names = {str(row["front_id"]): row["front_name"] for row in audit_fronts}
    _require(len(topology_by_front) == 104 and len(set(names.values())) == 104, "audit front IDs/names must be unique")
    _require(len(set(topology_by_front.values())) == 87, "topology audit must contain 87 groups")
    source_paths = [audit_path, repo_root / CONTRACT_SOURCE_PATH, repo_root / LABEL_SOURCE_PATH]
    cases: dict[str, Any] = {}
    for case, label_mode in CASE_LABEL_MODES.items():
        split_path = repo_root / f"docs/experiments/splits/{case}_split_manifest.json"
        features_path = repo_root / f"docs/experiments/contracts/{case}_feature_columns.json"
        data_manifest_path = repo_root / f"docs/experiments/contracts/{case}_data_manifest.json"
        split = _json(split_path)
        features = _json(features_path)
        data_manifest = _json(data_manifest_path)
        assigned = {str(row["front_id"]): row["front_name"] for row in split["assignments"]}
        _require(len(split["assignments"]) == 104 and assigned == names, f"{case}: front IDs/names differ from audit")
        data_path = repo_root / f"experiments/{case}/data/data_104.csv"
        digest = _sha256(data_path)
        _require(digest == data_manifest["data_csv_sha256"], f"{case}: data hash differs from public data manifest")
        # Only identifiers are selected; no AUPR values are loaded.
        front_column = pd.read_csv(data_path, usecols=["front_id"], dtype={"front_id": "int64"})
        _require(set(map(str, front_column["front_id"].unique())) == set(names), f"{case}: dataset front IDs differ from audit")
        _require(len(front_column) == data_manifest["n_rows"], f"{case}: data row count differs from manifest")
        _require(data_manifest["n_fronts"] == 104, f"{case}: data manifest front count differs")
        with data_path.open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        feature_columns = list(features["feature_sets"]["full"])
        _require(len(header) == len(set(header)), f"{case}: duplicate CSV column names")
        _require(set(feature_columns).issubset(header), f"{case}: predictor absent from CSV header")
        cases[case] = {
            "data_path": data_path.relative_to(repo_root).as_posix(),
            "sha256": digest,
            "feature_columns": feature_columns,
            "feature_contract_sha256": _sha256(features_path),
            "label_mode": label_mode,
            "n_rows": len(front_column),
            "n_fronts": 104,
        }
        source_paths.extend([split_path, features_path, data_manifest_path, data_path])
    all_topologies = sorted(set(topology_by_front.values()))
    outer_folds = []
    for outer_index, test_topologies in enumerate(_partitions(all_topologies, OUTER_SEED, 5)):
        train_topologies = sorted(set(all_topologies) - set(test_topologies))
        inner_folds = []
        for inner_index, validation in enumerate(_partitions(train_topologies, INNER_SEEDS[outer_index], 3)):
            training = sorted(set(train_topologies) - set(validation))
            inner_folds.append({
                "fold": inner_index,
                "train_front_ids": _front_ids(topology_by_front, training),
                "validation_front_ids": _front_ids(topology_by_front, validation),
                "train_topology_ids": training,
                "validation_topology_ids": validation,
            })
        outer_folds.append({
            "fold": outer_index,
            "train_front_ids": _front_ids(topology_by_front, train_topologies),
            "test_front_ids": _front_ids(topology_by_front, test_topologies),
            "train_topology_ids": train_topologies,
            "test_topology_ids": test_topologies,
            "inner_folds": inner_folds,
            "learning_subsets": _learning_subsets(topology_by_front, train_topologies),
        })
    pilot_train = outer_folds[0]["inner_folds"][0]
    contract = {
        "schema_version": 1,
        "status": STATUS,
        "evaluation_type": "internal_nested_cross_validation_by_labelled_topology",
        "ranking_group_column": "front_id",
        "split_group_column": "topology_id",
        "target_column": "AUPR",
        "cases": cases,
        "topology_by_front": dict(sorted(topology_by_front.items(), key=lambda item: int(item[0]))),
        "front_name_by_id": dict(sorted(names.items(), key=lambda item: int(item[0]))),
        "source_hashes": {path.relative_to(repo_root).as_posix(): _sha256(path) for path in source_paths},
        "arms": list(ARM_IDS),
        "grid": _grid(),
        "training_policy": _training_policy(),
        "metric_policy": _metric_policy(),
        "seeds": {"outer_split": OUTER_SEED, "inner_splits": list(INNER_SEEDS), "tuning": 1101, "final": list(FINAL_SEEDS), "learning_subsets": list(LEARNING_SEEDS), "bootstrap": 1401, "deployment_split": 20260916},
        "outer_folds": outer_folds,
        "pilot": {
            "outer_fold": 0,
            "inner_fold": 0,
            "data_scope": "outer_0_inner_0_training_only",
            "train_front_ids": list(pilot_train["train_front_ids"]),
            "train_topology_ids": list(pilot_train["train_topology_ids"]),
            "quality_metrics_for_protocol_selection": False,
        },
    }
    validate_contract(contract)
    return contract


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Check structural isolation and the fixed design without reading any files."""
    _require(contract.get("schema_version") == 1, "unsupported contract schema")
    _require(contract.get("status") == STATUS, "contract is only authorized for pilot readiness")
    _require(contract.get("ranking_group_column") == "front_id" and contract.get("split_group_column") == "topology_id", "query and split group roles must remain distinct")
    _require(contract.get("target_column") == "AUPR", "unexpected target")
    mapping = contract.get("topology_by_front", {})
    names = contract.get("front_name_by_id", {})
    _require(len(mapping) == 104 and set(mapping) == set(names) and len(set(names.values())) == 104, "front ID/name coverage must be 104 unique pairs")
    _require(all(str(int(front)) == front for front in mapping), "front keys must be canonical integer strings")
    all_topologies = sorted(set(mapping.values()))
    _require(len(all_topologies) == 87, "expected 87 topology groups")
    _require(all(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value) for value in all_topologies), "invalid topology hashes")
    cases = contract.get("cases", {})
    _require(set(cases) == set(CASE_LABEL_MODES), "both cases are required")
    for case, mode in CASE_LABEL_MODES.items():
        data = cases[case]
        columns = data.get("feature_columns", [])
        _require(data.get("label_mode") == mode, f"{case}: unexpected ranker label mode")
        _require(len(columns) == (104 if case == "BIO-INSIGHT" else 101) and len(columns) == len(set(columns)), f"{case}: incorrect or duplicate feature columns")
        _require(not set(columns) & FORBIDDEN_FEATURES, f"{case}: label/control column in predictors")
        _require(all("aupr" not in col.lower() and "target" not in col.lower() for col in columns), f"{case}: target-derived predictor name")
        _require(data.get("data_path") == f"experiments/{case}/data/data_104.csv", f"{case}: data_path must be the repository-relative dataset path")
        _require(data.get("n_fronts") == 104, f"{case}: inconsistent front count")
        for field in ["sha256", "feature_contract_sha256"]:
            value = data.get(field, "")
            _require(len(value) == 64 and all(c in "0123456789abcdef" for c in value), f"{case}: invalid {field}")
    _require(contract.get("arms") == ARM_IDS and contract.get("grid") == _grid(), "arms or shared grid differ from fixed proposal")
    _require(contract.get("training_policy") == _training_policy(), "training policy differs from fixed proposal")
    _require(contract.get("metric_policy") == _metric_policy(), "metric policy differs from fixed proposal")
    seeds = contract.get("seeds", {})
    _require(seeds == {"outer_split": OUTER_SEED, "inner_splits": INNER_SEEDS, "tuning": 1101, "final": FINAL_SEEDS, "learning_subsets": LEARNING_SEEDS, "bootstrap": 1401, "deployment_split": 20260916}, "seed roles differ from fixed proposal")
    folds = contract.get("outer_folds", [])
    _require(len(folds) == 5, "five outer folds required")
    seen_test: list[str] = []
    outer_counts = []
    for index, (fold, expected_test) in enumerate(zip(folds, _partitions(all_topologies, OUTER_SEED, 5), strict=True)):
        _require(fold.get("fold") == index, "outer fold indexes must be consecutive from zero")
        train = fold.get("train_topology_ids", [])
        test = fold.get("test_topology_ids", [])
        _require(test == expected_test and train == sorted(set(all_topologies) - set(test)), "outer topology split differs or overlaps")
        _require(len(train) in [69, 70] and len(test) in [17, 18], "unexpected outer topology counts")
        _require(fold.get("train_front_ids") == _front_ids(mapping, train) and fold.get("test_front_ids") == _front_ids(mapping, test), "outer front IDs do not match whole topology groups")
        inner = fold.get("inner_folds", [])
        _require(len(inner) == 3, "three inner folds required")
        for j, (part, expected_validation) in enumerate(zip(inner, _partitions(train, INNER_SEEDS[index], 3), strict=True)):
            _require(part.get("fold") == j, "inner fold indexes must be consecutive from zero")
            val = part.get("validation_topology_ids", [])
            tr = part.get("train_topology_ids", [])
            _require(val == expected_validation and tr == sorted(set(train) - set(val)), "inner topology split differs or includes external groups")
            _require(part.get("train_front_ids") == _front_ids(mapping, tr) and part.get("validation_front_ids") == _front_ids(mapping, val), "inner front IDs do not match whole topology groups")
        _require(fold.get("learning_subsets") == _learning_subsets(mapping, train), "learning subsets must be nested, complete groups, and training-only")
        seen_test.extend(test)
        outer_counts.append({"fold": index, "train_fronts": len(fold["train_front_ids"]), "test_fronts": len(fold["test_front_ids"]), "train_topologies": len(train), "test_topologies": len(test)})
    _require(sorted(seen_test) == all_topologies, "outer test folds must partition every topology exactly once")
    pilot = contract.get("pilot", {})
    reference = folds[0]["inner_folds"][0]
    _require(pilot.get("outer_fold") == 0 and pilot.get("inner_fold") == 0 and pilot.get("data_scope") == "outer_0_inner_0_training_only", "pilot scope differs")
    _require(pilot.get("train_front_ids") == reference["train_front_ids"] and pilot.get("train_topology_ids") == reference["train_topology_ids"], "pilot must use only designated inner training data")
    _require(pilot.get("quality_metrics_for_protocol_selection") is False, "pilot cannot select protocol using quality")
    source_hashes = contract.get("source_hashes", {})
    _require(set(source_hashes) == _source_hash_paths(), "source hashes must identify only the public experimental inputs and code")
    _require(all(_valid_sha256(value) for value in source_hashes.values()), "invalid source hash")
    _require("repo_root" not in contract, "contract must not contain a machine-specific repository root")
    for case, data in cases.items():
        _require(source_hashes.get(f"experiments/{case}/data/data_104.csv") == data["sha256"], "dataset source hash inconsistent")
        _require(source_hashes.get(f"docs/experiments/contracts/{case}_feature_columns.json") == data["feature_contract_sha256"], "feature source hash inconsistent")
    return {
        "passed": True,
        "n_fronts": 104,
        "n_topologies": 87,
        "n_cases": 2,
        "n_outer_folds": 5,
        "n_inner_folds_total": 15,
        "n_learning_subsets_total": 50,
        "n_grid_configurations": 6,
        "n_arms": 4,
        "outer_counts": outer_counts,
        "pilot_train_fronts": len(pilot["train_front_ids"]),
        "pilot_train_topologies": len(pilot["train_topology_ids"]),
        "checks": {"group_isolation": True, "complete_coverage": True, "front_alignment": True, "nested_learning_subsets": True, "seed_partition_determinism": True, "pilot_training_only": True, "predictor_roles": True},
    }


def write_contract(path: Path, contract: dict[str, Any]) -> Path:
    """Write a validated contract; refuse replacement by different content."""
    validate_contract(contract)
    path = Path(path)
    serialized = json.dumps(contract, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        _require(_json(path) == contract, f"refusing to overwrite different contract: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation guards against accidentally replacing a concurrent write.
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    return path


def main(argv: list[str] | None = None) -> int:
    """Build a contract or validate an existing one without fitting estimators."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[4])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path, help="Build and write a contract; relative paths are resolved from the repository root.")
    action.add_argument("--validate", type=Path, help="Validate an existing contract structurally; does not rehash its source files.")
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    try:
        if args.validate is not None:
            path = args.validate if args.validate.is_absolute() else repo_root / args.validate
            contract = _json(path)
        else:
            contract = build_contract(repo_root)
            path = args.output if args.output.is_absolute() else repo_root / args.output
            write_contract(path, contract)
        print(json.dumps(validate_contract(contract), indent=2, sort_keys=True))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"contract error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
