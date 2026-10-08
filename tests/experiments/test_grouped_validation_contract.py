"""Portable grouped-contract checks; no estimator is fitted."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import csv
import hashlib
import shutil
import json
from pathlib import Path

import pytest

from scripts.experiments.amiga_exp.grouped_validation import contract as MODULE

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def fixture_repo(tmp_path_factory):
    """Construct checksummed datasets with nonnumeric, deliberately unread labels."""
    root = tmp_path_factory.mktemp("grouped-contract-repo")
    sources = [MODULE.GROUP_METADATA_PATH, MODULE.CONTRACT_SOURCE_PATH, MODULE.LABEL_SOURCE_PATH]
    for case in MODULE.CASE_LABEL_MODES:
        sources.extend([
            f"docs/experiments/splits/{case}_split_manifest.json",
            f"docs/experiments/contracts/{case}_feature_columns.json",
            f"docs/experiments/contracts/{case}_data_manifest.json",
        ])
    for source in sources:
        destination = root / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / source, destination)
    metadata = json.loads((root / MODULE.GROUP_METADATA_PATH).read_text())
    for case in MODULE.CASE_LABEL_MODES:
        feature_path = root / f"docs/experiments/contracts/{case}_feature_columns.json"
        features = json.loads(feature_path.read_text())["feature_sets"]["full"]
        data_path = root / f"experiments/{case}/data/data_104.csv"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        with data_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["front_id", "item_id", "AUPR", *features])
            for front in metadata["fronts"]:
                for item in [1, 2]:
                    writer.writerow([front["front_id"], item, "unused_nonnumeric_label", *([0] * len(features))])
        manifest_path = root / f"docs/experiments/contracts/{case}_data_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["data_csv_sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
        manifest["n_rows"] = 208
        manifest_path.write_text(json.dumps(manifest))
    return root


@pytest.fixture(scope="module")
def contract(fixture_repo):
    return MODULE.build_contract(fixture_repo)


def test_contract_is_deterministic_and_pilot_only(contract, fixture_repo):
    assert contract == MODULE.build_contract(fixture_repo)
    assert contract["status"] == "pilot_ready_not_full_eval_frozen"
    result = MODULE.validate_contract(contract)
    assert result["passed"] is True
    assert (result["n_fronts"], result["n_topologies"]) == (104, 87)
    assert contract["cases"]["BIO-INSIGHT"]["label_mode"] == "rank_dense"
    assert contract["cases"]["MO-GENECI"]["label_mode"] == "rank_avg"
    assert len(contract["cases"]["BIO-INSIGHT"]["feature_columns"]) == 104
    assert len(contract["cases"]["MO-GENECI"]["feature_columns"]) == 101


def test_outer_and_inner_partitions_cover_groups_without_leakage(contract):
    mapping = contract["topology_by_front"]
    every_front = {int(value) for value in mapping}
    every_topology = set(mapping.values())
    front_test_counts = Counter()
    topology_test_counts = Counter()
    for outer in contract["outer_folds"]:
        training = set(outer["train_front_ids"])
        testing = set(outer["test_front_ids"])
        assert training.isdisjoint(testing)
        assert training | testing == every_front
        assert {mapping[str(front)] for front in training}.isdisjoint({mapping[str(front)] for front in testing})
        front_test_counts.update(testing)
        topology_test_counts.update(outer["test_topology_ids"])
        inner_validation_counts = Counter()
        for inner in outer["inner_folds"]:
            train = set(inner["train_front_ids"])
            valid = set(inner["validation_front_ids"])
            assert train.isdisjoint(valid)
            assert train | valid == training
            assert (train | valid).isdisjoint(testing)
            assert {mapping[str(front)] for front in train}.isdisjoint({mapping[str(front)] for front in valid})
            inner_validation_counts.update(valid)
        assert inner_validation_counts == Counter({front: 1 for front in training})
    assert front_test_counts == Counter({front: 1 for front in every_front})
    assert topology_test_counts == Counter({topology: 1 for topology in every_topology})
    # The test actually exercises groups with multiple conditions.
    assert sorted(Counter(mapping.values()).values()).count(3) == 8
    assert sorted(Counter(mapping.values()).values()).count(2) == 1


def test_learning_subsets_are_nested_whole_training_groups(contract):
    mapping = contract["topology_by_front"]
    for outer in contract["outer_folds"]:
        subsets = outer["learning_subsets"]
        assert len(subsets) == 10
        for seed in [1301, 1302, 1303]:
            seeded = sorted((row for row in subsets if row["seed"] == seed), key=lambda row: row["size"])
            assert [row["size"] for row in seeded] == [10, 20, 40]
            previous = set()
            for row in seeded:
                group = set(row["topology_ids"])
                assert len(group) == row["size"]
                assert previous < group <= set(outer["train_topology_ids"])
                assert set(row["front_ids"]) == {int(front) for front, topology in mapping.items() if topology in group}
                previous = group
        full = [row for row in subsets if row["seed"] is None]
        assert len(full) == 1
        assert full[0]["topology_ids"] == outer["train_topology_ids"]
        assert full[0]["front_ids"] == outer["train_front_ids"]
    pilot = contract["pilot"]
    inner = contract["outer_folds"][0]["inner_folds"][0]
    assert pilot["train_front_ids"] == inner["train_front_ids"]
    assert set(pilot["train_front_ids"]).isdisjoint(inner["validation_front_ids"])


@pytest.mark.parametrize("mutation", ["external_in_inner", "partial_topology", "label_feature", "seed", "pilot_external", "learning_external", "id_alignment", "training_policy", "metric_policy"])
def test_rejects_mutated_contracts(contract, mutation):
    bad = deepcopy(contract)
    outer = bad["outer_folds"][0]
    if mutation == "external_in_inner":
        outer["inner_folds"][0]["train_topology_ids"].append(outer["test_topology_ids"][0])
    elif mutation == "partial_topology":
        outer["train_front_ids"].pop()
    elif mutation == "label_feature":
        bad["cases"]["BIO-INSIGHT"]["feature_columns"][0] = "AUPR"
    elif mutation == "seed":
        bad["seeds"]["outer_split"] += 1
    elif mutation == "pilot_external":
        bad["pilot"]["train_front_ids"].append(outer["test_front_ids"][0])
    elif mutation == "learning_external":
        outer["learning_subsets"][0]["topology_ids"][0] = outer["test_topology_ids"][0]
    elif mutation == "id_alignment":
        bad["front_name_by_id"].pop(next(iter(bad["front_name_by_id"])))
    elif mutation == "training_policy":
        bad["training_policy"]["early_stopping"] = True
    elif mutation == "metric_policy":
        bad["metric_policy"]["hit_optimality"]["atol"] = 1e-3
    with pytest.raises(ValueError):
        MODULE.validate_contract(bad)


def test_write_is_idempotent_and_rejects_different_content(contract, tmp_path):
    path = tmp_path / "contract.json"
    assert MODULE.write_contract(path, contract) == path
    original = path.read_bytes()
    assert MODULE.write_contract(path, deepcopy(contract)) == path
    assert path.read_bytes() == original
    different = deepcopy(contract)
    different["source_hashes"][MODULE.CONTRACT_SOURCE_PATH] = "0" * 64
    with pytest.raises(ValueError, match="overwrite different"):
        MODULE.write_contract(path, different)
    assert path.read_bytes() == original
    assert json.loads(path.read_text()) == contract


def test_build_rejects_source_hash_and_case_name_mismatches(monkeypatch, fixture_repo):
    original_reader = MODULE._json

    def mismatched_data_hash(path):
        result = original_reader(path)
        if path.name == "BIO-INSIGHT_data_manifest.json":
            result["data_csv_sha256"] = "0" * 64
        return result

    monkeypatch.setattr(MODULE, "_json", mismatched_data_hash)
    with pytest.raises(ValueError, match="data hash"):
        MODULE.build_contract(fixture_repo)

    def mismatched_case_names(path):
        result = original_reader(path)
        if path.name == "MO-GENECI_split_manifest.json":
            result["assignments"][0]["front_name"] = "wrong_network"
        return result

    monkeypatch.setattr(MODULE, "_json", mismatched_case_names)
    with pytest.raises(ValueError, match="front IDs/names"):
        MODULE.build_contract(fixture_repo)


def test_contract_paths_are_clone_portable_and_metadata_is_neutral(contract, fixture_repo, tmp_path):
    cloned = tmp_path / "another_checkout"
    shutil.copytree(fixture_repo, cloned)
    assert MODULE.build_contract(cloned) == contract
    assert "repo_root" not in contract
    for case, data in contract["cases"].items():
        assert data["data_path"] == f"experiments/{case}/data/data_104.csv"
        assert not Path(data["data_path"]).is_absolute()
    metadata_text = (fixture_repo / MODULE.GROUP_METADATA_PATH).read_text()
    for forbidden in ["/home/", "/Users/", "/tmp/"]:
        assert forbidden not in metadata_text
        assert forbidden not in json.dumps(contract)


def test_contract_cli_build_and_validation(fixture_repo, capsys):
    relative_output = "experiments/grouped-validation/contract.json"
    assert MODULE.main(["--repo-root", str(fixture_repo), "--output", relative_output]) == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True
    assert MODULE.main(["--repo-root", str(fixture_repo), "--validate", relative_output]) == 0
    assert json.loads(capsys.readouterr().out)["n_topologies"] == 87


@pytest.mark.skipif(
    not all((REPO_ROOT / f"experiments/{case}/data/data_104.csv").exists() for case in MODULE.CASE_LABEL_MODES),
    reason="optional local benchmark datasets are unavailable",
)
def test_optional_real_dataset_contract():
    result = MODULE.validate_contract(MODULE.build_contract(REPO_ROOT))
    assert result["passed"] is True
    assert (result["n_fronts"], result["n_topologies"]) == (104, 87)
