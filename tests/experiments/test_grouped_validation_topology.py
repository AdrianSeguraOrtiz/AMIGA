"""Small source-collection fixtures for canonical topology verification."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.experiments.amiga_exp.grouped_validation.topology import (
    canonical_topology,
    main,
    verify_topology_metadata,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def source_collection(tmp_path):
    root = tmp_path / "input_data"
    root.mkdir()
    expression = root / "expression.csv"
    expression.write_text(",sample1,sample2\nA,1,2\nB,3,4\nC,5,6\n")
    first = root / "first.csv"
    first.write_text(",A,B,C\nA,0,2,0\nB,0,0,0\nC,0,0,0\n")
    second = root / "reordered.csv"
    second.write_text(",C,B,A\nC,0,0,0\nB,0,0,0\nA,0,-7,0\n")
    payload = json.dumps([["A", "B", "C"], [["A", "B"]]], ensure_ascii=False, separators=(",", ":")).encode()
    expected_topology = hashlib.sha256(payload).hexdigest()
    metadata = {
        "schema_version": 1,
        "source_collection": {"relative_root": "input_data"},
        "counts": {"fronts": 2, "topology_groups": 1, "source_gold_standard_files": 2},
        "fronts": [
            {
                "front_id": index,
                "expression": {"source_path": "input_data/expression.csv", "sha256": _sha(expression)},
                "gold_standard": {"source_path": f"input_data/{gold.name}", "sha256": _sha(gold)},
                "topology_sha256": expected_topology,
                "n_nodes": 3,
                "n_edges": 1,
            }
            for index, gold in enumerate([first, second], start=1)
        ],
    }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata))
    return root, metadata_path, metadata


def test_canonical_hash_preserves_labels_direction_and_isolates(source_collection):
    root, _, metadata = source_collection
    expected = metadata["fronts"][0]["topology_sha256"]
    assert canonical_topology(root / "first.csv") == (expected, 3, 1)
    assert canonical_topology(root / "reordered.csv") == (expected, 3, 1)
    # Reversing the directed edge changes the topology despite identical counts.
    reverse = root / "reverse.csv"
    reverse.write_text(",A,B,C\nA,0,0,0\nB,2,0,0\nC,0,0,0\n")
    assert canonical_topology(reverse)[0] != expected


def test_verifies_file_provenance_and_canonical_groups_without_writing(source_collection):
    root, path, _ = source_collection
    before = {file: file.read_bytes() for file in [path, *root.iterdir()]}
    result = verify_topology_metadata(root, path)
    assert result == {"passed": True, "n_fronts": 2, "n_topologies": 1, "expression_files": 1, "gold_standard_files": 2}
    assert before == {file: file.read_bytes() for file in before}


@pytest.mark.parametrize("mutation", ["expression_bytes", "gold_bytes", "topology_hash", "duplicate_id", "traversal", "absolute_path"])
def test_rejects_mutated_sources_or_metadata(source_collection, mutation):
    root, path, original = source_collection
    metadata = deepcopy(original)
    if mutation == "expression_bytes":
        (root / "expression.csv").write_text("different bytes")
    elif mutation == "gold_bytes":
        (root / "first.csv").write_text("different bytes")
    elif mutation == "topology_hash":
        metadata["fronts"][0]["topology_sha256"] = "0" * 64
    elif mutation == "duplicate_id":
        metadata["fronts"][1]["front_id"] = 1
    elif mutation == "traversal":
        metadata["fronts"][0]["expression"]["source_path"] = "input_data/../outside.csv"
    elif mutation == "absolute_path":
        metadata["fronts"][0]["expression"]["source_path"] = str(root / "expression.csv")
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        verify_topology_metadata(root, path)


def test_topology_cli_reports_counts(source_collection, capsys):
    root, path, _ = source_collection
    assert main(["--source-root", str(root), "--metadata", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["n_topologies"] == 1


def test_malformed_matrix_labels_are_rejected(tmp_path):
    matrix = tmp_path / "bad.csv"
    matrix.write_text(",A,B\nA,0,1\nA,0,0\n")
    with pytest.raises(ValueError, match="unique and equal"):
        canonical_topology(matrix)


@pytest.mark.parametrize("value", ["NaN", "inf", "-inf"])
def test_nonfinite_adjacency_values_are_rejected(tmp_path, value):
    matrix = tmp_path / "nonfinite.csv"
    matrix.write_text(f",A,B\nA,0,{value}\nB,0,0\n")
    with pytest.raises(ValueError, match="must be finite"):
        canonical_topology(matrix)
