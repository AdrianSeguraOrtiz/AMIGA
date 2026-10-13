"""Verify public topology metadata against a local source collection, read-only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_topology(path: Path) -> tuple[str, int, int]:
    """Return SHA-256, node count and edge count for a labelled adjacency CSV.

    All nodes, including isolates, and nonzero directed edges are sorted into
    [nodes, edges], serialized as compact UTF-8 JSON. Edge sign is ignored.
    """
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None or len(header) < 2:
            raise ValueError(f"missing adjacency header: {path}")
        columns, nodes, edges = header[1:], [], []
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"invalid adjacency row width: {path}")
            nodes.append(row[0])
            for target, cell in zip(columns, row[1:]):
                value = float(cell)
                if not math.isfinite(value):
                    raise ValueError(f"adjacency values must be finite: {path}")
                if value != 0:
                    edges.append((row[0], target))
    if len(nodes) != len(set(nodes)) or len(columns) != len(set(columns)) or set(nodes) != set(columns):
        raise ValueError(f"adjacency row/column labels must be unique and equal: {path}")
    payload = json.dumps([sorted(nodes), sorted(edges)], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(nodes), len(edges)


def _reference_path(root: Path, source_path: str, prefix: str) -> Path:
    relative, collection = PurePosixPath(source_path), PurePosixPath(prefix)
    if relative.is_absolute() or ".." in relative.parts or "\\" in source_path or ":" in source_path:
        raise ValueError("source paths must be portable and relative")
    if relative.parts[:len(collection.parts)] != collection.parts:
        raise ValueError(f"source path does not start with collection root: {source_path}")
    resolved = (root / Path(*relative.parts[len(collection.parts):])).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"source path escapes collection root: {source_path}")
    return resolved


def verify_topology_metadata(source_root: Path, metadata_path: Path) -> dict[str, Any]:
    """Check source file hashes and canonical groups; never read quality reports.

    source_root is the collection directory corresponding to relative_root in
    metadata, for example the GENECI input_data directory. No files are written.
    """
    root = Path(source_root).resolve()
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError("unsupported topology metadata schema")
    prefix = metadata["source_collection"]["relative_root"]
    rows, hashes, signatures = metadata["fronts"], {}, {}
    ids = [row["front_id"] for row in rows]
    if len(ids) != len(set(ids)) or len(ids) != metadata["counts"]["fronts"]:
        raise ValueError("front IDs must be unique and match metadata count")
    expression_paths, gold_paths, topology_ids = set(), set(), set()
    for row in rows:
        for source in ["expression", "gold_standard"]:
            reference = row[source]
            path = _reference_path(root, reference["source_path"], prefix)
            if path not in hashes:
                hashes[path] = _sha256(path)
            if hashes[path] != reference["sha256"]:
                raise ValueError(f"{source} SHA-256 mismatch for front {row['front_id']}: {path}")
            if source == "expression":
                expression_paths.add(path)
            else:
                gold_paths.add(path)
                if path not in signatures:
                    signatures[path] = canonical_topology(path)
                if signatures[path] != (row["topology_sha256"], row["n_nodes"], row["n_edges"]):
                    raise ValueError(f"canonical topology mismatch for front {row['front_id']}")
        topology_ids.add(row["topology_sha256"])
    if len(topology_ids) != metadata["counts"]["topology_groups"]:
        raise ValueError("topology group count mismatch")
    if len(gold_paths) != metadata["counts"]["source_gold_standard_files"]:
        raise ValueError("gold-standard source file count mismatch")
    return {"passed": True, "n_fronts": len(rows), "n_topologies": len(topology_ids), "expression_files": len(expression_paths), "gold_standard_files": len(gold_paths)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path, help="Local collection root, for example GENECI/input_data.")
    parser.add_argument("--metadata", type=Path, default=Path("docs/experiments/groups/topology_groups.json"))
    args = parser.parse_args(argv)
    try:
        print(json.dumps(verify_topology_metadata(args.source_root, args.metadata), indent=2, sort_keys=True))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"topology verification error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
