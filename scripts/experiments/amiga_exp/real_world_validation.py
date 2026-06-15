"""Reported real-world TCGA-BRCA validation for the AMIGA paper workflow."""

from __future__ import annotations

import io
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

from amiga.utils import row_weights_from_front, weighted_confidence
from scripts.experiments.amiga_exp.decision_baselines import (
    normalized_objective_badness_matrix,
    topsis_scores_from_badness,
)
from scripts.experiments.amiga_exp.manifests import WriteOptions, repo_relative, write_manifest


OBJECTIVE_COLUMNS = (
    "quality",
    "degreedistribution",
    "motifs",
    "reducenonessentialsinteractions",
    "dynamicity",
    "metricdistribution",
)
OBJECTIVE_DIRECTIONS = {name: "minimize" for name in OBJECTIVE_COLUMNS}
REPORTED_SELECTOR_IDS = (
    "AMIGA_top1",
    "objective_reducenonessentialsinteractions",
    "objective_mean_rank",
    "objective_topsis",
    "objective_metricdistribution",
)
SELECTOR_DISPLAY_NAMES = {
    "AMIGA_top1": "AMIGA",
    "objective_reducenonessentialsinteractions": "ReduceNEI",
    "objective_mean_rank": "Mean rank",
    "objective_topsis": "TOPSIS",
    "objective_metricdistribution": "Metric dist.",
}
REPORTED_SOURCES = (
    {
        "label": "CollecTRI",
        "resource": "OmniPath_CollecTRI",
        "cutoff": 250,
        "kind": "external",
    },
    {
        "label": "DoRothEA",
        "resource": "OmniPath_DoRothEA",
        "cutoff": 250,
        "kind": "external",
    },
    {
        "label": "TRRUST v2",
        "resource": "TRRUST_v2",
        "cutoff": 250,
        "kind": "external",
    },
    {
        "label": "JASPAR PWM",
        "resource": "Enrichr_JASPAR_2025",
        "cutoff": 250,
        "kind": "external",
    },
    {
        "label": "Cistrome BRCA-COR",
        "resource": "CistromeCancer_BRCA_COR",
        "cutoff": 5000,
        "kind": "cistrome_cor",
    },
)
OMNIPATH_URL = (
    "https://omnipathdb.org/interactions"
    "?datasets=dorothea,tf_target,collectri"
    "&genesymbols=1"
    "&fields=sources,references,dorothea_level"
    "&format=tsv"
)
TRRUST_URL = "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv"
ENRICHR_LIBRARY_URL = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName={library}"
CISTROME_TARGET_URL = "https://cistrome.org/CistromeCancer/CancerTarget/examples"
CISTROME_BRCA_COLUMNS = ("BRCA_1", "BRCA_2")
JASPAR_LIBRARY = "JASPAR_PWM_Human_2025"


class RealWorldValidationError(ValueError):
    """Raised when the reported real-world validation cannot be generated."""


@dataclass(frozen=True)
class Selector:
    selector_id: str
    method_family: str
    selected_index: int
    selected_item_id: int
    selector_score: float


@dataclass
class ResourceStatus:
    resource: str
    status: str
    url: str
    cache_path: str | None = None
    rows_raw: int = 0
    rows_normalized: int = 0
    error: str | None = None


@dataclass(frozen=True)
class RealWorldValidationResult:
    output_dir: Path
    selected_candidates: Path
    selected_networks_dir: Path
    evidence: Path
    table_csv: Path
    table_md: Path
    manifest: Path
    manifest_status: object


def validate_real_world_case(
    *,
    case_dir: Path,
    ranked_csv: Path | None = None,
    grn_dir: Path | None = None,
    gene_universe_csv: Path | None = None,
    output_dir: Path | None = None,
    cache_dir: Path | None = None,
    timeout: int = 120,
    force_refresh: bool = False,
    options: WriteOptions,
) -> RealWorldValidationResult:
    """Generate only the real-world validation table reported in the paper."""
    case_dir = Path(case_dir)
    ranked_csv = ranked_csv or case_dir / "amiga" / "ranked_real.csv"
    grn_dir = grn_dir or infer_grn_dir(case_dir)
    gene_universe_csv = gene_universe_csv or case_dir / "data" / "gene_universe_top500.csv"
    output_dir = output_dir or case_dir / "validation" / "amiga_exp_reported"
    cache_dir = cache_dir or case_dir / "validation" / "cache"
    networks_dir = output_dir / "selected_networks"

    planned_outputs = {
        "selected_candidates": output_dir / "selected_candidates.csv",
        "evidence": output_dir / "reported_external_tf_target_evidence.csv",
        "table_csv": output_dir / "real_world_source_support_top1.csv",
        "table_md": output_dir / "real_world_source_support_top1.md",
        "manifest": output_dir / "real_world_validation_manifest.json",
        "selected_networks_dir": networks_dir,
    }
    if options.dry_run:
        return RealWorldValidationResult(
            output_dir=output_dir,
            selected_candidates=planned_outputs["selected_candidates"],
            selected_networks_dir=networks_dir,
            evidence=planned_outputs["evidence"],
            table_csv=planned_outputs["table_csv"],
            table_md=planned_outputs["table_md"],
            manifest=planned_outputs["manifest"],
            manifest_status="dry_run",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    networks_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    ranked = pd.read_csv(ranked_csv)
    validate_ranked_frame(ranked)
    gene_universe = load_gene_universe(gene_universe_csv)
    selectors = select_reported_candidates(ranked)
    selected_candidates = write_selected_candidates(
        ranked=ranked,
        selectors=selectors,
        output_path=planned_outputs["selected_candidates"],
    )
    network_paths = reconstruct_selected_networks(
        ranked=ranked,
        selectors=selectors,
        grn_dir=grn_dir,
        networks_dir=networks_dir,
    )
    evidence, statuses = load_reported_evidence(
        cache_dir=cache_dir,
        gene_universe=gene_universe,
        network_paths=network_paths,
        timeout=timeout,
        force_refresh=force_refresh,
    )
    evidence.to_csv(planned_outputs["evidence"], index=False)

    table = build_reported_source_support_table(
        selectors=selectors,
        network_paths=network_paths,
        evidence=evidence,
    )
    table.to_csv(planned_outputs["table_csv"], index=False)
    planned_outputs["table_md"].write_text(build_reported_markdown(table), encoding="utf-8")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "reported_tcga_brca_top1_source_support",
        "case_dir": repo_relative(case_dir),
        "ranked_csv": repo_relative(ranked_csv),
        "grn_dir": repo_relative(grn_dir),
        "gene_universe_csv": repo_relative(gene_universe_csv),
        "selectors": list(REPORTED_SELECTOR_IDS),
        "sources": REPORTED_SOURCES,
        "metrics": ["tf_target_edges", "tf_source_edges"],
        "excluded_by_design": [
            "top5_mean",
            "top5_best",
            "coexpression",
            "tf_activity_proxy",
            "unreported_cutoffs",
        ],
        "selected_candidates": selected_candidates.to_dict(orient="records"),
        "resource_statuses": [asdict(status) for status in statuses],
        "outputs": {name: repo_relative(path) for name, path in planned_outputs.items()},
    }
    manifest_status = write_manifest(planned_outputs["manifest"], manifest, options)

    return RealWorldValidationResult(
        output_dir=output_dir,
        selected_candidates=planned_outputs["selected_candidates"],
        selected_networks_dir=networks_dir,
        evidence=planned_outputs["evidence"],
        table_csv=planned_outputs["table_csv"],
        table_md=planned_outputs["table_md"],
        manifest=planned_outputs["manifest"],
        manifest_status=manifest_status,
    )


def infer_grn_dir(case_dir: Path) -> Path:
    candidates = sorted((case_dir / "bioinsight").glob("*/lists"))
    if len(candidates) != 1:
        raise RealWorldValidationError(
            "could not infer BIO-INSIGHT lists directory; pass --grn-dir explicitly"
        )
    return candidates[0]


def validate_ranked_frame(ranked: pd.DataFrame) -> None:
    required = {"front_id", "item_id", "score", "rank_in_front", *OBJECTIVE_COLUMNS}
    missing = sorted(required - set(ranked.columns))
    if missing:
        raise RealWorldValidationError(f"ranked CSV missing required columns: {missing}")
    if ranked["front_id"].nunique() != 1:
        raise RealWorldValidationError("the reported real-world validation expects exactly one front")
    grn_cols = [column for column in ranked.columns if column.startswith("GRN_") and column.endswith(".csv")]
    if not grn_cols:
        raise RealWorldValidationError("ranked CSV contains no GRN_*.csv weight columns")


def load_gene_universe(path: Path) -> set[str]:
    frame = pd.read_csv(path)
    if "gene_symbol" not in frame.columns:
        raise RealWorldValidationError(f"gene universe CSV lacks 'gene_symbol': {path}")
    genes = {normalize_symbol(value) for value in frame["gene_symbol"].dropna()}
    genes.discard("")
    if not genes:
        raise RealWorldValidationError("gene universe is empty")
    return genes


def select_reported_candidates(ranked: pd.DataFrame) -> list[Selector]:
    score_map = candidate_score_series(ranked)
    missing = [selector_id for selector_id in REPORTED_SELECTOR_IDS if selector_id not in score_map]
    if missing:
        raise RealWorldValidationError(f"missing reported selector(s): {missing}")
    selectors: list[Selector] = []
    for selector_id in REPORTED_SELECTOR_IDS:
        method_family, scores = score_map[selector_id]
        selected_index = int(scores.astype(float).sort_values(ascending=False).index[0])
        row = ranked.loc[selected_index]
        selectors.append(
            Selector(
                selector_id=selector_id,
                method_family=method_family,
                selected_index=selected_index,
                selected_item_id=int(row["item_id"]),
                selector_score=float(scores.loc[selected_index]),
            )
        )
    return selectors


def candidate_score_series(ranked: pd.DataFrame) -> dict[str, tuple[str, pd.Series]]:
    badness = normalized_objective_badness_matrix(
        ranked,
        objective_columns=OBJECTIVE_COLUMNS,
        objective_directions=OBJECTIVE_DIRECTIONS,
        front_col="front_id",
    )
    scores: dict[str, tuple[str, pd.Series]] = {
        "AMIGA_top1": ("learned_post_pareto", ranked["score"].astype(float)),
        "objective_topsis": ("ideal_antiideal_distance", topsis_scores_from_badness(badness)),
    }
    for objective in OBJECTIVE_COLUMNS:
        scores[f"objective_{objective}"] = ("single_objective", -badness[objective])
    mean_rank_scores = -pd.concat(
        [
            ranked.groupby("front_id")[objective].rank(
                method="average",
                ascending=OBJECTIVE_DIRECTIONS[objective] == "minimize",
            )
            for objective in OBJECTIVE_COLUMNS
        ],
        axis=1,
    ).mean(axis=1)
    scores["objective_mean_rank"] = ("mean_rank", mean_rank_scores)
    return scores


def write_selected_candidates(*, ranked: pd.DataFrame, selectors: list[Selector], output_path: Path) -> pd.DataFrame:
    grn_cols = [column for column in ranked.columns if column.startswith("GRN_") and column.endswith(".csv")]
    rows: list[dict[str, Any]] = []
    for selector in selectors:
        row = ranked.loc[selector.selected_index]
        weights = row[grn_cols].astype(float).sort_values(ascending=False)
        top_weights = weights[weights > 0].head(8)
        payload: dict[str, Any] = {
            "selector_id": selector.selector_id,
            "method": display_selector_name(selector.selector_id),
            "method_family": selector.method_family,
            "selected_item_id": selector.selected_item_id,
            "selector_score": selector.selector_score,
            "amiga_rank": int(row["rank_in_front"]),
            "amiga_score": float(row["score"]),
            "top_weights": "; ".join(
                f"{name.replace('GRN_', '').replace('.csv', '')}={value:.4f}"
                for name, value in top_weights.items()
            ),
        }
        for objective in OBJECTIVE_COLUMNS:
            payload[objective] = float(row[objective])
        rows.append(payload)
    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    return out


def reconstruct_selected_networks(
    *,
    ranked: pd.DataFrame,
    selectors: list[Selector],
    grn_dir: Path,
    networks_dir: Path,
) -> dict[str, Path]:
    network_paths: dict[str, Path] = {}
    for selector in selectors:
        row = ranked.loc[selector.selected_index]
        weights = row_weights_from_front(row)
        summands = []
        for fname, weight in weights.items():
            path = grn_dir / fname
            if not path.exists():
                raise RealWorldValidationError(f"GRN file not found: {path}")
            summands.append(f"{weight}*{path}")
        network = weighted_confidence(summands)
        if network.duplicated(["Source", "Target"]).any():
            raise RealWorldValidationError(f"reconstructed network has duplicated edges: {selector.selector_id}")
        network = network.rename(columns={"Source": "source", "Target": "target", "Confidence": "confidence"})
        network.insert(0, "edge_rank", range(1, len(network) + 1))
        network["source_norm"] = network["source"].map(normalize_symbol)
        network["target_norm"] = network["target"].map(normalize_symbol)
        path = networks_dir / f"{selector.selector_id}_edges.csv"
        network.to_csv(path, index=False)
        network_paths[selector.selector_id] = path
    return network_paths


def load_reported_evidence(
    *,
    cache_dir: Path,
    gene_universe: set[str],
    network_paths: dict[str, Path],
    timeout: int,
    force_refresh: bool,
) -> tuple[pd.DataFrame, list[ResourceStatus]]:
    frames: list[pd.DataFrame] = []
    statuses: list[ResourceStatus] = []

    omnipath, omnipath_statuses = load_omnipath(cache_dir, timeout=timeout, force_refresh=force_refresh)
    frames.append(omnipath)
    statuses.extend(omnipath_statuses)

    trrust, trrust_status = load_trrust(cache_dir, timeout=timeout, force_refresh=force_refresh)
    frames.append(trrust)
    statuses.append(trrust_status)

    jaspar, jaspar_status = load_jaspar(cache_dir, gene_universe=gene_universe, timeout=timeout, force_refresh=force_refresh)
    frames.append(jaspar)
    statuses.append(jaspar_status)

    cistrome, cistrome_status = load_cistrome_cor(
        cache_dir=cache_dir,
        gene_universe=gene_universe,
        network_paths=network_paths,
        timeout=timeout,
        force_refresh=force_refresh,
    )
    frames.append(cistrome)
    statuses.append(cistrome_status)

    evidence = pd.concat(frames, ignore_index=True) if frames else empty_evidence_frame()
    if evidence.empty:
        return evidence, statuses
    evidence["source"] = evidence["source"].map(normalize_symbol)
    evidence["target"] = evidence["target"].map(normalize_symbol)
    evidence = evidence[(evidence["source"] != "") & (evidence["target"] != "")]
    evidence = evidence[evidence["source"].isin(gene_universe) & evidence["target"].isin(gene_universe)]
    evidence = evidence[evidence["source"] != evidence["target"]]
    evidence = evidence.drop_duplicates(["source", "target", "resource", "evidence_type"])
    evidence = evidence.sort_values(["resource", "source", "target"]).reset_index(drop=True)
    for status in statuses:
        status.rows_normalized = int((evidence["resource"] == status.resource).sum())
    return evidence, statuses


def load_omnipath(cache_dir: Path, *, timeout: int, force_refresh: bool) -> tuple[pd.DataFrame, list[ResourceStatus]]:
    cache_path = cache_dir / "omnipath_transcriptional.tsv"
    statuses = [
        ResourceStatus("OmniPath_CollecTRI", "pending", OMNIPATH_URL, str(cache_path)),
        ResourceStatus("OmniPath_DoRothEA", "pending", OMNIPATH_URL, str(cache_path)),
    ]
    try:
        text = get_text(OMNIPATH_URL, cache_path, timeout=timeout, force_refresh=force_refresh)
        raw = pd.read_csv(io.StringIO(text), sep="\t", low_memory=False)
        rows: list[dict[str, Any]] = []
        for _, row in raw.iterrows():
            source_norm = normalize_symbol(row.get("source_genesymbol"))
            if "_" in source_norm:
                continue
            resources = major_omnipath_resources(str(row.get("sources", "")))
            for resource in resources:
                rows.append(
                    {
                        "source": row.get("source_genesymbol"),
                        "target": row.get("target_genesymbol"),
                        "resource": resource,
                        "evidence_type": "tf_target",
                        "confidence": str(row.get("dorothea_level", "")),
                        "raw_fields": json.dumps(
                            {
                                "sources": str(row.get("sources", "")),
                                "references": str(row.get("references", ""))[:1000],
                                "dorothea_level": str(row.get("dorothea_level", "")),
                            },
                            sort_keys=True,
                        ),
                    }
                )
        for status in statuses:
            status.status = "ok"
            status.rows_raw = int(len(raw))
        return pd.DataFrame(rows, columns=empty_evidence_frame().columns), statuses
    except Exception as exc:  # noqa: BLE001
        for status in statuses:
            status.status = "failed"
            status.error = str(exc)
        return empty_evidence_frame(), statuses


def major_omnipath_resources(sources: str) -> list[str]:
    upper = sources.upper()
    resources = []
    if "COLLECTRI" in upper:
        resources.append("OmniPath_CollecTRI")
    if "DOROTHEA" in upper:
        resources.append("OmniPath_DoRothEA")
    return resources


def load_trrust(cache_dir: Path, *, timeout: int, force_refresh: bool) -> tuple[pd.DataFrame, ResourceStatus]:
    cache_path = cache_dir / "trrust_rawdata_human.tsv"
    status = ResourceStatus("TRRUST_v2", "pending", TRRUST_URL, str(cache_path))
    try:
        text = get_text(TRRUST_URL, cache_path, timeout=timeout, force_refresh=force_refresh)
        raw = pd.read_csv(io.StringIO(text), sep="\t", names=["source", "target", "mode", "pmid"])
        status.status = "ok"
        status.rows_raw = int(len(raw))
        return pd.DataFrame(
            {
                "source": raw["source"],
                "target": raw["target"],
                "resource": "TRRUST_v2",
                "evidence_type": "manual_curation",
                "confidence": raw["mode"],
                "raw_fields": raw[["mode", "pmid"]].astype(str).apply(lambda x: x.to_json(), axis=1),
            }
        ), status
    except Exception as exc:  # noqa: BLE001
        status.status = "failed"
        status.error = str(exc)
        return empty_evidence_frame(), status


def load_jaspar(
    cache_dir: Path,
    *,
    gene_universe: set[str],
    timeout: int,
    force_refresh: bool,
) -> tuple[pd.DataFrame, ResourceStatus]:
    url = ENRICHR_LIBRARY_URL.format(library=quote(JASPAR_LIBRARY, safe=""))
    cache_path = cache_dir / "enrichr_tf_target_libraries" / f"{JASPAR_LIBRARY}.gmt"
    status = ResourceStatus("Enrichr_JASPAR_2025", "pending", url, str(cache_path))
    try:
        text = get_text(url, cache_path, timeout=timeout, force_refresh=force_refresh)
        frame, rows_raw = parse_enrichr_gene_set_library(
            text=text,
            gene_universe=gene_universe,
        )
        status.status = "ok"
        status.rows_raw = rows_raw
        return frame, status
    except Exception as exc:  # noqa: BLE001
        status.status = "failed"
        status.error = str(exc)
        return empty_evidence_frame(), status


def parse_enrichr_gene_set_library(*, text: str, gene_universe: set[str]) -> tuple[pd.DataFrame, int]:
    rows: list[dict[str, Any]] = []
    rows_raw = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        rows_raw += 1
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        term = parts[0]
        tfs = parse_enrichr_term_tfs(term)
        targets = [normalize_symbol(target) for target in parts[2:] if normalize_symbol(target) in gene_universe]
        for source in tfs:
            if source not in gene_universe:
                continue
            for target in targets:
                if source == target:
                    continue
                rows.append(
                    {
                        "source": source,
                        "target": target,
                        "resource": "Enrichr_JASPAR_2025",
                        "evidence_type": "motif_target_gene_set",
                        "confidence": term,
                        "raw_fields": json.dumps({"library_term": term}, sort_keys=True),
                    }
                )
    return pd.DataFrame(rows, columns=empty_evidence_frame().columns), rows_raw


def parse_enrichr_term_tfs(term: str) -> list[str]:
    token = term.strip().split()[0] if term.strip() else ""
    token = token.split("_", 1)[0] if token.upper().startswith("PWM_") else token
    out: list[str] = []
    for candidate in token.replace("/", "::").split("::"):
        symbol = normalize_symbol(candidate.split("(", 1)[0])
        if symbol and symbol not in out:
            out.append(symbol)
    return out


def load_cistrome_cor(
    *,
    cache_dir: Path,
    gene_universe: set[str],
    network_paths: dict[str, Path],
    timeout: int,
    force_refresh: bool,
    max_tfs: int = 40,
) -> tuple[pd.DataFrame, ResourceStatus]:
    cistrome_dir = cache_dir / "cistrome_cancer"
    cistrome_dir.mkdir(parents=True, exist_ok=True)
    status = ResourceStatus("CistromeCancer_BRCA_COR", "pending", CISTROME_TARGET_URL, str(cistrome_dir))
    candidate_tfs = cistrome_candidate_tfs(network_paths, max_tfs=max_tfs)
    rows: list[dict[str, Any]] = []
    rows_raw = 0
    errors: list[str] = []
    for tf in candidate_tfs:
        url = f"{CISTROME_TARGET_URL}/{tf}.cor.csv"
        cache_path = cistrome_dir / f"{tf}.cor.csv"
        try:
            text = get_text(url, cache_path, timeout=timeout, force_refresh=force_refresh)
            raw = pd.read_csv(io.StringIO(text))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{tf}.cor.csv: {exc}")
            continue
        rows_raw += len(raw)
        if not set(CISTROME_BRCA_COLUMNS).issubset(raw.columns):
            errors.append(f"{tf}.cor.csv: missing BRCA columns")
            continue
        target_col = raw.columns[0]
        raw[target_col] = raw[target_col].map(normalize_symbol)
        raw = raw[raw[target_col].isin(gene_universe)]
        scores = raw[list(CISTROME_BRCA_COLUMNS)].apply(pd.to_numeric, errors="coerce").max(axis=1)
        raw = raw[scores > 0.0].copy()
        scores = scores.loc[raw.index]
        for target, score in zip(raw[target_col], scores, strict=False):
            if tf == target:
                continue
            rows.append(
                {
                    "source": tf,
                    "target": target,
                    "resource": "CistromeCancer_BRCA_COR",
                    "evidence_type": "brca_cor",
                    "confidence": float(score),
                    "raw_fields": json.dumps({"metric": "COR", "score": float(score)}, sort_keys=True),
                }
            )
    status.status = "ok" if rows else "no_pairs"
    status.rows_raw = int(rows_raw)
    if errors:
        status.error = " | ".join(errors[:20])
    return pd.DataFrame(rows, columns=empty_evidence_frame().columns), status


def cistrome_candidate_tfs(network_paths: dict[str, Path], *, max_tfs: int) -> list[str]:
    sources: list[str] = []
    for path in network_paths.values():
        network = pd.read_csv(path).head(5000)
        source_column = "source_norm" if "source_norm" in network.columns else "source"
        sources.extend(network[source_column].map(normalize_symbol).tolist())
    counts = pd.Series([source for source in sources if source]).value_counts()
    return list(counts.head(max_tfs).index)


def get_text(url: str, cache_path: Path, *, timeout: int, force_refresh: bool) -> str:
    if cache_path.exists() and not force_refresh:
        return cache_path.read_text(encoding="utf-8")
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "AMIGA-real-world-validation/1.0"})
    response.raise_for_status()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(response.text, encoding="utf-8")
    return response.text


def build_reported_source_support_table(
    *,
    selectors: list[Selector],
    network_paths: dict[str, Path],
    evidence: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for selector in selectors:
        network = pd.read_csv(network_paths[selector.selector_id])
        row: dict[str, Any] = {
            "method": display_selector_name(selector.selector_id),
            "selector_id": selector.selector_id,
            "candidate": selector.selected_item_id,
        }
        for source in REPORTED_SOURCES:
            label = source["label"]
            resource = source["resource"]
            cutoff = int(source["cutoff"])
            resource_evidence = evidence[evidence["resource"] == resource]
            pair_set = set(zip(resource_evidence["source"], resource_evidence["target"]))
            tf_sources = set(resource_evidence["source"])
            top = network.head(cutoff)
            pairs = list(zip(top["source_norm"].map(normalize_symbol), top["target_norm"].map(normalize_symbol)))
            tf_target_edges = int(sum(pair in pair_set for pair in pairs))
            tf_source_edges = int(sum(pair[0] in tf_sources for pair in pairs))
            prefix = slugify_label(label)
            row[f"{prefix}_cutoff"] = cutoff
            row[f"{prefix}_tf_target_edges"] = tf_target_edges
            row[f"{prefix}_tf_target_rate"] = tf_target_edges / cutoff
            row[f"{prefix}_tf_source_edges"] = tf_source_edges
            row[f"{prefix}_tf_source_rate"] = tf_source_edges / cutoff
        rows.append(row)
    return pd.DataFrame(rows)


def build_reported_markdown(table: pd.DataFrame) -> str:
    display_rows = []
    for _, row in table.iterrows():
        payload = {"Método": row["method"], "Candidato": int(row["candidate"])}
        for source in REPORTED_SOURCES:
            label = source["label"]
            prefix = slugify_label(label)
            cutoff = int(row[f"{prefix}_cutoff"])
            payload[f"{label} TF-target (N={cutoff})"] = format_count_rate(
                row[f"{prefix}_tf_target_edges"],
                row[f"{prefix}_tf_target_rate"],
            )
            payload[f"{label} TF-source (N={cutoff})"] = format_count_rate(
                row[f"{prefix}_tf_source_edges"],
                row[f"{prefix}_tf_source_rate"],
            )
        display_rows.append(payload)
    sections = [
        "# TCGA-BRCA reported real-world validation",
        "",
        "This table is intentionally restricted to the protocol reported in the paper: Top1 selectors only, "
        "CollecTRI/DoRothEA/TRRUST/JASPAR at top-250 and Cistrome Cancer BRCA-COR at top-5000.",
        "",
        dataframe_to_markdown(pd.DataFrame(display_rows)),
        "",
    ]
    return "\n".join(sections)


def empty_evidence_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["source", "target", "resource", "evidence_type", "confidence", "raw_fields"])


def display_selector_name(selector_id: str) -> str:
    return SELECTOR_DISPLAY_NAMES.get(selector_id, selector_id)


def slugify_label(label: str) -> str:
    return label.lower().replace(" ", "_").replace("-", "_").replace("/", "_").replace("(", "").replace(")", "")


def normalize_symbol(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip().upper()


def format_count_rate(count: Any, rate: Any) -> str:
    if pd.isna(count):
        return ""
    return f"{int(round(float(count)))} ({float(rate):.3f})"


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines)
