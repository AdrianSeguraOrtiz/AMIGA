"""Paired topology ranks and exploratory Friedman/Holm control comparisons."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import scipy
from scipy.stats import friedmanchisquare, norm
import statsmodels
from statsmodels.stats.multitest import multipletests

from scripts.experiments.amiga_exp.version import __version__

METHODS = ("ranking", "reg_aupr", "clf_top05", "clf_top10", "clf_top20")
CONTROL = "ranking"
MAIN_METRICS = ("Regret@5", "Regret@1", "Hit@1", "Hit@5")
METRICS = tuple(f"{metric}@{k}" for metric in ("Regret", "BestAUPR", "Hit")
                for k in (1, 3, 5, 10))
LABELS = {"ranking": "AMIGA", "reg_aupr": "AUPR regression",
          "clf_top05": "Classification top 5%", "clf_top10": "Classification top 10%",
          "clf_top20": "Classification top 20%"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(path: Path) -> tuple[dict, dict[Path, str]]:
    """Verify complete summaries, rejecting escaping or malformed artifact paths."""
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (payload.get("status") != "complete" or not isinstance(payload.get("artifacts"), dict)
            or not payload["artifacts"]):
        raise ValueError(f"Expected a complete summary with artifact hashes: {path}")
    identities = {path: sha256(path)}
    for name, expected in payload["artifacts"].items():
        relative = Path(name)
        target = (path.parent / relative).resolve()
        if (relative.is_absolute() or ".." in relative.parts
                or not target.is_relative_to(path.parent) or "\\" in name
                or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)):
            raise ValueError(f"Invalid artifact identity: {name}")
        if sha256(target) != expected:
            raise ValueError(f"Artifact SHA-256 mismatch: {target}")
        identities[target] = expected
    return payload, identities


def build_tables(frame: pd.DataFrame, *, expected_topologies: int | None = 87) -> dict[str, pd.DataFrame]:
    """Rank fixed methods within each case/topology, retaining raw metric means.

    Inputs must already average metrics over seeds and then conditions. The
    control is fixed in advance; it is never chosen from the observed ranks.
    """
    required = {"case", "topology_id", "method", *METRICS}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError(f"Missing topology metrics: {sorted(required - set(frame.columns))}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Topology metrics and identifiers must be nonmissing")
    if expected_topologies is not None and expected_topologies < 2:
        raise ValueError("At least two paired topologies are required")
    data = frame.loc[frame["method"].isin(METHODS)].copy()
    if data.empty or set(data["case"]) != set(frame["case"]):
        raise ValueError("Every case must contain the selected methods")
    if data.duplicated(["case", "topology_id", "method"]).any():
        raise ValueError("Duplicate case/topology/method observations")
    numeric = data.loc[:, list(METRICS)].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric < -1e-12).any() or (numeric > 1 + 1e-12).any():
        raise ValueError("Metrics must be finite and within [0, 1]")
    data.loc[:, list(METRICS)] = numeric
    tables, rank_rows, mean_rows, omnibus_rows, contrast_rows = {}, [], [], [], []
    for case, case_data in data.groupby("case", sort=True):
        if not isinstance(case, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", case):
            raise ValueError("Case identifiers must be portable filename components")
        n = case_data["topology_id"].nunique()
        if n < 2 or (expected_topologies is not None and n != expected_topologies):
            raise ValueError(f"Unexpected paired topology count for {case}: {n}")
        if set(case_data["method"]) != set(METHODS) or len(case_data) != n * len(METHODS):
            raise ValueError(f"Incomplete paired method coverage for {case}")
        for metric in METRICS:
            raw = case_data.pivot(index="topology_id", columns="method", values=metric).reindex(columns=METHODS)
            if raw.isna().any().any():
                raise ValueError(f"Incomplete paired topology coverage for {case}")
            values = raw.round(12)
            ranks = values.rank(axis=1, method="average", ascending=metric.startswith("Regret"))
            average = ranks.mean()
            for method in METHODS:
                rank_rows.append(dict(case=case, method=method, metric=metric,
                                      mean_rank=average[method], n_topologies=n))
                mean_rows.append(dict(case=case, method=method, metric=metric,
                                      metric_mean=raw[method].mean(), n_topologies=n))
            if metric != "Regret@5":
                continue
            tables[f"{case}-regret5.csv"] = values
            tables[f"{case}-regret5-ranks.csv"] = ranks
            k = len(METHODS)
            if np.equal(values.to_numpy(), values.to_numpy()[:, :1]).all():
                statistic, p = 0.0, 1.0
            else:
                statistic, p = friedmanchisquare(*[values[m].to_numpy() for m in METHODS])
            omnibus_rows.append(dict(case=case, metric=metric, statistic=statistic, df=k-1,
                                     p_value=p, n_topologies=n, n_methods=k))
            standard_error = np.sqrt(k * (k + 1) / (6 * n))
            contrasts = []
            for method in METHODS[1:]:
                delta = average[method] - average[CONTROL]
                z = delta / standard_error
                contrasts.append(dict(case=case, control=CONTROL, comparator=method, metric=metric,
                                      mean_rank_delta=delta, standard_error=standard_error,
                                      z=z, p_raw=2 * norm.sf(abs(z))))
            adjusted = multipletests([r["p_raw"] for r in contrasts], method="holm")[1]
            for row, adjusted_p in zip(contrasts, adjusted, strict=True):
                row.update(p_holm_within_case_four=adjusted_p, omnibus_p=p,
                           omnibus_rejected_005=bool(p < .05),
                           reject_005_after_omnibus_and_holm=bool(p < .05 and adjusted_p < .05))
            contrast_rows.extend(contrasts)
    ranks, means = pd.DataFrame(rank_rows), pd.DataFrame(mean_rows)
    contrasts = pd.DataFrame(contrast_rows)
    contrasts["p_holm_joint_eight"] = multipletests(contrasts["p_raw"], method="holm")[1]
    tables.update({"mean_ranks_all_metrics.csv": ranks, "raw_metric_means.csv": means,
                   "friedman_omnibus.csv": pd.DataFrame(omnibus_rows),
                   "regret5_posthoc_holm.csv": contrasts})
    for case in sorted(data["case"].unique()):
        table = ranks.loc[ranks["case"] == case].pivot(index="method", columns="metric", values="mean_rank")
        table = table.loc[list(METHODS), list(MAIN_METRICS)]
        ps = contrasts.loc[contrasts["case"] == case].set_index("comparator")["p_holm_within_case_four"]
        table["p_Holm_Regret@5_vs_AMIGA"] = [np.nan] + [ps[m] for m in METHODS[1:]]
        tables[f"{case}-publication-table.csv"] = table
    return tables


def latex_table(table: pd.DataFrame) -> str:
    """Export the fixed compact table without optional template dependencies."""
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             "Method & " + " & ".join(MAIN_METRICS) + r" & Holm $p$ (Regret@5) \\", r"\midrule"]
    for method in METHODS:
        label = LABELS[method].replace("%", r"\%")
        cells = [f"{table.loc[method, metric]:.4f}" for metric in MAIN_METRICS]
        p = table.loc[method, "p_Holm_Regret@5_vs_AMIGA"]
        lines.append(" & ".join([label, *cells, "--" if np.isnan(p) else f"{p:.4f}"]) + r" \\")
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}", ""])


def write_report(summary: Path, output: Path, *, audit_manifests: tuple[Path, ...] = (),
                 expected_topologies: int = 87, figures: bool = True) -> dict:
    """Audit inputs and write into a new destination without changing any run."""
    summary, output = Path(summary).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(summary):
        raise ValueError("Use a new output directory outside the source summary")
    source = summary / "topology_metrics.csv"
    audits, identities = [], {}
    for path in dict.fromkeys([summary / "manifest.json", *map(Path, audit_manifests)]):
        payload, verified = verify_manifest(path)
        identities.update(verified)
        audits.append(dict(manifest=str(Path(path).resolve()), sha256=verified[Path(path).resolve()],
                           verified_artifacts=len(payload["artifacts"])))
    if source not in identities:
        raise ValueError("Source summary manifest must cover topology_metrics.csv")
    tables = build_tables(pd.read_csv(source, dtype={"case": str, "topology_id": str, "method": str}),
                          expected_topologies=expected_topologies)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".amiga-report-", dir=output.parent) as temporary:
        destination = Path(temporary) / "report"
        destination.mkdir()
        for name, table in tables.items():
            table.to_csv(destination / name, index=name.endswith(("-regret5.csv", "-regret5-ranks.csv", "-publication-table.csv")))
            if name.endswith("-publication-table.csv"):
                (destination / name.replace(".csv", ".tex")).write_text(
                    latex_table(table), encoding="utf-8")
        if figures:
            from .plots import plot_tables
            plot_tables(tables, destination)
        for path, expected in identities.items():
            if sha256(path) != expected:
                raise ValueError(f"Input changed while reporting: {path}")
        manifest = dict(
            schema_version=1, status="complete", workflow="supervised_rank_report",
            workflow_version=__version__, created_at_utc=datetime.now(timezone.utc).isoformat(),
            evidence_role="exploratory_reporting_sensitivity_no_model_refits",
            methods=list(METHODS), control=CONTROL, metric_primary="Regret@5",
            reported_core_metrics=list(MAIN_METRICS), all_saved_metrics=list(METRICS),
            n_topologies_per_case=expected_topologies,
            unit="equally weighted topology means of seed-averaged, then condition-averaged metrics",
            numerical_ties="round metrics to 12 decimals before average tied ranks",
            omnibus="SciPy asymptotic chi-square Friedman including tie correction, separately per case",
            posthoc="two-sided normal z=(comparator rank-control rank)/sqrt(k*(k+1)/(6*N))",
            holm_primary_display="four fixed-control comparisons per case",
            holm_additional_sensitivity="jointly across all control comparisons in all cases",
            original_protocol_tests="unchanged; this reporting analysis is additional to saved Wilcoxon-Holm results",
            ranking_identity="sequentially selected family, labels, parameters and feature fraction per outer fold",
            artifact_audit=audits,
            inputs_sha256={str(p): digest for p, digest in identities.items()},
            implementation_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
            versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                          scipy=scipy.__version__, statsmodels=statsmodels.__version__),
            limitations=["Reused benchmarks and overlapping training sets limit independence and inference.",
                         "Top-5% and top-10% thresholds are exploratory sensitivity analyses.",
                         "A first mean rank alone does not demonstrate superiority; nonsignificance is not equivalence.",
                         "Ranks and multiplicity depend on the selected method set; complete source summaries remain unchanged."],
            artifacts={p.name: sha256(p) for p in sorted(destination.iterdir()) if p.is_file()})
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        destination.rename(output)
    return manifest
