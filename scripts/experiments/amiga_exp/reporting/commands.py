"""CLI for reproducible reports from complete evaluation summaries."""
from pathlib import Path
import json

import typer

app = typer.Typer(no_args_is_help=True, help="Audited statistical tables and figures from completed evaluations.")


@app.command("supervised")
def supervised(summary: Path = typer.Option(..., exists=True, file_okay=False,
                                            help="Complete summary containing topology_metrics.csv and manifest.json."),
               output: Path = typer.Option(..., help="New report directory outside the source summary."),
               audit_manifest: list[Path] | None = typer.Option(None, exists=True, dir_okay=False,
                                                               help="Additional completed summary to audit; repeatable."),
               expected_topologies: int = typer.Option(87, min=2),
               figures: bool = typer.Option(True, "--figures/--no-figures")):
    from .supervised import write_report
    try:
        result = write_report(summary, output, audit_manifests=tuple(audit_manifest or ()),
                              expected_topologies=expected_topologies, figures=figures)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        typer.secho(f"Report error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(dict(status=result["status"], output=str(output),
                               workflow_version=result["workflow_version"], methods=result["methods"],
                               audited_artifacts=sum(a["verified_artifacts"] for a in result["artifact_audit"])), indent=2))


@app.command("figures")
def figures(output: Path = typer.Option(..., help="New destination for figures and plotting CSVs."),
            selection_summary: Path = typer.Option(Path("experiments/sequential-selection/summaries/full-001"), exists=True, file_okay=False),
            selection_run: Path = typer.Option(Path("experiments/sequential-selection/runs/full-001"), exists=True, file_okay=False),
            top5_summary: Path = typer.Option(Path("experiments/top5-classification/full-001/selection-summary"), exists=True, file_okay=False),
            top5_run: Path = typer.Option(Path("experiments/top5-classification/full-001/selection-run"), exists=True, file_okay=False),
            top10_summary: Path = typer.Option(Path("experiments/top10-classification/full-001/selection-summary"), exists=True, file_okay=False),
            top10_run: Path = typer.Option(Path("experiments/top10-classification/full-001/selection-run"), exists=True, file_okay=False),
            outer_summary: Path = typer.Option(Path("experiments/top10-classification/full-001/outer-summary"), exists=True, file_okay=False),
            layout: str = typer.Option("separate", help="separate: supervised/objective panels; joint: a single comparison family."),
            top_features: int = typer.Option(20, "--feature-count", "--top-features", min=5, max=40,
                                             help="Total displayed columns, split between highest and lowest training SHAP.")):
    """Refresh all four phase designs using complete current results, without fitting."""
    from .figures import generate_figures
    try:
        result = generate_figures(selection_summary, selection_run, top5_summary, top5_run,
                                  top10_summary, top10_run, outer_summary, output,
                                  layout=layout, top_features=top_features)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        typer.secho(f"Figure error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(dict(status=result["status"], output=str(output), layout=layout,
                               pdf_figures=sum(p.endswith(".pdf") for p in result["artifacts"]),
                               no_model_fits=result["no_model_fits"]), indent=2))
