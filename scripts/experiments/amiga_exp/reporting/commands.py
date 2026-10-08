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
