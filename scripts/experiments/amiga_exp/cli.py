"""Command line interface for the AMIGA experimental protocol."""
import typer

from amiga import __version__
from scripts.experiments.amiga_exp.studies.commands import benchmark_app, real_app, costs_app, organize
from scripts.experiments.amiga_exp.supplementary.commands import freeze_command, run_command, status_command

app = typer.Typer(add_completion=False, no_args_is_help=True,
    help="Benchmark phases 1–5, real-data application and prediction costs.")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(real_app, name="real")
app.add_typer(costs_app, name="costs")
app.command("prepare", help="Freeze fixed learning curves and deployment after benchmark comparison.")(freeze_command)
app.command("run", help="Run the prepared learning curves and real-data deployment together.")(run_command)
app.command("status", help="Inspect the shared learning/deployment execution.")(status_command)
app.command("organize")(organize)


def _show_version(value: bool):
    if value:
        typer.echo(f"AMIGA {__version__}")
        raise typer.Exit()


@app.callback()
def main(version: bool = typer.Option(False, "--version", callback=_show_version,
                                      is_eager=True, help="Show the AMIGA project version.")):
    """Repository experimental workflow, distributed with the AMIGA source."""


if __name__ == "__main__":
    app()
