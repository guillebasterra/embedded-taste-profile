"""Root CLI. Each feature package registers its own commands.

SHARED CONTRACT FILE: owned by the orchestrator. Do not edit in a feature branch.
Add commands inside your package's `cli.py::register(app)`.
"""

from __future__ import annotations

import importlib

import typer

app = typer.Typer(help="Build and serve a portable visual taste profile.", no_args_is_help=True)



@app.callback()
def _root() -> None:
    """Build and serve a portable visual taste profile."""


_MODULES = (
    "taste.ingest.cli",   # auth, ingest, domains
    "taste.embed.cli",    # embed
    "taste.cluster.cli",  # cluster
    "taste.scoring.cli",  # critique, search, eval
    "taste.distill.cli",  # distill
    "taste.serve.cli",    # serve, feedback, install
    "taste.update.cli",   # update (chains the others)
)

for _name in _MODULES:
    importlib.import_module(_name).register(app)


@app.command()
def version() -> None:
    """Print the version."""
    from taste import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()
