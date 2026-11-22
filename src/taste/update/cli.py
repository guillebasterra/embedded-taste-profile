"""`taste update`: ingest (replaying the last recorded source), domains, embed, cluster,
distill and install/export, each as the regular command so output and exit codes match."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

STATE_FILE = "update_state.json"  # remembers --export so later runs keep exporting


def ingest_args(source: dict) -> list[str]:
    """`taste ingest` arguments that replay a recorded source."""
    args = ["ingest", "--source", source["source"]]
    if source.get("url"):
        args += ["--url", source["url"]]
    if source.get("path"):
        args += ["--path", source["path"]]
    if source.get("browser"):
        args += ["--browser", source["browser"]]
    for board in source.get("board") or []:
        args += ["--board", board]
    return args


def _read_state(home: Path) -> dict:
    try:
        data = json.loads((home / STATE_FILE).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _invoke(app: typer.Typer, args: list[str]) -> int:
    """Run one `taste` command in-process. Returns its exit code."""
    try:
        code = app(args, standalone_mode=False)
    except SystemExit as exc:  # a command that calls sys.exit itself
        code = exc.code
    return code if isinstance(code, int) else 0


def register(app: typer.Typer) -> None:
    @app.command()
    def update(
        home: Optional[Path] = typer.Option(None, "--home",
                                            help="TASTE_HOME (default ~/.taste)."),
        skip_ingest: bool = typer.Option(False, "--skip-ingest",
                                         help="Rebuild from the pins already on disk."),
        export: Optional[Path] = typer.Option(
            None, "--export",
            help="Copy profile/ here at the end (remembered for later runs)."),
    ) -> None:
        """Refresh everything: ingest, domains, embed, cluster, distill, install (+ export).

        Every step is incremental: only new pins are fetched and embedded, cluster ids stay
        stable, and only clusters whose exemplars changed are re-distilled."""
        from taste.config import load_settings
        from taste.ingest.pipeline import read_last_source
        from taste.store import Store

        store = Store(load_settings(home).home)
        steps: list[list[str]] = []
        if not skip_ingest:
            source = read_last_source(store)
            if source is None:
                typer.secho("error: no recorded ingest source. Run `taste ingest ...` once "
                            "(it is remembered), or pass --skip-ingest.",
                            fg=typer.colors.RED, err=True)
                raise typer.Exit(1)
            steps.append(ingest_args(source))
        steps += [["domains"], ["embed"], ["cluster"], ["distill"]]
        state = _read_state(store.home)
        export_dir = str(export.expanduser().resolve()) if export else state.get("export")
        steps.append(["install", *(["--export", export_dir] if export_dir else [])])

        for n, args in enumerate(steps, 1):
            typer.secho(f"\n[{n}/{len(steps)}] taste {' '.join(args)}", bold=True)
            code = _invoke(app, [*args, "--home", str(store.home)])
            if code:
                typer.secho(f"Stopped: `taste {args[0]}` exited with {code}. Fix that and "
                            "re-run `taste update`; finished steps are not redone.",
                            fg=typer.colors.RED, err=True)
                raise typer.Exit(code)
        if export_dir:
            (store.home / STATE_FILE).write_text(json.dumps({"export": export_dir}, indent=2))
        typer.secho("\nProfile is up to date.", fg=typer.colors.GREEN)
