"""`taste facets`."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer


def register(app: typer.Typer) -> None:
    @app.command()
    def facets(
        home: Optional[Path] = typer.Option(None, "--home", help="TASTE_HOME (default ~/.taste)."),
        sample: Optional[int] = typer.Option(None, "--sample", min=1,
                                             help="Only this many pending pins, at random."),
        model: Optional[str] = typer.Option(
            None, "--model", help="Model for readings (default TASTE_FACETS_MODEL or "
                                  "claude-haiku-5-5)."),
        measure_only: bool = typer.Option(False, "--measure-only",
                                          help="Pixel statistics only, no model calls."),
        workers: int = typer.Option(8, "--workers", min=1, help="Parallel readings."),
        force: bool = typer.Option(False, "--force", help="Redo every pin."),
    ) -> None:
        """Measure and read every included pin's visual qualities. Incremental."""
        from rich.console import Console
        from rich.progress import Progress

        from taste.config import load_settings
        from taste.distill.llm import VLMError
        from taste.facets.read import facets_dir, read_model, reading_vlm, run_facets
        from taste.store import Store

        console = Console()
        settings = load_settings(home)
        store = Store(settings.home)
        try:
            vlm = None if measure_only else reading_vlm(settings, read_model(model))
        except VLMError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)
        label = "Measuring" if vlm is None else f"Reading with {vlm.name}"
        with Progress(console=console, transient=True) as progress:
            task = progress.add_task(label, total=None)
            stats = run_facets(store, vlm, sample=sample, force=force, workers=workers,
                               on_progress=lambda d, t: progress.update(task, completed=d,
                                                                         total=t))
        console.print(f"{stats.measured} measured, {stats.read} read, {stats.kept} already "
                      f"done, {len(stats.failed)} failed. Files in {facets_dir(store)}.",
                      soft_wrap=True)
        for pin_id, reason in stats.failed[:5]:
            console.print(f"[yellow]  {pin_id}: {reason}[/yellow]", soft_wrap=True)
        if stats.failed:
            raise typer.Exit(1)

    @app.command()
    def threads(
        home: Optional[Path] = typer.Option(None, "--home", help="TASTE_HOME (default ~/.taste)."),
        k: int = typer.Option(16, "--k", min=2, help="Number of threads to look for."),
        seed: int = typer.Option(0, "--seed"),
        describe: bool = typer.Option(True, "--describe/--no-describe",
                                      help="Name each thread and write its card with the VLM."),
        force: bool = typer.Option(False, "--force", help="Rewrite every thread card."),
    ) -> None:
        """Find bundles of visual qualities that recur across boards and domains."""
        from rich.console import Console
        from rich.table import Table

        from taste.config import load_settings
        from taste.facets.threads import fit_threads, render_preview, write_threads
        from taste.store import Store

        console = Console()
        settings = load_settings(home)
        store = Store(settings.home)
        try:
            found = fit_threads(store, k=k, seed=seed)
        except ValueError as exc:  # too few readings, or tags too uniform to factorize
            # not an error for `taste update` on a small home: the rest of the chain still runs
            console.print(f"[yellow]Skipped threads: {exc}[/yellow]", soft_wrap=True)
            return
        path = write_threads(store, found)
        table = Table("id", "pins", "spread", "qualities", "top boards")
        for t in found:
            table.add_row(t.id, str(len(t.pins)), f"{t.spread}", ", ".join(t.terms[:8]),
                          ", ".join(list(t.boards)[:4]))
        console.print(table)
        console.print(f"Spread is the effective number of boards a thread reaches. "
                      f"Saved {path}; contact sheet: {render_preview(store, found)}",
                      soft_wrap=True)
        if not describe:
            return
        from taste.distill.llm import VLMError, get_vlm
        from taste.facets.describe import describe_threads

        try:
            vlm = get_vlm(settings)
            with console.status(f"Writing thread cards with {vlm.name}"):
                cards, generated = describe_threads(store, found, vlm, force=force)
        except VLMError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)
        for c in cards:
            console.print(f"  {c.thread_id} [bold]{c.name}[/bold] "
                          f"(distinctiveness {c.distinctiveness}/5): {c.one_liner}",
                          soft_wrap=True)
        console.print(f"{generated} thread cards written, {len(cards) - generated} reused. "
                      f"{store.profile_dir / 'threads.md'}. Run `taste distill` to fold them "
                      "into TASTE.md.", soft_wrap=True)
