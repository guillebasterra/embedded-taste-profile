"""`taste embed`."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer


def register(app: typer.Typer) -> None:
    @app.command()
    def embed(
        home: Optional[Path] = typer.Option(None, "--home", help="TASTE_HOME (default ~/.taste)."),
        force: bool = typer.Option(False, "--force", help="Re-embed every pin."),
        chunk_size: int = typer.Option(64, "--chunk-size", min=1,
                                       help="Pins per embedding call."),
    ) -> None:
        """Embed every included pin that has an image. Incremental."""
        from rich.console import Console
        from rich.progress import Progress

        from taste.config import load_settings
        from taste.embed import get_embedder
        from taste.embed.index import embed_pins
        from taste.store import Store

        console = Console()
        settings = load_settings(home)
        store = Store(settings.home)
        try:
            embedder = get_embedder(settings)
            with Progress(console=console, transient=True) as progress:
                task = progress.add_task(f"Embedding with {embedder.name}", total=None)
                stats = embed_pins(
                    store, embedder, chunk_size=chunk_size, force=force,
                    on_progress=lambda done, total: progress.update(
                        task, completed=done, total=total),
                )
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)

        if not stats.total and not stats.failed:
            console.print("No pins with images to embed. Run `taste ingest` first.")
            return
        console.print(
            f"Embedded {stats.new} new pins with [bold]{embedder.name}[/bold] "
            f"({stats.kept} already done, {stats.dropped} stale dropped, {stats.total} total)."
        )
        if stats.failed:
            console.print(f"[yellow]{len(stats.failed)} pins skipped: image missing or "
                          f"unreadable (e.g. {stats.failed[0]}).[/yellow]")
