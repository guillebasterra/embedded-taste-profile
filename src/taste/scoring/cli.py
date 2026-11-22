"""`taste critique`, `taste search`, `taste eval`."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

_HOME_HELP = "TASTE_HOME (default ~/.taste)."


def _setup(home: Path | None):
    from taste.config import load_settings
    from taste.embed import get_embedder
    from taste.store import Store

    settings = load_settings(home)
    return Store(settings.home), get_embedder(settings)


def register(app: typer.Typer) -> None:
    @app.command()
    def critique(
        image: Path = typer.Argument(..., help="Image to score, e.g. a screenshot."),
        home: Optional[Path] = typer.Option(None, "--home", help=_HOME_HELP),
        as_json: bool = typer.Option(False, "--json", help="Print the raw result as JSON."),
    ) -> None:
        """Score an image against my taste clusters."""
        from rich.console import Console
        from rich.table import Table

        from taste.scoring import api

        console = Console()
        try:
            store, embedder = _setup(home)
            result = api.critique(store, embedder, image)
        except (RuntimeError, FileNotFoundError) as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)
        if as_json:
            typer.echo(result.model_dump_json(indent=2))
            return
        console.print(f"Score [bold]{result.score:.2f}[/bold]  "
                      f"(similarity {result.raw_similarity:.3f}, domain {result.domain})")
        table = Table(title="Nearest clusters")
        for col in ("cluster", "name", "similarity"):
            table.add_column(col)
        for m in result.nearest:
            table.add_row(m.cluster_id, m.name or "", f"{m.similarity:.3f}")
        console.print(table)
        if result.nearest_anti:
            console.print("Disliked references: "
                          + ", ".join(f"{s:.3f}" for s in result.nearest_anti))
        for note in result.notes:
            console.print(f"- {note}")

    @app.command()
    def search(
        query: str = typer.Argument(..., help="What to look for, in words."),
        home: Optional[Path] = typer.Option(None, "--home", help=_HOME_HELP),
        domain: Optional[str] = typer.Option(None, "--domain", help="Only this domain."),
        k: int = typer.Option(8, "-k", "--k", min=1, help="How many pins to return."),
        as_json: bool = typer.Option(False, "--json", help="Print the hits as JSON."),
    ) -> None:
        """Find my pins that match a text query."""
        import json

        from rich.console import Console
        from rich.table import Table

        from taste.scoring import api

        console = Console()
        try:
            store, embedder = _setup(home)
            hits = api.search(store, embedder, query, domain=domain, k=k)
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)
        if as_json:
            typer.echo(json.dumps([h.model_dump() for h in hits], indent=2))
            return
        if not hits:
            console.print("No matches. Run `taste embed` first if the index is empty.")
            return
        table = Table(title=f"Search: {query}")
        for col in ("pin", "similarity", "cluster", "board", "image"):
            table.add_column(col)
        for h in hits:
            table.add_row(h.pin_id, f"{h.similarity:.3f}", h.cluster_id or "", h.board or "",
                          h.image_path)
        console.print(table)

    @app.command("eval")
    def eval_(
        distractors: Path = typer.Option(..., "--distractors",
                                         help="Folder of images I did not save."),
        home: Optional[Path] = typer.Option(None, "--home", help=_HOME_HELP),
        holdout: float = typer.Option(0.2, "--holdout", help="Share of pins held out."),
        seed: int = typer.Option(0, "--seed", help="Seed for the hold-out split."),
        threshold: Optional[float] = typer.Option(
            None, "--threshold", help="Clustering threshold (default: same as `taste cluster`)."),
        min_size: Optional[int] = typer.Option(None, "--min-size", min=1,
                                               help="Smallest cluster worth keeping."),
    ) -> None:
        """Check that scoring separates held-out pins from distractor images."""
        from rich.console import Console

        from taste.scoring.evaluate import evaluate

        console = Console()
        try:
            store, embedder = _setup(home)
            r = evaluate(store, embedder, distractors, holdout=holdout, seed=seed,
                         threshold=threshold, min_size=min_size)
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)
        console.print(f"{r.n_train} pins held in ({r.n_clusters} clusters), "
                      f"{r.n_heldout} held out, {r.n_distractors} distractors.")
        console.print(f"Mean similarity: held-out {r.heldout_mean:.3f}, "
                      f"distractors {r.distractor_mean:.3f}")
        console.print(f"ROC AUC: [bold]{r.auc:.3f}[/bold]")
        console.print(f"Suggested accept threshold: similarity >= {r.threshold:.3f} "
                      f"(accepts {r.tpr:.0%} of held-out pins, {r.fpr:.0%} of distractors)")
