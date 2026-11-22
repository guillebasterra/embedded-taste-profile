"""`taste cluster`."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from taste.cluster.core import DEFAULT_MIN_SIZE, DEFAULT_THRESHOLD


def register(app: typer.Typer) -> None:
    @app.command()
    def cluster(
        home: Optional[Path] = typer.Option(None, "--home", help="TASTE_HOME (default ~/.taste)."),
        threshold: float = typer.Option(
            DEFAULT_THRESHOLD, "--threshold", min=0.0, max=2.0,
            help="Cosine distance where merging stops. Lower = more, tighter clusters."),
        min_size: int = typer.Option(DEFAULT_MIN_SIZE, "--min-size", min=1,
                                     help="Smallest cluster worth keeping."),
    ) -> None:
        """Group pins into taste clusters per domain and write clusters.json."""
        from rich.console import Console
        from rich.table import Table

        from taste.cluster.core import cluster_store
        from taste.config import load_settings
        from taste.embed import get_embedder
        from taste.store import Store

        console = Console()
        settings = load_settings(home)
        store = Store(settings.home)
        try:
            model = get_embedder(settings).name
            missing = len(set(store.pin_domains()) - set(store.read_embeddings(model)[0]))
            result = cluster_store(store, model, threshold=threshold, min_size=min_size)
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)

        table = Table(title=f"Clusters (threshold {threshold:g}, min size {min_size})")
        for col in ("id", "size", "cohesion", "boards", "palette", "neighbors"):
            table.add_column(col)
        for c in result.clusters:
            swatches = " ".join(f"[{h}]██[/] {h}" for h in c.palette[:3])
            table.add_row(c.id, str(c.size), f"{c.cohesion:.3f}", ", ".join(c.top_boards),
                          swatches, ", ".join(c.neighbor_cluster_ids))
        console.print(table)
        for d in result.domains:
            if d.skipped:
                console.print(f"{d.domain}: skipped, only {d.n_pins} pins "
                              f"(needs {min_size}).")
            else:
                console.print(
                    f"{d.domain}: {d.n_pins} pins, {d.n_clusters} clusters, "
                    f"{d.n_unclustered} unclustered, median pairwise distance "
                    f"{d.median_distance:.2f}")
        if missing:
            console.print(f"[yellow]{missing} pins have no embedding yet. "
                          "Run `taste embed`.[/yellow]")
        console.print(f"Wrote {len(result.clusters)} clusters to {store.clusters_path}")
