"""`taste cluster`."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from taste.cluster.core import DEFAULT_MIN_SIZE, DEFAULT_THRESHOLD

PARAMS_FILE = "cluster_params.json"  # last --threshold / --min-size, reused by default


def read_params(store) -> dict:
    import json

    try:
        data = json.loads((store.home / PARAMS_FILE).read_text())
    except (OSError, ValueError):
        return {}
    return {k: data[k] for k in ("threshold", "min_size")
            if isinstance(data, dict) and isinstance(data.get(k), (int, float))}


def write_params(store, **params) -> None:
    import json

    (store.home / PARAMS_FILE).write_text(json.dumps(params, indent=2))


def register(app: typer.Typer) -> None:
    @app.command()
    def cluster(
        home: Optional[Path] = typer.Option(None, "--home", help="TASTE_HOME (default ~/.taste)."),
        threshold: Optional[float] = typer.Option(
            None, "--threshold", min=0.0, max=2.0,
            help="Cosine distance where merging stops. Lower = more, tighter clusters. "
                 f"Default: the last value used in this home, else {DEFAULT_THRESHOLD}."),
        min_size: Optional[int] = typer.Option(
            None, "--min-size", min=1,
            help=f"Smallest cluster worth keeping. Default: last used, else {DEFAULT_MIN_SIZE}."),
        fresh_ids: bool = typer.Option(
            False, "--fresh-ids",
            help="Renumber by size instead of keeping ids from the last run."),
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
        last = read_params(store)
        threshold = threshold if threshold is not None else last.get("threshold",
                                                                    DEFAULT_THRESHOLD)
        min_size = min_size if min_size is not None else last.get("min_size", DEFAULT_MIN_SIZE)
        try:
            model = get_embedder(settings).name
            missing = len(set(store.pin_domains()) - set(store.read_embeddings(model)[0]))
            result = cluster_store(store, model, threshold=threshold, min_size=min_size,
                                   keep_ids=not fresh_ids)
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)
        write_params(store, threshold=threshold, min_size=min_size)

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
        if result.kept_ids or result.retired_ids:
            console.print(f"Kept {len(result.kept_ids)} ids from the last run"
                          + (f"; retired {', '.join(result.retired_ids)}"
                             if result.retired_ids else "") + ".")
        console.print(f"Wrote {len(result.clusters)} clusters to {store.clusters_path}")
