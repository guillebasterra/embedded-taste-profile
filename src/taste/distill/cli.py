"""CLI commands for this package. Add them inside register()."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer


def register(app: typer.Typer) -> None:
    @app.command()
    def distill(
        force: bool = typer.Option(False, "--force", help="Redo every card, summary and the profile."),
        only: Optional[str] = typer.Option(
            None, "--only", metavar="CLUSTER_ID",
            help="Redo the card of this cluster only (then refresh what depends on it)."),
        render_only: bool = typer.Option(
            False, "--render-only",
            help="Rebuild all markdown from the JSON in profile/. No VLM call."),
        home: Optional[Path] = typer.Option(
            None, "--home", help="Taste home (default: TASTE_HOME, else ~/.taste)."),
    ) -> None:
        """Distill clusters into the taste profile (cards, domains, TASTE.md, DESIGN.md)."""
        from rich.console import Console

        from taste.config import load_settings
        from taste.store import Store

        from .llm import VLMError
        from .pipeline import DistillError, run_distill

        console = Console()
        settings = load_settings(home)
        store = Store(settings.home)
        if render_only and (force or only):
            console.print("[red]--render-only cannot be combined with --force or --only.[/red]")
            raise typer.Exit(2)
        try:
            report = run_distill(store, settings, force=force, only=only,
                                 render_only=render_only)
        except (DistillError, VLMError) as e:
            console.print(f"[red]{e}[/red]", soft_wrap=True)
            raise typer.Exit(1) from e

        if not render_only:
            console.print(
                f"Cards: {len(report.cards_generated)} generated, "
                f"{len(report.cards_skipped)} unchanged. "
                f"Domain summaries: {len(report.domains_generated)} generated. "
                f"Profile: {'regenerated' if report.profile_generated else 'unchanged'}.")
            console.print(f"VLM calls: {report.vlm_calls}")
        for cid in report.clusters_without_images:
            console.print(f"[yellow]Skipped {cid}: no exemplar image on disk.[/yellow]")
        if report.invariants_dropped:
            console.print(
                "[yellow]Dropped invariants without evidence in two domains: "
                f"{', '.join(report.invariants_dropped)}[/yellow]", soft_wrap=True)
        if report.pruned:
            console.print(f"Removed {len(report.pruned)} stale file(s) for clusters that no "
                          "longer exist.")
        if report.stale_feedback:
            console.print(
                f"[yellow]{report.stale_feedback} like/dislike note(s) are newer than the "
                "profile. Run `taste distill --force` to fold them in.[/yellow]",
                soft_wrap=True)
        for warning in report.warnings:
            console.print(f"[yellow]{warning}[/yellow]", soft_wrap=True)
        console.print(f"Standing corrections: {report.rules}")
        console.print(f"Rendered {len(report.files_rendered)} markdown file(s) in "
                      f"{store.profile_dir}", soft_wrap=True)
