"""CLI commands for this package: serve, feedback, install."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

HomeOpt = typer.Option(None, "--home", envvar="TASTE_HOME",
                       help="Taste home directory (default ~/.taste).")


def _store(home: Optional[Path]):
    from taste.config import load_settings
    from taste.store import Store

    return Store(load_settings(home).home)


def register(app: typer.Typer) -> None:
    @app.command()
    def serve(home: Optional[Path] = HomeOpt) -> None:
        """Run the MCP server over stdio (for Claude Code, Claude Desktop and other agents)."""
        from taste.serve.server import run_stdio

        run_stdio(home)

    feedback_app = typer.Typer(help="Log and list visual corrections.", no_args_is_help=True)
    app.add_typer(feedback_app, name="feedback")

    @feedback_app.command("add")
    def feedback_add(
        verdict: str = typer.Argument(..., help="like, dislike or rule"),
        note: str = typer.Argument(..., help="What was right or wrong, in your words."),
        image: Optional[Path] = typer.Option(None, "--image", help="Image the note is about."),
        domain: Optional[str] = typer.Option(None, "--domain", help="web, photo, art, ..."),
        context: Optional[str] = typer.Option(None, "--context", help="Project or task."),
        home: Optional[Path] = HomeOpt,
    ) -> None:
        """Log one correction: taste feedback add dislike "too much padding"."""
        from taste.serve import tools

        store = _store(home)
        try:
            fb = tools.log_feedback(store, verdict, note, image, domain, context)
        except tools.ToolInputError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(2) from None
        typer.echo(tools.feedback_confirmation(store, fb))

    @feedback_app.command("list")
    def feedback_list(home: Optional[Path] = HomeOpt) -> None:
        """Show logged feedback, oldest first."""
        from taste.serve import tools

        rows = tools.list_feedback(_store(home))
        if not rows:
            typer.echo('No feedback yet. Add some with: taste feedback add <verdict> "<note>"')
            return
        for fb in rows:
            extras = [x for x in (fb.domain, fb.context, fb.image_path) if x]
            tail = f"  [{' | '.join(extras)}]" if extras else ""
            typer.echo(f"{fb.ts:%Y-%m-%d %H:%M}  {fb.id}  {fb.verdict:<7}  {fb.note}{tail}")

    @app.command()
    def install(
        home: Optional[Path] = HomeOpt,
        print_mcp_config: bool = typer.Option(
            False, "--print-mcp-config",
            help="Print only the mcpServers JSON block for Claude Code / Claude Desktop."),
        export: Optional[Path] = typer.Option(
            None, "--export", help="Copy profile/ to this directory (e.g. a context repo)."),
    ) -> None:
        """Write SKILL.md and CLAUDE.snippet.md into profile/, and wire up agents."""
        from taste.serve import install as inst

        store = _store(home)
        if print_mcp_config:
            typer.echo(inst.mcp_config_json(store.home))
            return

        for path in inst.write_install_files(store):
            typer.echo(f"Wrote {path}")
        if export is not None:
            try:
                dest = inst.export_profile(store, export)
            except ValueError as exc:
                typer.echo(f"Error: {exc}", err=True)
                raise typer.Exit(2) from None
            typer.echo(f"Exported profile to {dest}")
        if not (store.profile_dir / "TASTE.md").exists():
            from taste.serve.tools import next_command

            typer.echo(f"Note: the profile is not built yet. Next: {next_command(store)}")
        typer.echo("Next steps:")
        typer.echo(f"  MCP config:   taste install --print-mcp-config --home {store.home}")
        typer.echo(f"  Claude Code:  claude mcp add taste -- taste serve --home {store.home}")
        typer.echo(f"  Skill:        copy {store.profile_dir / 'SKILL.md'} to "
                   "~/.claude/skills/taste/SKILL.md")
        typer.echo(f"  CLAUDE.md:    paste or import {store.profile_dir / 'CLAUDE.snippet.md'}")
