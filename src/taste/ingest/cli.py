"""CLI for module A: `taste auth`, `taste ingest`, `taste domains`."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import typer

HomeOpt = typer.Option(None, "--home", help="Taste home (default: $TASTE_HOME, else ~/.taste).")


class Source(str, Enum):
    api = "api"
    public = "public"
    local = "local"


def _fail(message: str) -> "typer.Exit":
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    return typer.Exit(1)


def _format_wait(seconds: float | None) -> str:
    if seconds is None:
        return "later (the trial cap of 1,000 requests resets daily)"
    if seconds < 90:
        return f"in about {int(seconds) + 1} seconds"
    if seconds < 5400:
        return f"in about {round(seconds / 60)} minutes"
    return f"in about {round(seconds / 3600)} hours"


def auth(
    home: Optional[Path] = HomeOpt,
    no_browser: bool = typer.Option(False, "--no-browser",
                                    help="Only print the login URL; do not open a browser."),
    timeout: float = typer.Option(300.0, "--timeout", help="Seconds to wait for the redirect."),
) -> None:
    """Log in to Pinterest (OAuth) and save tokens to auth/pinterest.json."""
    import httpx

    from taste.config import PINTEREST_REDIRECT_URI, load_settings
    from taste.ingest import oauth
    from taste.ingest.errors import IngestError
    from taste.store import Store

    settings = load_settings(home)
    if settings.pinterest_access_token:
        typer.echo("PINTEREST_ACCESS_TOKEN is set, so `taste ingest` will use it. Nothing to do.")
        return
    if not settings.pinterest_app_id or not settings.pinterest_app_secret:
        raise _fail(
            "PINTEREST_APP_ID and PINTEREST_APP_SECRET are not set. Create an app at "
            "https://developers.pinterest.com/apps/, add the redirect URI "
            f"{PINTEREST_REDIRECT_URI}, then put both values in {settings.home / '.env'}.")
    store = Store(settings.home).ensure()
    state = oauth.new_state()
    url = oauth.build_auth_url(settings.pinterest_app_id, state)

    def announce(_port: int) -> None:
        typer.echo("Open this URL to authorize taste:\n")
        typer.echo(f"  {url}\n")
        if not no_browser:
            import webbrowser

            try:
                webbrowser.open(url)
            except Exception:
                pass
        typer.echo(f"Waiting for the redirect on {PINTEREST_REDIRECT_URI} ...")

    try:
        code = oauth.wait_for_callback(state, timeout=timeout, on_ready=announce)
        with httpx.Client(timeout=30) as http:
            tokens = oauth.exchange_code(http, settings.pinterest_app_id,
                                         settings.pinterest_app_secret, code)
        path = oauth.save_tokens(store, tokens)
    except IngestError as exc:
        raise _fail(str(exc)) from None
    typer.echo(f"Logged in. Tokens saved to {path} (mode 600). Next: taste ingest")


def ingest(
    home: Optional[Path] = HomeOpt,
    source: Source = typer.Option(Source.api, "--source", help="api | public | local"),
    url: Optional[str] = typer.Option(None, "--url", help="Board URL (--source public)."),
    path: Optional[Path] = typer.Option(None, "--path", help="Image folder (--source local)."),
    board: list[str] = typer.Option([], "--board",
                                    help="Only this board, by name or id (repeatable; api)."),
    limit: Optional[int] = typer.Option(None, "--limit", min=1,
                                        help="Stop after this many pins in this run."),
    concurrency: int = typer.Option(8, "--concurrency", min=1, max=32,
                                    help="Parallel image downloads."),
    max_requests: Optional[int] = typer.Option(
        None, "--max-requests", min=1,
        help="Stop after this many Pinterest API requests (stay under a daily cap)."),
) -> None:
    """Pull pins and their images into the taste home. Re-running only fetches what is new."""
    import httpx

    from taste.config import load_settings
    from taste.ingest import pipeline, sources
    from taste.ingest.errors import IngestError
    from taste.store import Store

    settings = load_settings(home)
    store = Store(settings.home).ensure()
    stats = pipeline.PersistStats()
    stopped_wait: float | None = None
    stopped = False
    try:
        if source is Source.local:
            if path is None:
                raise IngestError("--source local needs --path <dir>.")
            local = sources.fetch_local(store, path, limit)
            stats = pipeline.persist(store, local.pins, None)
            stats.downloaded, stats.already = local.copied, local.already
            stats.no_image = 0
            if local.invalid:
                typer.echo(f"Skipped {len(local.invalid)} file(s) Pillow could not read.")
        elif source is Source.public:
            if not url:
                raise IngestError("--source public needs --url <board url>.")
            with httpx.Client(timeout=30) as http:
                pins, fallbacks = sources.fetch_public(http, url, limit)
                stats = pipeline.persist(store, pins, http, fallbacks=fallbacks,
                                         concurrency=concurrency)
            typer.echo("Note: the public feed is partial. It only lists a board's latest pins; "
                       "use `taste auth` + `taste ingest` for everything.")
        else:
            from taste.ingest.oauth import TokenManager
            from taste.ingest.pinterest import PinterestClient

            done = pipeline.read_done_boards(store)
            with httpx.Client(timeout=30) as http:
                client = PinterestClient(TokenManager(settings, store, http), http,
                                         max_requests=max_requests)

                def on_board(raw_board: dict, pins: list, complete: bool) -> None:
                    part = pipeline.persist(store, pins, http, concurrency=concurrency)
                    stats.add(part)
                    if complete:
                        done.add(str(raw_board["id"]))
                    mark = "" if complete else " (partial)"
                    typer.echo(f"  {raw_board.get('name', raw_board['id'])}: {len(pins)} pins, "
                               f"{part.downloaded} downloaded{mark}")

                result = sources.fetch_api(client, boards=board or None, limit=limit,
                                           skip_board_ids=done, on_board=on_board)
            if result.boards_skipped:
                typer.echo(f"Skipped {result.boards_skipped} board(s) finished in the "
                           "interrupted run.")
            stopped, stopped_wait = result.stopped, result.retry_after
            if stopped:
                pipeline.write_done_boards(store, done)
            elif not result.limit_hit and not board:
                pipeline.clear_done_boards(store)  # full pass: next run re-checks every board
            if not stats.total:
                stats.total = len(store.read_pins())
    except IngestError as exc:
        raise _fail(str(exc)) from None

    typer.echo(f"{stats.seen} pins seen, {stats.new} new, {stats.downloaded} images saved, "
               f"{stats.already} already on disk, {stats.failed} failed. "
               f"{stats.total} pins in {store.pins_path}.")
    if stats.no_image:
        typer.echo(f"{stats.no_image} pin(s) have no image and were recorded without one.")
    for line in stats.errors[:5]:
        typer.echo(f"  failed {line}")
    if stats.failed:
        typer.echo("Re-run the same command to retry failed downloads.")
    if stopped:
        typer.echo("Stopped: Pinterest rate limit reached. Progress is saved. Run the same "
                   f"`taste ingest` command again {_format_wait(stopped_wait)} to resume where "
                   "this left off.")
        raise typer.Exit(3)
    typer.echo("Next: taste domains")


def domains(home: Optional[Path] = HomeOpt) -> None:
    """Guess a domain for every board and write boards.toml (existing entries are kept)."""
    from rich.console import Console
    from rich.table import Table

    from taste.config import load_settings
    from taste.ingest.domains import update_boards
    from taste.store import Store

    settings = load_settings(home)
    store = Store(settings.home).ensure()
    pins = store.read_pins()
    counts: dict[str, int] = {}
    for pin in pins:
        counts[pin.board] = counts.get(pin.board, 0) + 1
    boards, added = update_boards(store, list(counts))
    if not boards:
        typer.echo("No boards yet. Run `taste ingest` first.")
        return
    table = Table("board", "domain", "include", "pins", "status")
    for name, entry in sorted(boards.items()):
        table.add_row(name, entry.domain, "yes" if entry.include else "no",
                      str(counts.get(name, 0)), "new (guessed)" if name in added else "kept")
    Console().print(table)
    typer.echo(f"{len(added)} board(s) added, {len(boards) - len(added)} kept as they were.")
    typer.echo(f"Guesses come from board names. Edit {store.boards_path} to fix a domain "
               "(web, photo, art, fashion, other) or set include = false. Then: taste embed")


def register(app: typer.Typer) -> None:
    app.command("auth")(auth)
    app.command("ingest")(ingest)
    app.command("domains")(domains)
