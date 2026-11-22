"""The three sources, each producing `Pin`s: Pinterest API, public board RSS, local folder."""

from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable

import httpx

from taste.ingest.errors import IngestError, RateLimitStop
from taste.ingest.pinterest import PinterestClient
from taste.models import Pin
from taste.store import Store

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


# ---- Pinterest API ----------------------------------------------------------------

def largest_image(raw_pin: dict) -> dict | None:
    """Largest entry (by area) of media.images, or None when the pin has no image."""
    media = raw_pin.get("media")
    images = media.get("images") if isinstance(media, dict) else None
    if not isinstance(images, dict):
        return None
    candidates = [v for v in images.values() if isinstance(v, dict) and v.get("url")]
    if not candidates:
        return None
    return max(candidates, key=lambda v: (v.get("width") or 0) * (v.get("height") or 0))


def _parse_dt(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def pin_from_api(raw: dict, board_name: str, board_id: str | None = None,
                 sections: dict[str, str] | None = None, section: str | None = None) -> Pin:
    image = largest_image(raw) or {}
    section_id = raw.get("board_section_id")
    return Pin(
        id=f"pin:{raw['id']}",
        source="pinterest_api",
        board=board_name,
        board_id=str(raw.get("board_id") or board_id or "") or None,
        section=section or (sections or {}).get(str(section_id)) if section_id or section else None,
        title=raw.get("title") or None,
        description=raw.get("description") or None,
        alt_text=raw.get("alt_text") or None,
        link=raw.get("link") or None,
        image_url=image.get("url"),
        width=image.get("width"),
        height=image.get("height"),
        created_at=_parse_dt(raw.get("created_at")),
    )


@dataclass
class ApiResult:
    pins: list[Pin] = field(default_factory=list)
    boards_total: int = 0
    boards_done: list[str] = field(default_factory=list)  # board ids fully listed in this run
    boards_skipped: int = 0  # finished in an earlier, interrupted run
    stopped: bool = False  # rate limit / daily cap
    retry_after: float | None = None
    limit_hit: bool = False


def fetch_api(client: PinterestClient, *, boards: list[str] | None = None,
              limit: int | None = None, skip_board_ids: frozenset[str] | set[str] = frozenset(),
              on_board: Callable[[dict, list[Pin], bool], None] | None = None) -> ApiResult:
    """Walk boards -> sections -> pins. `on_board(board, pins, complete)` fires after every
    board (also for the partial board when a rate limit interrupts), so progress can be saved."""
    result = ApiResult()
    wanted = {b.strip().lower() for b in boards or [] if b.strip()}
    try:
        all_boards = list(client.boards())
    except RateLimitStop as stop:
        result.stopped, result.retry_after = True, stop.retry_after
        return result
    if wanted:
        names = {str(b.get("name", "")).lower() for b in all_boards}
        ids = {str(b.get("id")) for b in all_boards}
        missing = sorted(w for w in wanted if w not in names and w not in ids)
        if missing:
            raise IngestError("No such board: " + ", ".join(missing))
        all_boards = [b for b in all_boards
                      if str(b.get("name", "")).lower() in wanted or str(b.get("id")) in wanted]
    result.boards_total = len(all_boards)

    for board in all_boards:
        board_id, name = str(board["id"]), str(board.get("name") or board["id"])
        if board_id in skip_board_ids:
            result.boards_skipped += 1
            continue
        room = None if limit is None else limit - len(result.pins)
        if room is not None and room <= 0:
            result.limit_hit = True
            break
        collected: dict[str, Pin] = {}
        complete = False

        def full() -> bool:
            return room is not None and len(collected) >= room

        try:
            sections = {str(s["id"]): str(s.get("name") or s["id"])
                        for s in client.sections(board_id)}
            for raw in client.board_pins(board_id):
                if full():
                    break
                collected.setdefault(f"pin:{raw['id']}",
                                     pin_from_api(raw, name, board_id, sections))
            for section_id, section_name in sections.items():
                if full():
                    break
                for raw in client.section_pins(board_id, section_id):
                    if full():
                        break
                    # the section listing is authoritative for the section name
                    collected[f"pin:{raw['id']}"] = pin_from_api(
                        raw, name, board_id, section=section_name)
            complete = not full()
            result.limit_hit = result.limit_hit or full()
        except RateLimitStop as stop:
            result.stopped, result.retry_after = True, stop.retry_after
        pins = list(collected.values())
        result.pins.extend(pins)
        if complete:
            result.boards_done.append(board_id)
        if on_board:
            on_board(board, pins, complete)
        if result.stopped or result.limit_hit:
            break
    return result


# ---- public board (RSS) -----------------------------------------------------------

_PIN_ID = re.compile(r"/pin/(?:[^/]*--)?(\d+)")
_IMG_SRC = re.compile(r"<img[^>]+src=[\"']([^\"']+)[\"']", re.I)
_TAGS = re.compile(r"<[^>]+>")
_SIZE_SEGMENT = re.compile(r"(//i\.pinimg\.com)/\d+x\d*/")


def rss_url(board_url: str) -> str:
    url = board_url.strip().split("?")[0].split("#")[0].rstrip("/")
    if not re.match(r"https?://([a-z0-9-]+\.)?pinterest\.[a-z.]+/[^/]+/[^/]+$", url, re.I):
        raise IngestError("Expected a board URL like https://www.pinterest.com/<user>/<board>/")
    return url if url.endswith(".rss") else f"{url}.rss"


def parse_board_rss(xml_text: str, board_fallback: str) -> tuple[list[Pin], dict[str, list[str]]]:
    """Pins from a board RSS feed, plus per-pin fallback image URLs (the feed's own small
    thumbnail, used when the upsized URL does not exist)."""
    import html
    import xml.etree.ElementTree as ET

    if "<!DOCTYPE" in xml_text or "<!ENTITY" in xml_text:  # no entity-expansion tricks
        raise IngestError("The board feed is not plain RSS.")
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        raise IngestError("The board feed could not be parsed (is the board public?).") from None
    channel = root.find("channel")
    if channel is None:
        raise IngestError("The board feed has no channel (is the board public?).")
    board = (channel.findtext("title") or "").strip() or board_fallback
    pins: list[Pin] = []
    fallbacks: dict[str, list[str]] = {}
    for item in channel.findall("item"):
        link = (item.findtext("link") or item.findtext("guid") or "").strip()
        match = _PIN_ID.search(link)
        if not match:
            continue
        desc_html = item.findtext("description") or ""
        img = _IMG_SRC.search(desc_html)
        thumb = html.unescape(img.group(1)) if img else None
        big = _SIZE_SEGMENT.sub(r"\1/736x/", thumb) if thumb else None
        text = " ".join(html.unescape(_TAGS.sub(" ", desc_html)).split()) or None
        created = None
        try:
            created = parsedate_to_datetime(item.findtext("pubDate") or "")
        except (TypeError, ValueError):
            pass
        pin = Pin(id=f"pub:{match.group(1)}", source="public_board", board=board,
                  title=(item.findtext("title") or "").strip() or None, description=text,
                  link=link, image_url=big, created_at=created)
        if pin.id in fallbacks or any(p.id == pin.id for p in pins):
            continue
        pins.append(pin)
        if thumb and thumb != big:
            fallbacks[pin.id] = [thumb]
    return pins, fallbacks


def fetch_public(http: httpx.Client, board_url: str,
                 limit: int | None = None) -> tuple[list[Pin], dict[str, list[str]]]:
    """Best effort and partial: the RSS feed only carries a board's latest pins."""
    feed = rss_url(board_url)
    try:
        resp = http.get(feed, follow_redirects=True, headers={"User-Agent": "taste/0.1"})
    except httpx.HTTPError as exc:
        raise IngestError(f"Could not fetch the board feed ({type(exc).__name__}).") from None
    if resp.status_code != 200:
        raise IngestError(f"The board feed returned HTTP {resp.status_code} "
                          "(private or misspelled board?).")
    fallback_name = feed[: -len(".rss")].rsplit("/", 1)[-1]
    pins, fallbacks = parse_board_rss(resp.text, fallback_name)
    if limit is not None:
        pins = pins[:limit]
    return pins, {k: v for k, v in fallbacks.items() if any(p.id == k for p in pins)}


# ---- local folder -----------------------------------------------------------------

@dataclass
class LocalResult:
    pins: list[Pin] = field(default_factory=list)
    copied: int = 0
    already: int = 0
    invalid: list[str] = field(default_factory=list)


def fetch_local(store: Store, root: str | Path, limit: int | None = None) -> LocalResult:
    """Every image under `root`, copied into images/. Board = first-level subfolder (the root
    folder's own name for loose files); a second-level subfolder becomes the section."""
    from taste.ingest.download import inspect_image

    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise IngestError(f"Not a directory: {root}")
    store.images_dir.mkdir(parents=True, exist_ok=True)
    result = LocalResult()
    seen: set[str] = set()
    files = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
                   and not any(part.startswith(".") for part in p.relative_to(root).parts))
    for path in files:
        if limit is not None and len(result.pins) >= limit:
            break
        try:
            ext, width, height = inspect_image(path)
        except Exception:
            result.invalid.append(str(path.relative_to(root)))
            continue
        digest = hashlib.sha1(path.read_bytes()).hexdigest()[:16]
        pin_id = f"local:{digest}"
        if pin_id in seen:  # byte-identical duplicate: the first location wins
            continue
        seen.add(pin_id)
        parts = path.relative_to(root).parts
        dest = store.images_dir / f"local-{digest}.{ext}"
        if dest.exists():
            result.already += 1
        else:
            shutil.copyfile(path, dest)
            result.copied += 1
        result.pins.append(Pin(
            id=pin_id, source="local",
            board=parts[0] if len(parts) > 1 else root.name,
            section=parts[1] if len(parts) > 2 else None,
            title=path.stem, image_path=store.rel(dest), width=width, height=height,
        ))
    return result
