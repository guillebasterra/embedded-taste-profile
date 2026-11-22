"""The three sources, each producing `Pin`s: Pinterest API, public board RSS, local folder."""

from __future__ import annotations

import hashlib
import json
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
HEIF_SUFFIXES = {".heic", ".heif"}  # Pillow cannot read these: converted to JPEG on import
QUIET_SUFFIXES = {".json", ".part", ".sqlite3"}  # gallery-dl sidecars, partials, archive


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
    unsupported: dict[str, int] = field(default_factory=dict)  # suffix -> files skipped


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
    convert = heif_converter()
    files = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or any(part.startswith(".") for part in p.relative_to(root).parts):
            continue
        suffix = p.suffix.lower()
        if suffix in IMAGE_SUFFIXES or (suffix in HEIF_SUFFIXES and convert):
            files.append(p)
        elif suffix not in QUIET_SUFFIXES:
            key = suffix or "(none)"
            result.unsupported[key] = result.unsupported.get(key, 0) + 1
    for path in files:
        if limit is not None and len(result.pins) >= limit:
            break
        heif = path.suffix.lower() in HEIF_SUFFIXES
        try:
            ext, width, height = ("jpg", 0, 0) if heif else inspect_image(path)
        except Exception:
            result.invalid.append(str(path.relative_to(root)))
            continue
        digest = hashlib.sha1(path.read_bytes()).hexdigest()[:16]
        pin_id = f"local:{digest}"
        if pin_id in seen:  # byte-identical duplicate: the first location wins
            continue
        parts = path.relative_to(root).parts
        dest = store.images_dir / f"local-{digest}.{ext}"
        if dest.exists():
            result.already += 1
        else:
            try:
                (convert if heif else shutil.copyfile)(path, dest)  # type: ignore[operator]
            except Exception:
                dest.unlink(missing_ok=True)
                result.invalid.append(str(path.relative_to(root)))
                continue
            result.copied += 1
        if heif:
            _, width, height = inspect_image(dest)
        seen.add(pin_id)
        meta = gallery_dl_metadata(path)
        result.pins.append(Pin(
            id=pin_id, source="local",
            board=parts[0] if len(parts) > 1 else root.name,
            board_id=meta.get("board_id"),
            section=parts[1] if len(parts) > 2 else None,
            title=meta.get("title") or path.stem, description=meta.get("description"),
            alt_text=meta.get("alt_text"), link=meta.get("link"),
            image_url=meta.get("image_url"), created_at=meta.get("created_at"),
            image_path=store.rel(dest), width=width, height=height,
        ))
    return result


def heif_converter() -> Callable[[Path, Path], None] | None:
    """HEIC/HEIF -> JPEG with macOS `sips` (no extra Python dependency). None elsewhere."""
    sips = shutil.which("sips")
    if not sips:
        return None

    def convert(src: Path, dest: Path) -> None:
        import subprocess

        subprocess.run([sips, "-s", "format", "jpeg", "-s", "formatOptions", "92",
                        str(src), "--out", str(dest)],
                       check=True, capture_output=True, timeout=60)

    return convert


def gallery_dl_metadata(image: Path) -> dict:
    """Pin fields from a gallery-dl `--write-metadata` sidecar (`<image>.json`), if any.
    Without it, local pins only have their file name, which is noise to the VLM."""
    sidecar = image.with_name(image.name + ".json")
    try:
        raw = json.loads(sidecar.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}

    def text(*keys: str) -> str | None:
        for key in keys:
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    board = raw.get("board") if isinstance(raw.get("board"), dict) else {}
    return {k: v for k, v in {
        "board_id": str(board["id"]) if board.get("id") else None,
        "title": text("title", "grid_title"),
        "description": text("description", "unauth_on_page_description"),
        "alt_text": text("alt_text", "auto_alt_text", "seo_alt_text"),
        "link": text("link"),
        "image_url": text("url"),
        "created_at": _parse_dt(raw.get("created_at")) or _parse_rfc2822(raw.get("created_at")),
    }.items() if v}


def _parse_rfc2822(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None


# ---- gallery-dl (unofficial, logged-in or public scrape into a local folder) -------------

GALLERY_DL_DIR = "gallery-dl"  # under TASTE_HOME; kept so reruns only fetch new pins
_PROFILE_URL = re.compile(r"^https?://([a-z0-9-]+\.)?pinterest\.[a-z.]+/([^/?#]+)", re.I)


def gallery_dl_command() -> list[str]:
    """`gallery-dl` when installed, else through `uvx` (no dependency of ours)."""
    if exe := shutil.which("gallery-dl"):
        return [exe]
    if uvx := shutil.which("uvx"):
        return [uvx, "gallery-dl"]
    raise IngestError("gallery-dl is not installed. Install it (`uv tool install gallery-dl`) "
                      "or install uv so `uvx gallery-dl` works.")


def gallery_dl_root(store: Store, url: str) -> Path:
    """Folder gallery-dl writes this profile's boards to: <home>/gallery-dl/pinterest/<user>."""
    match = _PROFILE_URL.match(url.strip())
    if not match or match.group(2).lower() in {"pin", "search", "ideas"}:
        raise IngestError("Expected a profile or board URL like "
                          "https://www.pinterest.com/<user>/")
    return store.home / GALLERY_DL_DIR / "pinterest" / match.group(2)


def run_gallery_dl(store: Store, url: str, browser: str | None = None,
                   run: Callable[..., object] | None = None) -> tuple[Path, int]:
    """Download (new) pins with gallery-dl: images without videos, a metadata sidecar per
    image, and a download archive so reruns skip what is on disk. Returns (folder, exit code).
    gallery-dl exits non-zero when any single request fails, so the caller only warns."""
    import subprocess

    root = gallery_dl_root(store, url)
    base = store.home / GALLERY_DL_DIR
    base.mkdir(parents=True, exist_ok=True)
    cmd = [*gallery_dl_command(), "-o", "videos=false", "--write-metadata",
           "-d", str(base), "--download-archive", str(base / "archive.sqlite3")]
    if browser:
        cmd += ["--cookies-from-browser", browser]
    cmd.append(url.strip())
    proc = (run or subprocess.run)(cmd, check=False)
    return root, int(getattr(proc, "returncode", 0) or 0)
