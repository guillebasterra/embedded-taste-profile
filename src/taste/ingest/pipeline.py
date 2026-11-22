"""Glue between sources, downloads and the Store. Keeps re-runs incremental."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from taste.ingest.download import download_images
from taste.models import Pin
from taste.store import Store

STATE_FILE = "ingest_state.json"  # boards finished by an API run that hit the rate limit
SOURCE_FILE = "ingest_source.json"  # how the last successful ingest ran, for `taste update`


@dataclass
class PersistStats:
    seen: int = 0
    new: int = 0
    downloaded: int = 0
    already: int = 0  # image was on disk already
    failed: int = 0
    no_image: int = 0
    total: int = 0  # pins in the store afterwards
    errors: list[str] = field(default_factory=list)

    def add(self, other: "PersistStats") -> None:
        for name in ("seen", "new", "downloaded", "already", "failed", "no_image"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.total = other.total or self.total
        self.errors.extend(other.errors)


def persist(store: Store, pins: list[Pin], http: httpx.Client | None, *,
            fallbacks: dict[str, list[str]] | None = None, concurrency: int = 8,
            sleep=None) -> PersistStats:
    """Upsert pins, downloading only the images that are not on disk yet."""
    stats = PersistStats(seen=len(pins))
    if not pins:
        stats.total = len(store.read_pins())
        return stats
    existing = {p.id: p for p in store.read_pins()}
    ready: list[Pin] = []
    todo: list[Pin] = []
    for pin in pins:
        old = existing.get(pin.id)
        if old is None:
            stats.new += 1
        if pin.image_path and store.abs(pin.image_path).exists():
            ready.append(pin)
        elif old and old.image_path and store.abs(old.image_path).exists():
            stats.already += 1
            ready.append(pin.model_copy(update={
                "image_path": old.image_path, "width": old.width, "height": old.height}))
        elif pin.image_url and http is not None:
            todo.append(pin)
        else:
            stats.no_image += 1
            ready.append(pin)
    kwargs = {"sleep": sleep} if sleep is not None else {}
    for res in download_images(todo, store, http, concurrency=concurrency,  # type: ignore[arg-type]
                               fallbacks=fallbacks, **kwargs):
        if res.status == "downloaded":
            stats.downloaded += 1
        elif res.status == "skipped":
            stats.already += 1
        else:
            stats.failed += 1
            stats.errors.append(f"{res.pin.id}: {res.error}")
        ready.append(res.pin)
    stats.total = store.upsert_pins(ready)
    return stats


# ---- resume state for rate-limited API runs ---------------------------------------

def _state_path(store: Store) -> Path:
    return store.home / STATE_FILE


def read_done_boards(store: Store) -> set[str]:
    path = _state_path(store)
    if not path.exists():
        return set()
    try:
        return {str(b) for b in json.loads(path.read_text()).get("done_board_ids", [])}
    except (ValueError, AttributeError):
        return set()


def write_done_boards(store: Store, board_ids: set[str]) -> None:
    store.home.mkdir(parents=True, exist_ok=True)
    _state_path(store).write_text(json.dumps({"done_board_ids": sorted(board_ids)}, indent=2))


def clear_done_boards(store: Store) -> None:
    _state_path(store).unlink(missing_ok=True)


# ---- last source, replayed by `taste update` ----------------------------------------

def write_last_source(store: Store, source: dict) -> None:
    store.home.mkdir(parents=True, exist_ok=True)
    (store.home / SOURCE_FILE).write_text(json.dumps(source, indent=2))


def read_last_source(store: Store) -> dict | None:
    try:
        data = json.loads((store.home / SOURCE_FILE).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("source") else None
