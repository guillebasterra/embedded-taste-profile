"""Concurrent image download: retries, skip-existing, Pillow verification, width/height."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx

from taste.models import Pin
from taste.store import Store, slug

_EXT = {"JPEG": "jpg", "MPO": "jpg", "PNG": "png", "WEBP": "webp", "GIF": "gif",
        "BMP": "bmp", "TIFF": "tif"}
_RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


@dataclass
class DownloadResult:
    pin: Pin
    status: str  # "downloaded" | "skipped" | "failed"
    error: str | None = None


def inspect_image(path: Path) -> tuple[str, int, int]:
    """(extension, width, height). Raises if Pillow cannot verify the file."""
    from PIL import Image

    with Image.open(path) as im:
        fmt, (width, height) = im.format or "", im.size
        im.verify()
    if width <= 0 or height <= 0:
        raise ValueError("empty image")
    return _EXT.get(fmt, fmt.lower() or "img"), width, height


def existing_image(store: Store, pin_id: str) -> Path | None:
    stem = slug(pin_id)
    for path in sorted(store.images_dir.glob(f"{stem}.*")):
        if path.suffix != ".part":
            return path
    return None


def _with_image(store: Store, pin: Pin, path: Path, width: int, height: int) -> Pin:
    return pin.model_copy(update={"image_path": store.rel(path), "width": width,
                                  "height": height})


def _fetch(http: httpx.Client, url: str, retries: int, sleep: Callable[[float], None]) -> bytes:
    last = "no attempt"
    for attempt in range(retries + 1):
        try:
            resp = http.get(url, follow_redirects=True)
        except httpx.TransportError as exc:
            last = type(exc).__name__
        else:
            if resp.status_code == 200:
                return resp.content
            last = f"HTTP {resp.status_code}"
            if resp.status_code not in _RETRY_STATUS:
                break
        if attempt < retries:
            sleep(min(0.5 * 2 ** attempt, 8.0))
    raise RuntimeError(last)


def download_one(pin: Pin, store: Store, http: httpx.Client, *, retries: int = 3,
                 sleep: Callable[[float], None] = time.sleep,
                 fallback_urls: list[str] | None = None) -> DownloadResult:
    found = existing_image(store, pin.id)
    if found is not None:
        try:
            _ext, w, h = inspect_image(found)
            return DownloadResult(_with_image(store, pin, found, w, h), "skipped")
        except Exception:
            found.unlink(missing_ok=True)  # corrupt leftover: fetch it again
    urls = [u for u in [pin.image_url, *(fallback_urls or [])] if u]
    if not urls:
        return DownloadResult(pin, "failed", "no image url")
    stem = slug(pin.id)
    part = store.images_dir / f"{stem}.part"
    error = "unknown"
    for url in urls:
        try:
            part.write_bytes(_fetch(http, url, retries, sleep))
            ext, w, h = inspect_image(part)
        except Exception as exc:
            error = f"not a valid image ({type(exc).__name__})" \
                if not isinstance(exc, RuntimeError) else str(exc)
            part.unlink(missing_ok=True)
            continue
        final = store.images_dir / f"{stem}.{ext}"
        part.replace(final)
        done = _with_image(store, pin, final, w, h)
        return DownloadResult(done.model_copy(update={"image_url": url}), "downloaded")
    return DownloadResult(pin, "failed", error)


def download_images(pins: list[Pin], store: Store, http: httpx.Client, *, concurrency: int = 8,
                    retries: int = 3, sleep: Callable[[float], None] = time.sleep,
                    fallbacks: dict[str, list[str]] | None = None) -> list[DownloadResult]:
    """Download every pin's image into images/. Results come back in input order."""
    if not pins:
        return []
    store.images_dir.mkdir(parents=True, exist_ok=True)
    fallbacks = fallbacks or {}

    def work(pin: Pin) -> DownloadResult:
        try:
            return download_one(pin, store, http, retries=retries, sleep=sleep,
                                fallback_urls=fallbacks.get(pin.id))
        except Exception as exc:  # one bad pin never sinks the batch
            return DownloadResult(pin, "failed", type(exc).__name__)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        return list(pool.map(work, pins))
