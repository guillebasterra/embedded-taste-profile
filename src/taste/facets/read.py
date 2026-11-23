"""VLM readings of each pin's visual qualities, plus the incremental runner for both layers."""

from __future__ import annotations

import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, Field

from taste.config import Settings
from taste.interfaces import VLM
from taste.models import Pin
from taste.store import Store

from .measure import Measures, measure

FACETS_DIR = "facets"
MEASURES_FILE = "measures.jsonl"
READINGS_FILE = "readings.jsonl"
DEFAULT_READ_MODEL = "claude-haiku-5-5"  # one image per call, ~1,400 calls: the cheap tier
DEFAULT_WORKERS = 8

Medium = Literal["photo", "illustration", "painting", "drawing", "3d render", "graphic design",
                 "typography", "screenshot or UI", "collage", "product shot", "other"]


class PinReading(BaseModel):
    """How one image looks. Every field describes what is visible, not what it depicts."""

    medium: Medium
    subject: str = Field(description="What the image shows, in a short phrase. The only "
                                     "field about content.")
    style_tags: list[str] = Field(description=(
        "6 to 10 short tags for how it looks, with the subject left out: they should fit an "
        "image of something else made the same way. E.g. 'hard flash', 'flat color fields', "
        "'film grain', 'wide margins', 'chunky sans serif', 'worn denim texture'."))
    palette: str = Field(description="One sentence: the hues, how warm, how saturated, how "
                                     "many colors, and how they are used.")
    light: str = Field(description="One sentence: light source, direction, hardness, "
                                   "exposure. For flat graphics, how value is used instead.")
    texture: str = Field(description="One sentence: surfaces, materials, grain, wear, gloss.")
    line_and_shape: str = Field(description="One sentence: edges, line weight and quality, "
                                            "geometric or organic shapes.")
    composition: str = Field(description="One sentence: density, structure, symmetry, "
                                         "negative space, framing, scale.")
    typography: str = Field("none", description="One sentence on any type (classification, weight, "
                                        "case, spacing, layout), or 'none'.")
    finish: int = Field(ge=1, le=5, description="1 raw, handmade, worn or lo-fi; 3 considered "
                                               "but unfussy; 5 polished, slick, high-production.")
    era: str = Field("none", description="Period, scene or movement it draws on, if visible "
                                 "(e.g. '90s skate zine', 'Swiss modernism'), or 'none'.")
    mood: list[str] = Field(description="2 to 4 mood words.")
    distinctive: str = Field(description=(
        "One sentence: the observable choice that makes this image different from a generic "
        "image of the same subject. Something a designer could copy."))


SYSTEM = ("You catalog the visual qualities of images someone saved as taste references. "
          "Describe what you can see: concrete, observable and specific. Never use 'clean', "
          "'modern', 'aesthetic', 'stylish' or 'beautiful' without saying what makes it so.")

PROMPT = ("Image 1 is one saved pin. Read how it looks, not just what it shows. The board "
          "name and title are context only; describe the image itself.\n\n"
          "Board: {board}\nTitle: {title}")


@dataclass
class FacetStats:
    measured: int = 0
    read: int = 0
    kept: int = 0  # already had both layers
    failed: list[tuple[str, str]] = field(default_factory=list)  # (pin id, reason)


def facets_dir(store: Store) -> Path:
    return store.home / FACETS_DIR


def read_model(override: str | None = None) -> str:
    return override or os.environ.get("TASTE_FACETS_MODEL") or DEFAULT_READ_MODEL


def reading_vlm(settings: Settings, model: str) -> VLM:
    """`get_vlm` with the facets model swapped in; fake stays fake."""
    from taste.distill.llm import get_vlm

    return get_vlm(replace(settings, vlm_model=model))


def _rows(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["pin_id"]] = row  # last row wins
    return out


def load_measures(store: Store) -> dict[str, Measures]:
    return {k: Measures.model_validate(v)
            for k, v in _rows(facets_dir(store) / MEASURES_FILE).items()}


def load_readings(store: Store) -> dict[str, PinReading]:
    return {k: PinReading.model_validate(v["reading"])
            for k, v in _rows(facets_dir(store) / READINGS_FILE).items()}


def run_facets(store: Store, vlm: VLM | None, *, sample: int | None = None,
               force: bool = False, workers: int = DEFAULT_WORKERS, seed: int = 0,
               on_progress: Callable[[int, int], None] | None = None) -> FacetStats:
    """Measure and read every included pin that is missing a layer.

    `vlm=None` measures only. `sample` picks that many of the pending pins at random.
    Rows are appended as they finish, so an interrupted run keeps its work.
    """
    out_dir = facets_dir(store)
    out_dir.mkdir(parents=True, exist_ok=True)
    m_path, r_path = out_dir / MEASURES_FILE, out_dir / READINGS_FILE
    have_m = set() if force else set(_rows(m_path))
    have_r = set() if force else {k for k, v in _rows(r_path).items()
                                  if v.get("model") == getattr(vlm, "name", None)}

    included = store.pin_domains()
    pins = [p for p in store.read_pins() if p.id in included]
    pending = [p for p in pins
               if p.id not in have_m or (vlm is not None and p.id not in have_r)]
    stats = FacetStats(kept=len(pins) - len(pending))
    if sample is not None and sample < len(pending):
        pending = random.Random(seed).sample(pending, sample)

    lock = threading.Lock()  # guards the two files and the counters

    def append(path: Path, row: dict) -> None:
        with lock, path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def work(pin: Pin) -> None:
        path = store.abs(pin.image_path)
        if pin.id not in have_m:
            append(m_path, measure(pin.id, path).model_dump())
            with lock:
                stats.measured += 1
        if vlm is not None and pin.id not in have_r:
            prompt = PROMPT.format(board=pin.board, title=pin.title or "(none)")
            reading = vlm.generate(prompt, PinReading, images=[path], system=SYSTEM)
            append(r_path, {"pin_id": pin.id, "model": vlm.name,
                            "reading": reading.model_dump()})
            with lock:
                stats.read += 1

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(work, p): p for p in pending}
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc is not None:
                stats.failed.append((futures[fut].id, f"{type(exc).__name__}: {exc}"[:200]))
            done += 1
            if on_progress:
                on_progress(done, len(pending))
    return stats
