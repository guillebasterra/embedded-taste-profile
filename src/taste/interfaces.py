"""Protocols for the two model dependencies, plus deterministic fakes for tests.

SHARED CONTRACT FILE: owned by the orchestrator. Do not edit in a feature branch.

Real implementations live in the feature packages:
    taste.embed.get_embedder(settings) -> Embedder      (module B)
    taste.distill.llm.get_vlm(settings) -> VLM          (module C)
Both factories must return the fakes below when settings say "fake".
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, TypeVar, get_args, get_origin

import numpy as np
from PIL import Image
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class Embedder(Protocol):
    name: str  # used as the embeddings directory key
    dim: int

    def embed_images(self, paths: list[Path]) -> np.ndarray:
        """float32 (len(paths), dim), rows L2-normalized."""

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """float32 (len(texts), dim), rows L2-normalized, same space as images."""


class VLM(Protocol):
    name: str

    def generate(self, prompt: str, schema: type[T], images: list[Path] | None = None,
                 system: str | None = None) -> T:
        """Return a validated instance of `schema`."""


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

_COLORS = {
    "black": (10, 10, 10), "dark": (10, 10, 10), "white": (245, 245, 245),
    "cream": (240, 228, 200), "warm": (240, 228, 200), "red": (200, 30, 30),
    "green": (30, 160, 60), "blue": (30, 60, 200), "yellow": (230, 210, 40),
}


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return (v / np.where(n == 0, 1, n)).astype(np.float32)


class FakeEmbedder:
    """Color-layout embedder: a 4x4 grid of mean RGB, centered. Deterministic, no weights.

    Text containing a known color word embeds near images of that color, so retrieval
    is testable end to end.
    """

    name = "fake"
    dim = 48

    def _from_rgb_grid(self, grid: np.ndarray) -> np.ndarray:
        return (grid.astype(np.float32) / 255.0 - 0.5).reshape(-1)

    def embed_images(self, paths: list[Path]) -> np.ndarray:
        out = []
        for p in paths:
            with Image.open(p) as im:
                grid = np.asarray(im.convert("RGB").resize((4, 4), Image.BILINEAR))
            out.append(self._from_rgb_grid(grid))
        return _normalize(np.stack(out)) if out else np.zeros((0, self.dim), np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            words = t.lower().split()
            hit = next((c for w in words if (c := _COLORS.get(w))), None)
            if hit:
                grid = np.tile(np.array(hit, dtype=np.uint8), (4, 4, 1))
                out.append(self._from_rgb_grid(grid))
            else:
                seed = int.from_bytes(hashlib.sha1(t.encode()).digest()[:4], "big")
                out.append(np.random.default_rng(seed).normal(size=self.dim))
        return _normalize(np.stack(out)) if out else np.zeros((0, self.dim), np.float32)


def _fake_value(annotation, field_name: str, depth: int = 0):
    origin = get_origin(annotation)
    args = get_args(annotation)
    if annotation is str:
        return f"fake {field_name.replace('_', ' ')}"
    if annotation is int:
        return 1
    if annotation is float:
        return 0.5
    if annotation is bool:
        return True
    if annotation is datetime:
        return datetime(2026, 1, 1, tzinfo=timezone.utc)
    if origin is list:
        return [_fake_value(args[0], field_name, depth + 1) for _ in range(2)]
    if origin is dict:
        return {}
    if origin is not None and type(None) in args:  # Optional[X]
        return None
    if origin is not None and args and all(isinstance(a, str) for a in args):  # Literal
        return args[0]
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _fake_instance(annotation, depth + 1)
    return None


def _fake_instance(schema: type[T], depth: int = 0) -> T:
    data = {}
    for name, field in schema.model_fields.items():
        if not field.is_required() and depth > 0:
            continue
        data[name] = _fake_value(field.annotation, name, depth)
    return schema.model_validate(data)


class FakeVLM:
    """Returns a schema-valid placeholder and records every call for assertions."""

    name = "fake"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, prompt: str, schema: type[T], images: list[Path] | None = None,
                 system: str | None = None) -> T:
        self.calls.append({"prompt": prompt, "schema": schema, "images": images or [],
                           "system": system})
        return _fake_instance(schema)
