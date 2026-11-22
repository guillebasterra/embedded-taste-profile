"""Scoring API. Signatures are a shared contract: module B implements, module D consumes.

Module B: replace the bodies, keep the names and signatures.
"""

from __future__ import annotations

from pathlib import Path

from taste.interfaces import Embedder
from taste.models import CritiqueResult, ExemplarHit
from taste.store import Store


def search(store: Store, embedder: Embedder, query: str, domain: str | None = None,
           k: int = 8) -> list[ExemplarHit]:
    """Text-to-image search over my pins, best first."""
    raise NotImplementedError


def critique(store: Store, embedder: Embedder, image_path: str | Path) -> CritiqueResult:
    """Score one image against my taste clusters."""
    raise NotImplementedError
