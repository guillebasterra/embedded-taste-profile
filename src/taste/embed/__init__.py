"""Embedding: the factory every module uses, plus the incremental index builder."""

from __future__ import annotations

from taste.config import Settings
from taste.interfaces import Embedder

__all__ = ["get_embedder", "EmbedderUnavailable"]


class EmbedderUnavailable(RuntimeError):
    """The requested embedder cannot be built or loaded (unknown name, missing extra)."""


def get_embedder(settings: Settings) -> Embedder:
    """Resolve the embedder named in settings. Cheap: real weights load on first use."""
    kind = (settings.embedder or "").strip().lower()
    if kind == "fake":
        from taste.interfaces import FakeEmbedder

        return FakeEmbedder()
    if kind == "siglip2":
        from taste.embed.siglip import SigLIP2Embedder

        return SigLIP2Embedder(settings.embed_model)
    raise EmbedderUnavailable(
        f"Unknown TASTE_EMBEDDER {settings.embedder!r}. Use 'siglip2' or 'fake'."
    )
