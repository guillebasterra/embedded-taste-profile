"""Incremental embedding index: embed pins that are new, drop ids that are gone."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from taste.interfaces import Embedder
from taste.store import Store


@dataclass
class EmbedStats:
    total: int = 0  # vectors in the index after the run
    new: int = 0
    kept: int = 0
    dropped: int = 0  # stale ids removed (pin deleted, board excluded, image gone)
    failed: list[str] = field(default_factory=list)  # pin ids whose image is missing/unreadable


def _readable(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except (OSError, ValueError, SyntaxError):
        return False


def embed_pins(store: Store, embedder: Embedder, *, chunk_size: int = 64, force: bool = False,
               save_every: int = 8,
               on_progress: Callable[[int, int], None] | None = None) -> EmbedStats:
    """Bring embeddings/<embedder.name>/ in line with the included pins. Returns counts."""
    wanted = list(store.pin_domains())  # included pins with an image, pins.jsonl order
    paths = {p.id: store.abs(p.image_path) for p in store.read_pins() if p.image_path}

    old_ids, old_vecs = store.read_embeddings(embedder.name)
    if force or old_vecs.ndim != 2 or old_vecs.shape[0] != len(old_ids):
        old_ids, old_vecs = [], np.zeros((0, 0), dtype=np.float32)

    wanted_set = set(wanted)
    keep = [i for i, pid in enumerate(old_ids) if pid in wanted_set]
    ids = [old_ids[i] for i in keep]
    vecs = old_vecs[keep] if keep else None
    have = set(ids)
    stats = EmbedStats(kept=len(ids), dropped=len(old_ids) - len(ids))

    todo: list[str] = []
    for pid in wanted:
        if pid in have:
            continue
        if _readable(paths[pid]):
            todo.append(pid)
        else:
            stats.failed.append(pid)

    def save() -> None:
        if vecs is not None:
            store.write_embeddings(embedder.name, ids, vecs)

    chunk_size = max(1, chunk_size)
    for n, start in enumerate(range(0, len(todo), chunk_size), 1):
        chunk = todo[start:start + chunk_size]
        new = np.asarray(embedder.embed_images([paths[pid] for pid in chunk]), dtype=np.float32)
        if vecs is not None and vecs.shape[0] and vecs.shape[1] != new.shape[1]:
            raise RuntimeError(
                f"Stored vectors have dim {vecs.shape[1]} but {embedder.name} produces "
                f"{new.shape[1]}. Re-run with --force."
            )
        vecs = new if vecs is None or not vecs.shape[0] else np.concatenate([vecs, new])
        ids.extend(chunk)
        stats.new += len(chunk)
        if n % save_every == 0:
            save()  # checkpoint: an interrupted run resumes from here
        if on_progress:
            on_progress(stats.new, len(todo))

    if stats.new or stats.dropped:
        save()
    stats.total = len(ids)
    return stats
