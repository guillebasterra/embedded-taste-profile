"""Scoring API. Signatures are a shared contract: module B implements, module D consumes.

Module B: replace the bodies, keep the names and signatures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from taste.interfaces import Embedder
from taste.models import ClusterMatch, CritiqueResult, ExemplarHit
from taste.store import Store

MAX_NEAREST = 3  # cluster matches returned by critique
MAX_ANTI = 3  # disliked references returned by critique
# My own pins score uniformly between 0 and 1 by construction, so the bands sit at the tail:
ON_TASTE = 0.1  # at or above: inside the range where 90% of my own pins sit
OFF_TASTE = 0.01  # below: further out than practically all of my own pins
ANTI_MARGIN = 0.05  # a disliked reference this close to the best cluster gets a warning
TIE_MARGIN = 0.03  # two clusters this close in similarity count as a tie

# Disliked images are embedded on the fly; a long-lived process (the MCP server) reuses them.
_anti_cache: dict[tuple[str, str, int], np.ndarray] = {}


def _included_embeddings(store: Store, embedder: Embedder,
                         domain: str | None = None) -> tuple[list[str], np.ndarray]:
    """Embedding rows for pins that are still included (optionally in one domain)."""
    ids, vectors = store.read_embeddings(embedder.name)
    domains = store.pin_domains()
    keep = [i for i, pid in enumerate(ids)
            if pid in domains and (domain is None or domains[pid] == domain)]
    if not keep:
        return [], np.zeros((0, 0), dtype=np.float32)
    return [ids[i] for i in keep], np.asarray(vectors[keep], dtype=np.float32)


def percentile_score(value: float, reference: np.ndarray) -> float:
    """Share of the reference values that `value` is at least as high as."""
    if reference.size == 0:
        return 0.0
    return float(np.mean(reference <= value + 1e-6))


def _disliked_vectors(store: Store, embedder: Embedder) -> np.ndarray:
    vecs = []
    for fb in store.read_feedback():
        if fb.verdict != "dislike" or not fb.image_path:
            continue
        path = store.abs(fb.image_path)
        try:
            key = (embedder.name, str(path), path.stat().st_mtime_ns)
        except OSError:
            continue  # image no longer there
        if key not in _anti_cache:
            try:
                _anti_cache[key] = embedder.embed_images([path])[0]
            except (OSError, ValueError):
                continue  # unreadable image: skip it, keep scoring
        vecs.append(_anti_cache[key])
    return np.stack(vecs) if vecs else np.zeros((0, 0), dtype=np.float32)


def search(store: Store, embedder: Embedder, query: str, domain: str | None = None,
           k: int = 8) -> list[ExemplarHit]:
    """Text-to-image search over my pins, best first."""
    if k <= 0 or not query.strip():
        return []
    ids, vectors = _included_embeddings(store, embedder, domain)
    if not ids:
        return []
    q = embedder.embed_texts([query])[0]
    if q.shape[0] != vectors.shape[1]:
        raise RuntimeError("Stored embeddings do not match the current embedder. "
                           "Re-run `taste embed`.")
    sims = vectors @ q
    order = sorted(range(len(ids)), key=lambda i: (-float(sims[i]), ids[i]))[:k]
    pins = {p.id: p for p in store.read_pins()}
    cluster_of = {pid: c.id for c in store.read_clusters() for pid in c.pin_ids}
    return [
        ExemplarHit(pin_id=ids[i], image_path=pins[ids[i]].image_path,
                    similarity=float(sims[i]), cluster_id=cluster_of.get(ids[i]),
                    board=pins[ids[i]].board, link=pins[ids[i]].link)
        for i in order
    ]


def critique(store: Store, embedder: Embedder, image_path: str | Path) -> CritiqueResult:
    """Score one image against my taste clusters."""
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"No image at {path}")
    clusters = store.read_clusters()
    if not clusters:
        raise RuntimeError("No clusters yet. Run `taste embed` and `taste cluster` first.")
    centroids = np.asarray([c.centroid for c in clusters], dtype=np.float32)
    vec = embedder.embed_images([path])[0]
    if vec.shape[0] != centroids.shape[1]:
        raise RuntimeError("clusters.json was built with a different embedder. "
                           "Re-run `taste embed` and `taste cluster`.")

    sims = centroids @ vec
    order = np.argsort(-sims, kind="stable")
    best = clusters[int(order[0])]
    raw = float(sims[order[0]])
    names = {card.cluster_id: card.name for card in store.read_cards()}
    nearest = [ClusterMatch(cluster_id=clusters[i].id, similarity=float(sims[i]),
                            name=names.get(clusters[i].id))
               for i in order[:MAX_NEAREST]]

    # Reference distribution: how close each of my own pins sits to its nearest centroid.
    _, own = _included_embeddings(store, embedder)
    if own.size == 0 or own.shape[1] != centroids.shape[1]:
        raise RuntimeError(f"No embeddings for {embedder.name!r} to score against. "
                           "Run `taste embed`.")
    reference = (own @ centroids.T).max(axis=1)
    score = percentile_score(raw, reference)

    anti = _disliked_vectors(store, embedder)
    nearest_anti: list[float] = []
    if anti.size and anti.shape[1] == vec.shape[0]:
        nearest_anti = [float(s) for s in np.sort(anti @ vec)[::-1][:MAX_ANTI]]

    def label(m: ClusterMatch) -> str:
        return f"{m.name} ({m.cluster_id})" if m.name else m.cluster_id

    notes = [f"Closest to {label(nearest[0])} in {best.domain}, similarity {raw:.2f}."]
    pct = round(score * 100)
    if score >= ON_TASTE:
        notes.append(f"On taste: at least as close to a cluster as {pct}% of my own pins.")
    elif score >= OFF_TASTE:
        notes.append(f"Borderline: only the loosest {max(pct, 1)}% of my own pins sit this far "
                     "from their cluster.")
    else:
        notes.append("Off taste: further from every cluster than practically all of my own "
                     "pins.")
    if len(nearest) > 1 and nearest[0].similarity - nearest[1].similarity < TIE_MARGIN:
        notes.append(f"Sits between {label(nearest[0])} and {label(nearest[1])}.")
    if nearest_anti:
        if nearest_anti[0] > raw:
            notes.append(f"Closer to a disliked reference ({nearest_anti[0]:.2f}) than to any "
                         f"cluster ({raw:.2f}).")
        elif nearest_anti[0] > raw - ANTI_MARGIN:
            notes.append(f"Nearly as close to a disliked reference ({nearest_anti[0]:.2f}) as "
                         f"to its best cluster ({raw:.2f}).")

    return CritiqueResult(score=score, raw_similarity=raw, domain=best.domain, nearest=nearest,
                          nearest_anti=nearest_anti, notes=notes)
