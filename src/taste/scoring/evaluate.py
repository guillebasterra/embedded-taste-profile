"""`taste eval`: can centroid similarity tell my held-out pins from images I never saved?"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from taste.interfaces import Embedder
from taste.store import Store

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


@dataclass
class EvalReport:
    auc: float
    threshold: float  # suggested accept threshold on raw centroid similarity
    tpr: float  # share of held-out pins accepted at the threshold
    fpr: float  # share of distractors accepted at the threshold
    n_train: int
    n_heldout: int
    n_distractors: int
    n_clusters: int
    heldout_mean: float
    distractor_mean: float


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def suggest_threshold(pos: np.ndarray, neg: np.ndarray) -> float:
    """Cut that maximizes TPR - FPR, moved halfway down toward the nearest rejected distractor."""
    from sklearn.metrics import roc_curve

    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    fpr, tpr, cuts = roc_curve(y, np.r_[pos, neg])
    cut = float(cuts[int(np.argmax(tpr - fpr))])
    if not np.isfinite(cut):
        return float(np.median(pos))
    below = neg[neg < cut]
    return (cut + float(below.max())) / 2 if below.size else cut


def evaluate(store: Store, embedder: Embedder, distractors: str | Path, *,
             holdout: float = 0.2, seed: int = 0, threshold: float | None = None,
             min_size: int | None = None) -> EvalReport:
    """Hold out a share of pins, rebuild clusters from the rest, score held-out vs distractors."""
    from sklearn.metrics import roc_auc_score

    from taste.cluster import DEFAULT_MIN_SIZE, DEFAULT_THRESHOLD, build_clusters

    if not 0 < holdout < 1:
        raise RuntimeError("--holdout must be between 0 and 1.")
    domains = store.pin_domains()
    ids, vectors = store.read_embeddings(embedder.name)
    keep = [i for i, pid in enumerate(ids) if pid in domains]
    if len(keep) < 5:
        raise RuntimeError("Not enough embedded pins to evaluate. Run `taste embed` first.")
    ids, vectors = [ids[i] for i in keep], np.asarray(vectors[keep], dtype=np.float32)

    paths = list_images(Path(distractors))
    if not paths:
        raise RuntimeError(f"No images found under {distractors}.")

    perm = np.random.default_rng(seed).permutation(len(ids))
    n_held = max(1, int(round(len(ids) * holdout)))
    held, train = perm[:n_held], np.sort(perm[n_held:])

    result = build_clusters(
        [ids[i] for i in train], vectors[train], domains,
        threshold=DEFAULT_THRESHOLD if threshold is None else threshold,
        min_size=DEFAULT_MIN_SIZE if min_size is None else min_size, with_palette=False,
    )
    if not result.clusters:
        raise RuntimeError("The held-in pins form no clusters. Try a higher --threshold or a "
                           "lower --min-size.")
    centroids = np.asarray([c.centroid for c in result.clusters], dtype=np.float32)

    pos = (vectors[held] @ centroids.T).max(axis=1)
    neg = (embedder.embed_images(paths) @ centroids.T).max(axis=1)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    cut = suggest_threshold(pos, neg)
    return EvalReport(
        auc=float(roc_auc_score(y, np.r_[pos, neg])), threshold=cut,
        tpr=float(np.mean(pos >= cut)), fpr=float(np.mean(neg >= cut)),
        n_train=len(train), n_heldout=len(held), n_distractors=len(neg),
        n_clusters=len(result.clusters), heldout_mean=float(pos.mean()),
        distractor_mean=float(neg.mean()),
    )
