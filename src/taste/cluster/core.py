"""Per-domain agglomerative clustering over pin embeddings.

Cosine distance, average linkage, cut at a distance threshold, so the number of tastes per
domain comes from the data. Clusters under `min_size` are merged into the nearest full-size
cluster when close enough, otherwise their pins stay unclustered.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from taste.config import DOMAINS
from taste.models import ClusterRecord, Pin
from taste.store import Store

# Cosine distance at which average linkage stops merging. Chosen for SigLIP 2 image
# embeddings, where pins inside one domain already sit fairly close together; see DECISIONS.md.
DEFAULT_THRESHOLD = 0.35
DEFAULT_MIN_SIZE = 4
# A small cluster joins the nearest full-size one if their average-linkage distance is within
# threshold * MERGE_SLACK (it has to be above the threshold, or they would already be one).
MERGE_SLACK = 1.25
MAX_EXEMPLARS = 12
MAX_NEIGHBORS = 3
MAX_PALETTE = 6
MAX_TOP_BOARDS = 3
_PALETTE_SIDE = 32  # exemplars are downsampled to this many pixels per side
_PALETTE_MERGE_DIST = 28.0  # RGB distance under which two palette colors count as one


@dataclass
class DomainReport:
    domain: str
    n_pins: int
    n_clusters: int = 0
    n_unclustered: int = 0
    skipped: bool = False  # fewer than min_size pins
    median_distance: float | None = None  # median pairwise cosine distance, for tuning


@dataclass
class ClusterResult:
    clusters: list[ClusterRecord] = field(default_factory=list)
    unclustered: list[str] = field(default_factory=list)
    domains: list[DomainReport] = field(default_factory=list)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return (v / np.where(n == 0, 1, n)).astype(np.float32)


def _distances(x: np.ndarray) -> np.ndarray:
    d = np.clip(1.0 - x @ x.T, 0.0, 2.0).astype(np.float64)
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0.0)
    return d


def group_domain(dist: np.ndarray, threshold: float,
                 min_size: int) -> tuple[list[np.ndarray], list[int]]:
    """Cluster one domain from its distance matrix. Returns (member index arrays, leftovers)."""
    n = dist.shape[0]
    if n == 1:
        labels = np.zeros(1, dtype=int)
    else:
        from sklearn.cluster import AgglomerativeClustering

        labels = AgglomerativeClustering(
            n_clusters=None, metric="precomputed", linkage="average",
            distance_threshold=threshold,
        ).fit(dist).labels_
    groups = [np.flatnonzero(labels == lab) for lab in np.unique(labels)]
    big = [g for g in groups if len(g) >= min_size]
    small = sorted((g for g in groups if len(g) < min_size), key=lambda g: (-len(g), g[0]))
    leftover: list[int] = []
    for g in small:
        if big:
            gaps = [float(dist[np.ix_(g, b)].mean()) for b in big]
            j = int(np.argmin(gaps))
            if gaps[j] <= threshold * MERGE_SLACK:
                big[j] = np.concatenate([big[j], g])
                continue
        leftover.extend(int(i) for i in g)
    return big, leftover


def palette_for(paths: list[Path], max_colors: int = MAX_PALETTE) -> list[str]:
    """Dominant colors of a set of images as hex, most common first."""
    from PIL import Image

    pixels = []
    for p in paths:
        try:
            with Image.open(p) as im:
                small = im.convert("RGB").resize((_PALETTE_SIDE, _PALETTE_SIDE), Image.BILINEAR)
            pixels.append(np.asarray(small, dtype=np.float32).reshape(-1, 3))
        except (OSError, ValueError):
            continue
    if not pixels:
        return []
    px = np.concatenate(pixels)
    k = min(max_colors, len(np.unique(px, axis=0)))
    if k <= 1:
        centers, counts = px[:1], np.array([len(px)])
    else:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(px)
        centers = km.cluster_centers_
        counts = np.bincount(km.labels_, minlength=k)
    # Fold near-identical colors together so a flat image does not yield six shades of one hue.
    merged: list[tuple[np.ndarray, int]] = []
    for i in np.argsort(-counts, kind="stable"):
        c, w = centers[i].astype(np.float64), int(counts[i])
        if w == 0:
            continue
        for j, (mc, mw) in enumerate(merged):
            if np.linalg.norm(mc - c) < _PALETTE_MERGE_DIST:
                merged[j] = ((mc * mw + c * w) / (mw + w), mw + w)
                break
        else:
            merged.append((c, w))
    merged.sort(key=lambda t: -t[1])
    return ["#%02x%02x%02x" % tuple(int(round(float(v))) for v in np.clip(c, 0, 255))
            for c, _ in merged[:max_colors]]


def build_clusters(ids: list[str], vectors: np.ndarray, pin_domains: dict[str, str], *,
                   pins: dict[str, Pin] | None = None, image_root: Path | None = None,
                   threshold: float = DEFAULT_THRESHOLD, min_size: int = DEFAULT_MIN_SIZE,
                   with_palette: bool = True) -> ClusterResult:
    """Cluster embedded pins per domain. Pure: nothing is read from or written to the store.

    `ids`/`vectors` are the embedding index; only ids present in `pin_domains` are used.
    `pins` (id -> Pin) supplies boards and image paths; `image_root` is TASTE_HOME (palettes).
    """
    pins = pins or {}
    min_size = max(1, int(min_size))
    result = ClusterResult()
    row = {pid: i for i, pid in enumerate(ids)}
    drafts: list[dict] = []

    for domain in DOMAINS:
        d_ids = sorted(pid for pid, dom in pin_domains.items() if dom == domain and pid in row)
        if not d_ids:
            continue
        report = DomainReport(domain=domain, n_pins=len(d_ids))
        result.domains.append(report)
        if len(d_ids) < min_size:
            report.skipped = True
            report.n_unclustered = len(d_ids)
            result.unclustered.extend(d_ids)
            continue
        x = _unit(np.asarray(vectors[[row[pid] for pid in d_ids]], dtype=np.float32))
        dist = _distances(x)
        if len(d_ids) > 1:
            report.median_distance = float(np.median(dist[np.triu_indices(len(d_ids), 1)]))
        groups, leftover = group_domain(dist, threshold, min_size)
        result.unclustered.extend(d_ids[i] for i in sorted(leftover))
        report.n_unclustered = len(leftover)
        report.n_clusters = len(groups)

        groups.sort(key=lambda g: (-len(g), min(d_ids[i] for i in g)))
        for n, g in enumerate(groups, 1):
            g = np.sort(g)
            v = x[g]
            centroid = _unit(v.mean(axis=0))
            sims = v @ centroid
            order = sorted(range(len(g)), key=lambda i: (-float(sims[i]), d_ids[g[i]]))
            member_ids = [d_ids[i] for i in g]
            medoid = member_ids[int(np.argmax((v @ v.T).mean(axis=1)))]
            exemplars = [member_ids[i] for i in order[:MAX_EXEMPLARS]]
            boards = Counter(pins[pid].board for pid in member_ids if pid in pins)
            palette: list[str] = []
            if with_palette and image_root is not None:
                palette = palette_for([Path(image_root) / pins[pid].image_path
                                       for pid in exemplars
                                       if pid in pins and pins[pid].image_path])
            drafts.append(dict(
                id=f"{domain}-{n:02d}", domain=domain, size=len(member_ids),
                pin_ids=member_ids, medoid_pin_id=medoid, exemplar_pin_ids=exemplars,
                centroid=centroid, cohesion=float(sims.mean()), palette=palette,
                top_boards=[b for b, _ in boards.most_common(MAX_TOP_BOARDS)],
            ))

    if drafts:
        cents = np.stack([d["centroid"] for d in drafts])
        sim = cents @ cents.T
        for i, d in enumerate(drafts):
            others = [j for j in range(len(drafts)) if j != i]
            # Same-domain neighbors first, then the closest clusters from other domains.
            others.sort(key=lambda j: (drafts[j]["domain"] != d["domain"], -float(sim[i, j])))
            d["neighbor_cluster_ids"] = [drafts[j]["id"] for j in others[:MAX_NEIGHBORS]]
            d["centroid"] = [round(float(c), 6) for c in d["centroid"]]
            result.clusters.append(ClusterRecord(**d))
    return result


def cluster_store(store: Store, model: str, *, threshold: float = DEFAULT_THRESHOLD,
                  min_size: int = DEFAULT_MIN_SIZE, write: bool = True) -> ClusterResult:
    """Cluster the store's pins from embeddings/<model>/ and write clusters.json."""
    ids, vectors = store.read_embeddings(model)
    if not ids:
        raise RuntimeError(f"No embeddings for {model!r}. Run `taste embed` first.")
    result = build_clusters(
        ids, vectors, store.pin_domains(), pins={p.id: p for p in store.read_pins()},
        image_root=store.home, threshold=threshold, min_size=min_size,
    )
    if write:
        store.write_clusters(result.clusters)
    return result
