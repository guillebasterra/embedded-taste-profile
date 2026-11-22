import re

import numpy as np
import pytest
from typer.testing import CliRunner

from taste.cluster import build_clusters, cluster_store
from taste.cluster.core import group_domain, palette_for
from taste.embed.index import embed_pins
from taste.fixtures import GROUPS, make_fixture_home, make_image
from taste.interfaces import FakeEmbedder
from taste.models import Pin

# FakeEmbedder groups are ~0 apart inside and >0.8 apart across, so anything in between works.
FAKE_THRESHOLD = 0.3


@pytest.fixture
def store(tmp_path):
    s = make_fixture_home(tmp_path)
    embed_pins(s, FakeEmbedder())
    return s


def _group(pin_id: str) -> str:
    return pin_id.split(":")[1].rsplit("-", 1)[0]


def test_recovers_the_five_groups(store):
    result = cluster_store(store, "fake", threshold=FAKE_THRESHOLD, min_size=4)
    clusters = store.read_clusters()
    assert clusters == result.clusters and result.unclustered == []
    assert sorted(c.id for c in clusters) == [
        "fashion-01", "photo-01", "photo-02", "web-01", "web-02"]
    domain_of = {g: d for _b, d, g, _c in GROUPS}
    seen = set()
    for c in clusters:
        groups = {_group(p) for p in c.pin_ids}
        assert len(groups) == 1 and c.size == len(c.pin_ids) == 8
        (g,) = groups
        assert domain_of[g] == c.domain
        seen.add(g)
    assert seen == set(domain_of)


def test_cluster_record_fields(store):
    clusters = cluster_store(store, "fake", threshold=FAKE_THRESHOLD).clusters
    by_id = {c.id: c for c in clusters}
    ids, vecs = store.read_embeddings("fake")
    row = {p: i for i, p in enumerate(ids)}
    for c in clusters:
        cen = np.array(c.centroid)
        assert abs(np.linalg.norm(cen) - 1) < 1e-4
        assert c.cohesion > 0.99
        assert c.medoid_pin_id in c.pin_ids
        assert len(c.exemplar_pin_ids) == 8 and set(c.exemplar_pin_ids) == set(c.pin_ids)
        sims = [float(vecs[row[p]] @ cen) for p in c.exemplar_pin_ids]
        assert all(a >= b - 1e-5 for a, b in zip(sims, sims[1:]))  # nearest first
        assert 1 <= len(c.palette) <= 6
        assert all(re.fullmatch(r"#[0-9a-f]{6}", h) for h in c.palette)
        assert 1 <= len(c.neighbor_cluster_ids) <= 3 and c.id not in c.neighbor_cluster_ids
    # same-domain neighbor comes first
    assert by_id["web-01"].neighbor_cluster_ids[0] == "web-02"
    assert by_id["photo-02"].neighbor_cluster_ids[0] == "photo-01"
    assert by_id["fashion-01"].top_boards == ["fits"]
    # palette of the red cluster is red
    r, g, b = (int(by_id["fashion-01"].palette[0][i:i + 2], 16) for i in (1, 3, 5))
    assert r > 150 and g < 80 and b < 80


def test_default_threshold_works_on_fixture(store):
    assert len(cluster_store(store, "fake").clusters) == 5


def test_ids_ordered_by_size_and_small_domain_skipped(tmp_path):
    store = make_fixture_home(tmp_path)
    rng = np.random.default_rng(3)
    extra = []
    for i in range(3):  # blue becomes the biggest photo cluster
        make_image(store.abs(f"images/blue-x{i}.jpg"), (30, 60, 200), rng)
        extra.append(Pin(id=f"local:blue-x{i}", source="local", board="film photo",
                         image_path=f"images/blue-x{i}.jpg"))
    for i in range(2):  # a board with no boards.toml entry -> domain "other", too small
        make_image(store.abs(f"images/misc-{i}.jpg"), (230, 210, 40), rng)
        extra.append(Pin(id=f"local:misc-{i}", source="local", board="misc",
                         image_path=f"images/misc-{i}.jpg"))
    store.upsert_pins(extra)
    embed_pins(store, FakeEmbedder())
    result = cluster_store(store, "fake", threshold=FAKE_THRESHOLD)
    by_id = {c.id: c for c in result.clusters}
    assert by_id["photo-01"].size == 11 and _group(by_id["photo-01"].pin_ids[0]) == "blue"
    assert by_id["photo-02"].size == 8
    assert not any(c.domain == "other" for c in result.clusters)
    other = next(d for d in result.domains if d.domain == "other")
    assert other.skipped and sorted(result.unclustered) == ["local:misc-0", "local:misc-1"]


def _blobs(spec, dim=16, seed=0):
    """Unit vectors: `spec` is a list of (axis, n, second_axis, mix)."""
    rng = np.random.default_rng(seed)
    rows = []
    for axis, n, other, mix in spec:
        for _ in range(n):
            v = rng.normal(0, 0.01, dim)
            v[axis] += 1 - mix
            v[other] += mix
            rows.append(v / np.linalg.norm(v))
    return np.array(rows, dtype=np.float32)


def test_small_cluster_merged_when_close_else_unclustered():
    # 6 pins on axis 0, 6 on axis 1, 2 pins tilted off axis 0 (close), 2 on axis 5 (far)
    x = _blobs([(0, 6, 0, 0.0), (1, 6, 1, 0.0), (0, 2, 2, 0.47), (5, 2, 5, 0.0)])
    dist = np.clip(1 - x @ x.T, 0, 2).astype(np.float64)
    tilted_gap = dist[12:14, :6].mean()
    threshold = tilted_gap / 1.1  # above the cut, inside the merge slack
    groups, leftover = group_domain(dist, threshold, min_size=4)
    assert sorted(len(g) for g in groups) == [6, 8]
    assert sorted(leftover) == [14, 15]
    big = max(groups, key=len)
    assert {12, 13} <= set(big.tolist()) and set(range(6)) <= set(big.tolist())

    ids = [f"p{i:02d}" for i in range(len(x))]
    result = build_clusters(ids, x, {i: "art" for i in ids}, threshold=threshold, min_size=4)
    assert [c.id for c in result.clusters] == ["art-01", "art-02"]
    assert [c.size for c in result.clusters] == [8, 6]
    assert result.unclustered == ["p14", "p15"]
    assert result.clusters[0].palette == []  # no images given


def test_stale_embeddings_are_ignored(store):
    pins = [p for p in store.read_pins() if not p.id.startswith("local:red")]
    store.write_pins(pins)  # embeddings still hold the red pins
    result = cluster_store(store, "fake", threshold=FAKE_THRESHOLD)
    assert len(result.clusters) == 4
    assert not any(p.startswith("local:red") for c in result.clusters for p in c.pin_ids)


def test_palette_merges_near_duplicates(tmp_path):
    rng = np.random.default_rng(0)
    make_image(tmp_path / "a.jpg", (200, 30, 30), rng)
    make_image(tmp_path / "b.jpg", (30, 60, 200), rng)
    palette = palette_for([tmp_path / "a.jpg", tmp_path / "b.jpg", tmp_path / "missing.jpg"])
    assert len(palette) == 2


def test_cluster_requires_embeddings(tmp_path):
    store = make_fixture_home(tmp_path, n_per_group=4)
    with pytest.raises(RuntimeError, match="taste embed"):
        cluster_store(store, "fake")


def test_cluster_cli(tmp_path, monkeypatch):
    from taste.cli import app

    monkeypatch.setenv("TASTE_EMBEDDER", "fake")
    monkeypatch.setenv("COLUMNS", "200")
    store = make_fixture_home(tmp_path)
    runner = CliRunner()
    missing = runner.invoke(app, ["cluster", "--home", str(tmp_path)])
    assert missing.exit_code == 1 and "taste embed" in missing.output
    assert runner.invoke(app, ["embed", "--home", str(tmp_path)]).exit_code == 0
    result = runner.invoke(app, ["cluster", "--home", str(tmp_path), "--threshold", "0.3",
                                 "--min-size", "4"])
    assert result.exit_code == 0, result.output
    for cid in ("web-01", "web-02", "photo-01", "photo-02", "fashion-01"):
        assert cid in result.output
    assert len(store.read_clusters()) == 5
