from datetime import datetime, timezone

import pytest

from taste.fixtures import make_fixture_home
from taste.interfaces import FakeEmbedder
from taste.models import (
    ClusterCard,
    ClusterMatch,
    ClusterRecord,
    CritiqueResult,
    DomainSummary,
    ExemplarHit,
    TasteProfile,
)
from taste.serve import tools


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """No real embedder, no leaked env, fresh embedder cache."""
    monkeypatch.delenv("TASTE_HOME", raising=False)
    monkeypatch.setattr(tools, "_get_embedder", lambda settings: FakeEmbedder())
    tools._EMBEDDERS.clear()
    yield
    tools._EMBEDDERS.clear()


@pytest.fixture
def bare(tmp_path):
    """Ingested only: pins and images, nothing built."""
    return make_fixture_home(tmp_path / "home", n_per_group=3)


def _card(cid, domain, name):
    return ClusterCard(cluster_id=cid, domain=domain, name=name, one_liner=f"{name} one-liner",
                       principles=[f"{name} principle A", f"{name} principle B"],
                       do=[f"{name} do"], avoid=[f"{name} avoid X"],
                       contrast_note=f"{name} contrast")


@pytest.fixture
def built(bare):
    """A home with a small hand-written profile (what module C would produce)."""
    store = bare
    cards = [_card("web-01", "web", "Dark Brutalist"), _card("web-02", "web", "Warm Editorial"),
             _card("photo-01", "photo", "Blue Film")]
    for c in cards:
        store.write_card(c)
    # web-01 has module C's markdown; web-02 only has json (fallback renderer path)
    (store.profile_dir / "clusters" / "web-01.md").write_text("## Dark Brutalist CARD-MD\n")
    web = DomainSummary(domain="web", summary="Web summary text.",
                        principles=["web principle"], cluster_ids=["web-01", "web-02"])
    photo = DomainSummary(domain="photo", summary="Photo summary text.",
                          principles=["photo principle"], cluster_ids=["photo-01"])
    store.write_domain_summary(web)
    store.write_domain_summary(photo)
    (store.profile_dir / "domains" / "web.md").write_text("# Web DOMAIN-MD\n\nWeb summary.\n")
    store.write_profile(TasteProfile(generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                                     embed_model="fake", vlm_model="fake", summary="s",
                                     invariants=[], domains=[web, photo]))
    (store.profile_dir / "TASTE.md").write_text("# TASTE\n\nTop-level taste summary.\n")
    ex = store.profile_dir / "exemplars" / "web-01"
    ex.mkdir(parents=True)
    (ex / "0.jpg").write_bytes(store.abs("images/dark-00.jpg").read_bytes())
    pins = store.read_pins()
    emb = FakeEmbedder()
    store.write_embeddings(emb.name, [p.id for p in pins],
                           emb.embed_images([store.abs(p.image_path) for p in pins]))
    store.write_clusters([ClusterRecord(
        id="web-01", domain="web", size=3, pin_ids=["local:dark-00"],
        medoid_pin_id="local:dark-00", exemplar_pin_ids=["local:dark-00"],
        centroid=[1.0, 0.0], cohesion=0.9)])
    return store


@pytest.fixture
def scoring(monkeypatch):
    """Replace taste.scoring.api with recorders (module B is not merged here)."""
    calls = {"search": [], "critique": []}
    state = {
        "hits": [
            ExemplarHit(pin_id="local:blue-00", image_path="images/blue-00.jpg",
                        similarity=0.91, cluster_id="photo-01", board="film photo",
                        link="https://example.com/a"),
            ExemplarHit(pin_id="local:blue-01", image_path="images/blue-01.jpg",
                        similarity=0.88),
        ],
        "result": CritiqueResult(
            score=0.42, raw_similarity=0.7, domain="web",
            nearest=[ClusterMatch(cluster_id="web-01", similarity=0.7),
                     ClusterMatch(cluster_id="web-02", similarity=0.5)],
            nearest_anti=[0.3], notes=["a bit generic"]),
    }

    def search(store, embedder, query, domain=None, k=8):
        calls["search"].append({"embedder": embedder, "query": query, "domain": domain, "k": k})
        return state["hits"][:k]

    def critique(store, embedder, image_path):
        calls["critique"].append({"embedder": embedder, "image_path": image_path})
        return state["result"]

    monkeypatch.setattr(tools.scoring_api, "search", search)
    monkeypatch.setattr(tools.scoring_api, "critique", critique)
    return type("Scoring", (), {"calls": calls, "state": state})
