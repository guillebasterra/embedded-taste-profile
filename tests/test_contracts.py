"""Smoke tests for the shared contracts."""

from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from taste.fixtures import GROUPS, make_fixture_home
from taste.interfaces import FakeEmbedder, FakeVLM
from taste.models import ClusterCard, TasteProfile


def test_fixture_home(tmp_path: Path):
    store = make_fixture_home(tmp_path, n_per_group=4)
    pins = store.read_pins()
    assert len(pins) == 4 * len(GROUPS)
    assert all(store.abs(p.image_path).exists() for p in pins)
    assert set(store.pin_domains().values()) == {"web", "photo", "fashion"}


def test_fake_embedder_separates_groups(tmp_path: Path):
    store = make_fixture_home(tmp_path, n_per_group=4)
    pins = store.read_pins()
    emb = FakeEmbedder()
    v = emb.embed_images([store.abs(p.image_path) for p in pins])
    assert v.shape == (len(pins), emb.dim)
    assert np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-5)
    q = emb.embed_texts(["blue"])[0]
    best = pins[int(np.argmax(v @ q))]
    assert best.id.startswith("local:blue")


def test_fake_vlm_fills_schemas():
    vlm = FakeVLM()
    card = vlm.generate("x", ClusterCard)
    assert card.principles and card.name
    assert vlm.generate("x", TasteProfile).invariants
    assert len(vlm.calls) == 2


def test_cli_loads():
    from taste.cli import app

    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
