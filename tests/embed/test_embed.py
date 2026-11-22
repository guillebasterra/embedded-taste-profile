import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from taste.config import load_settings
from taste.embed import EmbedderUnavailable, get_embedder
from taste.embed.index import embed_pins
from taste.fixtures import make_fixture_home, make_image
from taste.interfaces import FakeEmbedder
from taste.models import BoardEntry, Pin


class CountingEmbedder(FakeEmbedder):
    def __init__(self):
        self.seen: list[Path] = []

    def embed_images(self, paths):
        self.seen.extend(paths)
        return super().embed_images(paths)


def test_get_embedder_fake(tmp_path, monkeypatch):
    monkeypatch.setenv("TASTE_EMBEDDER", "fake")
    assert isinstance(get_embedder(load_settings(tmp_path)), FakeEmbedder)


def test_get_embedder_siglip_is_lazy_and_named_after_model(tmp_path, monkeypatch):
    monkeypatch.setenv("TASTE_EMBEDDER", "siglip2")
    monkeypatch.setenv("TASTE_EMBED_MODEL", "google/siglip2-base-patch16-384")
    emb = get_embedder(load_settings(tmp_path))  # must not import torch or load weights
    assert emb.name == "google/siglip2-base-patch16-384"
    assert emb.embed_images([]).shape[0] == 0
    if importlib.util.find_spec("torch") is None:
        with pytest.raises(EmbedderUnavailable, match="uv sync --extra ml"):
            emb.embed_texts(["hello"])


def test_get_embedder_unknown(tmp_path):
    settings = dataclasses.replace(load_settings(tmp_path), embedder="nope")
    with pytest.raises(EmbedderUnavailable):
        get_embedder(settings)


def test_embed_is_incremental(tmp_path):
    store = make_fixture_home(tmp_path, n_per_group=4)
    emb = CountingEmbedder()
    stats = embed_pins(store, emb, chunk_size=7)
    assert (stats.new, stats.kept, stats.dropped, stats.total) == (20, 0, 0, 20)
    ids, vecs = store.read_embeddings("fake")
    assert ids == [p.id for p in store.read_pins()]
    assert vecs.shape == (20, emb.dim) and vecs.dtype == np.float32
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1, atol=1e-5)

    emb.seen.clear()
    stats = embed_pins(store, emb)
    assert (stats.new, stats.kept) == (0, 20) and emb.seen == []

    # one new pin -> exactly one image embedded, old rows untouched
    make_image(store.abs("images/extra.jpg"), (30, 60, 200), np.random.default_rng(1))
    store.upsert_pins([Pin(id="local:extra", source="local", board="film photo",
                           image_path="images/extra.jpg")])
    stats = embed_pins(store, emb)
    assert stats.new == 1 and emb.seen == [store.abs("images/extra.jpg")]
    ids2, vecs2 = store.read_embeddings("fake")
    assert ids2 == ids + ["local:extra"] and np.array_equal(vecs2[:20], vecs)

    emb.seen.clear()
    assert embed_pins(store, emb, force=True).new == 21 and len(emb.seen) == 21


def test_embed_drops_stale_and_skips_broken(tmp_path):
    store = make_fixture_home(tmp_path, n_per_group=4)
    emb = FakeEmbedder()
    embed_pins(store, emb)
    # exclude a board, delete a pin, add a pin whose image is not an image
    boards = store.read_boards()
    boards["fits"] = BoardEntry(domain="fashion", include=False)
    store.write_boards(boards)
    pins = [p for p in store.read_pins() if p.id != "local:blue-00"]
    store.abs("images/broken.jpg").write_text("not an image")
    pins.append(Pin(id="local:broken", source="local", board="film photo",
                    image_path="images/broken.jpg"))
    store.write_pins(pins)

    stats = embed_pins(store, emb)
    assert stats.dropped == 5 and stats.new == 0 and stats.failed == ["local:broken"]
    ids, vecs = store.read_embeddings("fake")
    assert len(ids) == 15 == vecs.shape[0]
    assert "local:blue-00" not in ids and not any(i.startswith("local:red") for i in ids)


def test_embed_cli(tmp_path, monkeypatch):
    from taste.cli import app

    monkeypatch.setenv("TASTE_EMBEDDER", "fake")
    make_fixture_home(tmp_path, n_per_group=4)
    runner = CliRunner()
    first = runner.invoke(app, ["embed", "--home", str(tmp_path)])
    assert first.exit_code == 0, first.output
    assert "Embedded 20 new pins" in first.output
    assert (tmp_path / "embeddings" / "fake" / "vectors.npy").exists()
    second = runner.invoke(app, ["embed", "--home", str(tmp_path)])
    assert second.exit_code == 0 and "Embedded 0 new pins" in second.output


def test_embed_cli_empty_home(tmp_path, monkeypatch):
    from taste.cli import app

    monkeypatch.setenv("TASTE_EMBEDDER", "fake")
    result = CliRunner().invoke(app, ["embed", "--home", str(tmp_path)])
    assert result.exit_code == 0 and "taste ingest" in result.output
