import numpy as np
import pytest
from PIL import Image

from taste.cluster import cluster_store
from taste.embed.index import embed_pins
from taste.fixtures import make_fixture_home, make_image
from taste.interfaces import FakeEmbedder


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def store(tmp_path, embedder):
    s = make_fixture_home(tmp_path / "home")
    embed_pins(s, embedder)
    cluster_store(s, embedder.name, threshold=0.3)
    return s


@pytest.fixture
def images(tmp_path):
    """Probe images outside the home: on-taste red, off-taste magenta, random noise."""
    d = tmp_path / "probes"
    rng = np.random.default_rng(42)
    make_image(d / "red.jpg", (200, 30, 30), rng)
    make_image(d / "magenta.jpg", (200, 30, 200), rng)
    Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(d / "noise.png")
    return d


@pytest.fixture
def distractors(tmp_path):
    d = tmp_path / "distractors" / "nested"
    d.mkdir(parents=True)
    rng = np.random.default_rng(7)
    for i in range(20):
        Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(d / f"n{i}.png")
    return d.parent
