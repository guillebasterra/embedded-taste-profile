"""Synthetic TASTE_HOME for tests and demos. No network, no model weights.

SHARED CONTRACT FILE: owned by the orchestrator. Do not edit in a feature branch.

Groups are solid-ish color fields with noise, so the FakeEmbedder separates them cleanly:
    board "web inspo"  (web)     -> dark group, cream group
    board "film photo" (photo)   -> blue group, green group
    board "fits"       (fashion) -> red group
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from taste.models import BoardEntry, Pin
from taste.store import Store

GROUPS = [
    # (board, domain, group name, base rgb)
    ("web inspo", "web", "dark", (12, 12, 14)),
    ("web inspo", "web", "cream", (240, 228, 200)),
    ("film photo", "photo", "blue", (30, 60, 200)),
    ("film photo", "photo", "green", (30, 160, 60)),
    ("fits", "fashion", "red", (200, 30, 30)),
]


def make_image(path: Path, rgb: tuple[int, int, int], rng: np.random.Generator,
               size: int = 64) -> None:
    base = np.tile(np.array(rgb, dtype=np.float32), (size, size, 1))
    noise = rng.normal(0, 6, size=base.shape)
    arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, quality=90)


def make_fixture_home(home: str | Path, n_per_group: int = 8, seed: int = 0) -> Store:
    """Create pins.jsonl, images/ and boards.toml. Returns the Store."""
    store = Store(home).ensure()
    rng = np.random.default_rng(seed)
    pins: list[Pin] = []
    for board, _domain, group, rgb in GROUPS:
        for i in range(n_per_group):
            pid = f"local:{group}-{i:02d}"
            rel = f"images/{group}-{i:02d}.jpg"
            make_image(store.abs(rel), rgb, rng)
            pins.append(Pin(id=pid, source="local", board=board, title=f"{group} {i}",
                            image_path=rel, width=64, height=64))
    store.write_pins(pins)
    store.write_boards({b: BoardEntry(domain=d) for b, d, _g, _c in GROUPS})
    return store
