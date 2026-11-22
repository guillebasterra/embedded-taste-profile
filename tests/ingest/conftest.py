from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from taste.config import Settings

API = "https://api.pinterest.com/v5"


def image_bytes(color=(200, 30, 30), size=(40, 30), fmt="JPEG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, fmt)
    return buf.getvalue()


def make_settings(home: Path, **over) -> Settings:
    base = dict(home=home, embedder="fake", embed_model="m", vlm="fake", vlm_model="v",
                anthropic_api_key=None, pinterest_app_id="app123",
                pinterest_app_secret="s3cr3t-value", pinterest_access_token=None)
    base.update(over)
    return Settings(**base)


def raw_pin(pid: str, board_id: str = "b1", section_id: str | None = None, **over) -> dict:
    pin = {
        "id": pid, "title": f"title {pid}", "description": "desc", "alt_text": None,
        "link": f"https://example.com/{pid}", "board_id": board_id,
        "board_section_id": section_id, "created_at": "2024-05-01T10:00:00",
        "media": {"media_type": "image", "images": {
            "150x150": {"url": f"https://i.pinimg.com/150x150/{pid}.jpg",
                        "width": 150, "height": 150},
            "1200x": {"url": f"https://i.pinimg.com/1200x/{pid}.jpg",
                      "width": 1200, "height": 900},
            "600x": {"url": f"https://i.pinimg.com/600x/{pid}.jpg", "width": 600, "height": 450},
        }},
    }
    pin.update(over)
    return pin


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("TASTE_HOME", "PINTEREST_APP_ID", "PINTEREST_APP_SECRET",
                "PINTEREST_ACCESS_TOKEN"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def home(tmp_path: Path) -> Path:
    return tmp_path / "home"
