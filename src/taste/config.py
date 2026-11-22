"""Settings. Everything is read from the environment, with TASTE_HOME/.env loaded first.

SHARED CONTRACT FILE: owned by the orchestrator. Do not edit in a feature branch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

DOMAINS = ("web", "photo", "art", "fashion", "other")

PINTEREST_REDIRECT_URI = "http://localhost:8085/callback"
PINTEREST_SCOPES = (
    "boards:read",
    "boards:read_secret",
    "pins:read",
    "pins:read_secret",
    "user_accounts:read",
)


@dataclass(frozen=True)
class Settings:
    home: Path
    embedder: str  # "siglip2" (default) or "fake"
    embed_model: str  # HF model id used by the siglip2 embedder
    vlm: str  # "anthropic" (default) or "fake"
    vlm_model: str
    anthropic_api_key: str | None
    pinterest_app_id: str | None
    pinterest_app_secret: str | None
    pinterest_access_token: str | None


def load_settings(home: str | Path | None = None) -> Settings:
    """Resolve settings. Precedence: process env > TASTE_HOME/.env > defaults."""
    home_path = Path(home or os.environ.get("TASTE_HOME") or "~/.taste").expanduser()
    file_env = dotenv_values(home_path / ".env") if (home_path / ".env").exists() else {}

    def get(key: str, default: str | None = None) -> str | None:
        return os.environ.get(key) or file_env.get(key) or default

    return Settings(
        home=home_path,
        embedder=get("TASTE_EMBEDDER", "siglip2"),
        embed_model=get("TASTE_EMBED_MODEL", "google/siglip2-base-patch16-384"),
        vlm=get("TASTE_VLM", "anthropic"),
        vlm_model=get("TASTE_VLM_MODEL", "claude-sonnet-5-5"),
        anthropic_api_key=get("ANTHROPIC_API_KEY"),
        pinterest_app_id=get("PINTEREST_APP_ID"),
        pinterest_app_secret=get("PINTEREST_APP_SECRET"),
        pinterest_access_token=get("PINTEREST_ACCESS_TOKEN"),
    )
