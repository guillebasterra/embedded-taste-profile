"""Storage layout under TASTE_HOME and typed read/write helpers.

SHARED CONTRACT FILE: owned by the orchestrator. Do not edit in a feature branch.

Layout
    .env                      secrets (never committed)
    auth/pinterest.json       OAuth tokens (mode 600)
    pins.jsonl                one Pin per line
    images/<pin-id-slug>.<ext>
    boards.toml               board -> domain map (user-editable)
    embeddings/<model-slug>/vectors.npy   float32, L2-normalized, shape (N, D)
    embeddings/<model-slug>/ids.json      pin ids, same order as vectors
    embeddings/<model-slug>/meta.json     {"model": ..., "dim": ...}
    clusters.json             list[ClusterRecord]
    feedback/                 images attached to feedback
    profile/                  the portable part (safe to commit to a context repo)
        profile.json  TASTE.md  DESIGN.md  SKILL.md  CLAUDE.snippet.md
        domains/<domain>.json|.md
        clusters/<id>.json|.md
        exemplars/<cluster-id>/<n>.jpg   small thumbnails
        feedback.jsonl
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Iterable, TypeVar

import numpy as np
import tomli_w
from pydantic import BaseModel

from taste.models import (
    BoardEntry,
    ClusterCard,
    ClusterRecord,
    DomainSummary,
    Feedback,
    Pin,
    TasteProfile,
)

M = TypeVar("M", bound=BaseModel)


def slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower() or "x"


class Store:
    def __init__(self, home: str | Path):
        self.home = Path(home).expanduser()

    # ---- paths -------------------------------------------------------------
    @property
    def pins_path(self) -> Path:
        return self.home / "pins.jsonl"

    @property
    def images_dir(self) -> Path:
        return self.home / "images"

    @property
    def boards_path(self) -> Path:
        return self.home / "boards.toml"

    @property
    def clusters_path(self) -> Path:
        return self.home / "clusters.json"

    @property
    def auth_dir(self) -> Path:
        return self.home / "auth"

    @property
    def feedback_dir(self) -> Path:
        return self.home / "feedback"

    @property
    def profile_dir(self) -> Path:
        return self.home / "profile"

    @property
    def feedback_path(self) -> Path:
        return self.profile_dir / "feedback.jsonl"

    def embeddings_dir(self, model: str) -> Path:
        return self.home / "embeddings" / slug(model)

    def abs(self, rel: str | Path) -> Path:
        """Resolve a path stored relative to TASTE_HOME."""
        return self.home / rel

    def rel(self, path: str | Path) -> str:
        return str(Path(path).resolve().relative_to(self.home.resolve()))

    def ensure(self) -> "Store":
        for d in (self.home, self.images_dir, self.auth_dir, self.feedback_dir, self.profile_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self

    # ---- generic helpers ---------------------------------------------------
    @staticmethod
    def _read_jsonl(path: Path, model: type[M]) -> list[M]:
        if not path.exists():
            return []
        with path.open() as f:
            return [model.model_validate_json(line) for line in f if line.strip()]

    @staticmethod
    def _write_jsonl(path: Path, rows: Iterable[BaseModel]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as f:
            for row in rows:
                f.write(row.model_dump_json() + "\n")
        tmp.replace(path)

    @staticmethod
    def _write_json(path: Path, obj: BaseModel | list[BaseModel]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = (
            [o.model_dump(mode="json") for o in obj]
            if isinstance(obj, list)
            else obj.model_dump(mode="json")
        )
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)

    # ---- pins --------------------------------------------------------------
    def read_pins(self) -> list[Pin]:
        return self._read_jsonl(self.pins_path, Pin)

    def write_pins(self, pins: Iterable[Pin]) -> None:
        self._write_jsonl(self.pins_path, pins)

    def upsert_pins(self, pins: Iterable[Pin]) -> int:
        """Merge by id (new wins, but keeps an existing image_path). Returns total count."""
        existing = {p.id: p for p in self.read_pins()}
        for p in pins:
            old = existing.get(p.id)
            if old and old.image_path and not p.image_path:
                p = p.model_copy(update={"image_path": old.image_path})
            existing[p.id] = p
        self.write_pins(existing.values())
        return len(existing)

    # ---- boards ------------------------------------------------------------
    def read_boards(self) -> dict[str, BoardEntry]:
        if not self.boards_path.exists():
            return {}
        raw = tomllib.loads(self.boards_path.read_text()).get("boards", {})
        return {name: BoardEntry.model_validate(v) for name, v in raw.items()}

    def write_boards(self, boards: dict[str, BoardEntry]) -> None:
        self.boards_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"boards": {name: e.model_dump() for name, e in sorted(boards.items())}}
        self.boards_path.write_text(tomli_w.dumps(data))

    def pin_domains(self) -> dict[str, str]:
        """pin id -> domain for every included pin that has a downloaded image."""
        boards = self.read_boards()
        out: dict[str, str] = {}
        for p in self.read_pins():
            entry = boards.get(p.board, BoardEntry())
            if entry.include and p.image_path:
                out[p.id] = entry.domain
        return out

    # ---- embeddings --------------------------------------------------------
    def write_embeddings(self, model: str, ids: list[str], vectors: np.ndarray) -> None:
        d = self.embeddings_dir(model)
        d.mkdir(parents=True, exist_ok=True)
        assert vectors.shape[0] == len(ids)
        np.save(d / "vectors.npy", vectors.astype(np.float32))
        (d / "ids.json").write_text(json.dumps(ids))
        (d / "meta.json").write_text(json.dumps({"model": model, "dim": int(vectors.shape[1])}))

    def read_embeddings(self, model: str) -> tuple[list[str], np.ndarray]:
        d = self.embeddings_dir(model)
        if not (d / "vectors.npy").exists():
            return [], np.zeros((0, 0), dtype=np.float32)
        return json.loads((d / "ids.json").read_text()), np.load(d / "vectors.npy")

    # ---- clusters ----------------------------------------------------------
    def read_clusters(self) -> list[ClusterRecord]:
        if not self.clusters_path.exists():
            return []
        return [ClusterRecord.model_validate(c) for c in json.loads(self.clusters_path.read_text())]

    def write_clusters(self, clusters: list[ClusterRecord]) -> None:
        self._write_json(self.clusters_path, clusters)

    # ---- profile -----------------------------------------------------------
    def card_path(self, cluster_id: str) -> Path:
        return self.profile_dir / "clusters" / f"{cluster_id}.json"

    def read_cards(self) -> list[ClusterCard]:
        d = self.profile_dir / "clusters"
        if not d.exists():
            return []
        return [ClusterCard.model_validate_json(p.read_text()) for p in sorted(d.glob("*.json"))]

    def write_card(self, card: ClusterCard) -> None:
        self._write_json(self.card_path(card.cluster_id), card)

    def read_domain_summaries(self) -> list[DomainSummary]:
        d = self.profile_dir / "domains"
        if not d.exists():
            return []
        return [DomainSummary.model_validate_json(p.read_text()) for p in sorted(d.glob("*.json"))]

    def write_domain_summary(self, s: DomainSummary) -> None:
        self._write_json(self.profile_dir / "domains" / f"{s.domain}.json", s)

    def read_profile(self) -> TasteProfile | None:
        p = self.profile_dir / "profile.json"
        return TasteProfile.model_validate_json(p.read_text()) if p.exists() else None

    def write_profile(self, profile: TasteProfile) -> None:
        self._write_json(self.profile_dir / "profile.json", profile)

    # ---- feedback ----------------------------------------------------------
    def read_feedback(self) -> list[Feedback]:
        return self._read_jsonl(self.feedback_path, Feedback)

    def append_feedback(self, fb: Feedback) -> None:
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with self.feedback_path.open("a") as f:
            f.write(fb.model_dump_json() + "\n")
