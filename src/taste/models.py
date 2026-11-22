"""Data contracts shared by every module.

SHARED CONTRACT FILE: owned by the orchestrator. Do not edit in a feature branch.
If a field is missing, note it in your final report and work around it locally.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Domain = Literal["web", "photo", "art", "fashion", "other"]
Source = Literal["pinterest_api", "public_board", "local"]


class Pin(BaseModel):
    """One saved image. A line of pins.jsonl."""

    id: str  # stable and unique across sources, e.g. "pin:123", "local:<sha1[:16]>"
    source: Source
    board: str  # board name (or folder name for local)
    board_id: str | None = None
    section: str | None = None
    title: str | None = None
    description: str | None = None
    alt_text: str | None = None
    link: str | None = None  # outbound link the pin points to
    image_url: str | None = None
    image_path: str | None = None  # relative to TASTE_HOME, None until downloaded
    width: int | None = None
    height: int | None = None
    created_at: datetime | None = None


class BoardEntry(BaseModel):
    """One board in boards.toml."""

    domain: Domain = "other"
    include: bool = True


class ClusterRecord(BaseModel):
    """One taste cluster. clusters.json is a list of these."""

    id: str  # "<domain>-<nn>", e.g. "web-03"
    domain: Domain
    size: int
    pin_ids: list[str]
    medoid_pin_id: str
    exemplar_pin_ids: list[str]  # nearest to centroid first, at most 12
    centroid: list[float]  # L2-normalized
    cohesion: float  # mean cosine similarity of members to the centroid
    palette: list[str] = Field(default_factory=list)  # hex colors, at most 6
    neighbor_cluster_ids: list[str] = Field(default_factory=list)  # nearest first, at most 3
    top_boards: list[str] = Field(default_factory=list)


class WebTokens(BaseModel):
    """Concrete tokens for web clusters. Everything optional: only what the images support."""

    fonts: list[str] = Field(default_factory=list)  # typeface families or classes
    type_scale: str | None = None
    palette: dict[str, str] = Field(default_factory=dict)  # role -> hex
    spacing: str | None = None
    radius: str | None = None
    layout: str | None = None
    imagery: str | None = None
    motion: str | None = None


class ClusterCard(BaseModel):
    """Distilled description of one cluster. profile/clusters/<id>.json (+ .md)."""

    cluster_id: str
    domain: Domain
    name: str  # 2-4 words
    one_liner: str
    principles: list[str]
    do: list[str]
    avoid: list[str]
    contrast_note: str  # what separates this from its nearest neighbors / the generic version
    tokens: WebTokens | None = None  # web only
    exemplar_pin_ids: list[str] = Field(default_factory=list)


class DomainSummary(BaseModel):
    """profile/domains/<domain>.json (+ .md)."""

    domain: Domain
    summary: str
    principles: list[str]
    cluster_ids: list[str]


class Invariant(BaseModel):
    """A trait that holds across domains."""

    name: str
    statement: str
    evidence_cluster_ids: list[str] = Field(default_factory=list)


class TasteProfile(BaseModel):
    """profile/profile.json. Source of truth for TASTE.md."""

    generated_at: datetime
    embed_model: str
    vlm_model: str
    summary: str
    invariants: list[Invariant]
    domains: list[DomainSummary]
    rules: list[str] = Field(default_factory=list)  # standing corrections folded in from feedback


class Feedback(BaseModel):
    """A line of profile/feedback.jsonl."""

    id: str
    ts: datetime
    verdict: Literal["like", "dislike", "rule"]
    note: str
    image_path: str | None = None  # relative to TASTE_HOME (copied into feedback/)
    domain: Domain | None = None
    context: str | None = None  # project or task this came from


class ClusterMatch(BaseModel):
    cluster_id: str
    similarity: float
    name: str | None = None  # from the ClusterCard when distilled


class CritiqueResult(BaseModel):
    """What critique() returns."""

    score: float  # 0..1, share of held-in pins this image is at least as on-taste as
    raw_similarity: float  # max cosine similarity to any cluster centroid
    domain: Domain | None  # domain of the best match
    nearest: list[ClusterMatch]
    nearest_anti: list[float] = Field(default_factory=list)  # similarity to closest disliked images
    notes: list[str] = Field(default_factory=list)


class ExemplarHit(BaseModel):
    pin_id: str
    image_path: str  # relative to TASTE_HOME
    similarity: float
    cluster_id: str | None = None
    board: str | None = None
    link: str | None = None
