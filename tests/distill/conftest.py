"""Shared helpers: a fixture home with clusters.json written (module B is not available)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from taste.config import load_settings
from taste.fixtures import GROUPS, make_fixture_home
from taste.interfaces import FakeVLM
from taste.models import ClusterRecord, Feedback, Invariant, WebTokens
from taste.store import Store

# group -> cluster id, mirroring module B's "<domain>-<nn>" ids
CLUSTER_IDS = {"dark": "web-01", "cream": "web-02", "blue": "photo-01", "green": "photo-02",
               "red": "fashion-01"}
NEIGHBORS = {"web-01": ["web-02", "photo-01"], "web-02": ["web-01"],
             "photo-01": ["photo-02", "web-01", "fashion-01"], "photo-02": ["photo-01"],
             "fashion-01": []}


def write_fixture_clusters(store: Store) -> list[ClusterRecord]:
    pins = store.read_pins()
    records = []
    for board, domain, group, rgb in GROUPS:
        ids = [p.id for p in pins if p.id.startswith(f"local:{group}-")]
        cid = CLUSTER_IDS[group]
        records.append(ClusterRecord(
            id=cid, domain=domain, size=len(ids), pin_ids=ids, medoid_pin_id=ids[0],
            exemplar_pin_ids=ids[:12], centroid=[1.0, 0.0], cohesion=0.95,
            palette=["#%02x%02x%02x" % rgb], neighbor_cluster_ids=NEIGHBORS[cid],
            top_boards=[board]))
    store.write_clusters(records)
    return records


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = make_fixture_home(tmp_path / "home", n_per_group=10)
    write_fixture_clusters(s)
    return s


@pytest.fixture
def settings(store: Store, monkeypatch):
    monkeypatch.setenv("TASTE_VLM", "fake")
    monkeypatch.setenv("TASTE_EMBEDDER", "fake")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return load_settings(store.home)


def add_feedback(store: Store, verdict: str, note: str, domain: str | None = None,
                 n: int = 0, ts: datetime | None = None) -> None:
    store.append_feedback(Feedback(id=f"fb-{verdict}-{n}-{abs(hash(note)) % 10000}",
                                   ts=ts or datetime(2026, 1, 2, tzinfo=timezone.utc),
                                   verdict=verdict, note=note, domain=domain))


class ScriptedVLM(FakeVLM):
    """FakeVLM that fills in web tokens and cross-domain invariants like a real model would."""

    TOKENS = {
        "web-01": WebTokens(fonts=["Neo-grotesque sans", "Monospace"],
                            palette={"background": "#0c0c0e", "text": "#f2f2f2",
                                     "accent": "#ff3b00", "junk": "not a color"},
                            radius="0px, hard corners", spacing="8px base, 120px sections",
                            layout="12-column grid, left aligned"),
        "web-02": WebTokens(fonts=["Old-style serif"],
                            palette={"background": "#f0e4c8", "text": "#f2f2f2"},
                            radius="12px on cards", spacing="8px base, 120px sections",
                            layout="single centered column"),
    }

    def generate(self, prompt, schema, images=None, system=None):
        out = super().generate(prompt, schema, images=images, system=system)
        if schema.__name__ == "WebCardDraft":
            cid = prompt.split("Describe taste cluster ")[1].split(" ")[0]
            return out.model_copy(update={"name": f"Name {cid}", "tokens": self.TOKENS[cid]})
        if schema.__name__ == "CrossDomainDraft":
            return out.model_copy(update={"invariants": [
                Invariant(name="Single accent", statement="One saturated accent on neutral.",
                          evidence_cluster_ids=["web-01", "photo-01", "ghost-99"]),
                Invariant(name="Web only", statement="Rests on one domain.",
                          evidence_cluster_ids=["web-01", "web-02"]),
            ]})
        return out


def parse_front_matter(text: str) -> dict:
    """Parse the YAML subset `render.to_yaml` emits: nested maps, JSON scalars and lists."""
    assert text.startswith("---\n")
    body = text[4:].split("\n---\n", 1)[0]
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for line in body.splitlines():
        assert line.strip(), "blank line in front matter"
        indent = len(line) - len(line.lstrip(" "))
        assert indent % 2 == 0
        key, sep, value = line.strip().partition(":")
        assert sep, f"not a mapping line: {line!r}"
        key = json.loads(key) if key.startswith('"') else key
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        assert key not in parent, f"duplicate key {key}"
        value = value.strip()
        if value:
            parent[key] = json.loads(value)
        else:
            parent[key] = {}
            stack.append((indent, parent[key]))
    return root
