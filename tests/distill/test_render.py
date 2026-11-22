from __future__ import annotations

from datetime import datetime, timezone

from taste.distill import render
from taste.models import ClusterCard, DomainSummary, Invariant, TasteProfile, WebTokens

from .conftest import parse_front_matter


def card(cid="web-01", domain="web", name="Ink grids", tokens=None):
    return ClusterCard(cluster_id=cid, domain=domain, name=name, one_liner="One line.",
                       principles=["p1", "p2"], do=["d1"], avoid=["a1"],
                       contrast_note="Differs.", tokens=tokens, exemplar_pin_ids=["a", "b"])


def test_design_placeholder_without_web_clusters():
    md = render.render_design_md([])
    fm = parse_front_matter(md)
    assert fm["version"] == "alpha" and "no web clusters" in fm["description"].lower()
    assert "no web clusters" in md.split("\n---\n", 1)[1]


def test_design_single_cluster_has_no_variants():
    tokens = WebTokens(fonts=["Inter", "JetBrains Mono"], palette={"Background": "#fff"},
                       radius="hard corners", type_scale="display 8x body")
    md = render.render_design_md([card(tokens=tokens)])
    fm = parse_front_matter(md)
    assert fm["colors"] == {"background": "#FFF"}
    assert fm["typography"] == {"display": {"fontFamily": "Inter"},
                                "body": {"fontFamily": "JetBrains Mono"}}
    assert "rounded" not in fm  # no CSS dimension in the prose, so no token is invented
    assert "variant" not in md.split("\n---\n", 1)[1]
    assert "## Shapes\n\nhard corners" in md and "display 8x body" in md


def test_design_without_tokens_still_parses():
    md = render.render_design_md([card(), card("web-02", name="Paper serif")])
    fm = parse_front_matter(md)
    assert set(fm) == {"version", "name", "description"}
    assert "No tokens have been extracted yet" in md and "- Don't: a1" in md


def test_yaml_quotes_awkward_strings():
    text = render.to_yaml({"name": 'a: "b" #c', "weird key": {"x": 1}, "list": ["a", "b"],
                           "empty": {}, "none": None})
    assert parse_front_matter(f"---\n{text}\n---\n") == {
        "name": 'a: "b" #c', "weird key": {"x": 1}, "list": ["a", "b"]}


def test_taste_md_stays_short_and_complete():
    profile = TasteProfile(
        generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc), embed_model="e", vlm_model="v",
        summary="Long summary. " * 400,
        invariants=[Invariant(name=f"inv {i}", statement="s", evidence_cluster_ids=["web-01"])
                    for i in range(30)],
        domains=[DomainSummary(domain=d, summary="First sentence. Second sentence.",
                               principles=["p"], cluster_ids=[f"{d}-01"])
                 for d in ("web", "photo", "art", "fashion", "other")],
        rules=[f"rule {i}" for i in range(60)])
    md = render.render_taste_md(profile, [card()])
    lines = md.splitlines()
    assert len(lines) <= 80 and max(len(line) for line in lines) < 1500
    assert "…and 20 more in `profile.json`" in md and "…and 40 more in `profile.json`" in md
    assert "- **[web](domains/web.md):** First sentence. Clusters: [Ink grids](clusters/web-01.md)." in md
    assert "Second sentence" not in md
    assert "## Query deeper" in md and all(t in md for t in render.MCP_TOOLS)


def test_cluster_and_domain_md_are_pure_and_linked():
    c = card(tokens=WebTokens(fonts=["Inter"], palette={"text": "#111111"}))
    md = render.render_cluster_md(c)
    assert md == render.render_cluster_md(c)
    assert "[web](../domains/web.md)" in md and "## Tokens" in md and "text `#111111`" in md
    assert "../exemplars/web-01/2.jpg" in md and "3.jpg" not in md
    summary = DomainSummary(domain="web", summary="S.", principles=["p"],
                            cluster_ids=["web-01", "web-09"])
    dmd = render.render_domain_md(summary, [c])
    assert "[Ink grids](../clusters/web-01.md)" in dmd and "- `web-09`" in dmd
