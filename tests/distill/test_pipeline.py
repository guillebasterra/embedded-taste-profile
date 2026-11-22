from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from PIL import Image

from taste.distill.pipeline import DistillError, run_distill
from taste.distill.schemas import CardDraft, CrossDomainDraft, DomainDraft, WebCardDraft
from taste.interfaces import FakeVLM

from .conftest import CLUSTER_IDS, ScriptedVLM, add_feedback, parse_front_matter

ALL_IDS = sorted(CLUSTER_IDS.values())


def card_calls(vlm):
    return [c for c in vlm.calls if c["schema"] in (CardDraft, WebCardDraft)]


def test_all_files_produced(store, settings):
    vlm = FakeVLM()
    report = run_distill(store, settings, vlm)
    p = store.profile_dir
    assert sorted(report.cards_generated) == ALL_IDS
    assert report.vlm_calls == len(vlm.calls) == 5 + 3 + 1
    for cid in ALL_IDS:
        assert (p / "clusters" / f"{cid}.json").exists()
        md = (p / "clusters" / f"{cid}.md").read_text()
        assert f"`{cid}`" in md and "## Principles" in md and "## What sets it apart" in md
        thumbs = sorted((p / "exemplars" / cid).glob("*.jpg"))
        assert [t.name for t in thumbs] == [f"{n}.jpg" for n in range(1, 9)]
        with Image.open(thumbs[0]) as im:
            assert max(im.size) <= 512
        assert f"../exemplars/{cid}/8.jpg" in md
    for domain in ("web", "photo", "fashion"):
        assert (p / "domains" / f"{domain}.json").exists()
        assert "## Clusters" in (p / "domains" / f"{domain}.md").read_text()
    assert not (p / "domains" / "art.json").exists()
    for name in ("profile.json", "TASTE.md", "DESIGN.md"):
        assert (p / name).exists()

    profile = store.read_profile()
    assert [d.domain for d in profile.domains] == ["web", "photo", "fashion"]
    assert profile.domains[0].cluster_ids == ["web-01", "web-02"]
    assert profile.vlm_model == "fake" and profile.embed_model == "fake"
    # FakeVLM cites no clusters, so no invariant survives the two-domain check
    assert profile.invariants == [] and len(report.invariants_dropped) == 2

    taste = (p / "TASTE.md").read_text()
    assert len(taste.splitlines()) <= 80
    for tool in ("get_taste", "get_exemplars", "critique", "log_feedback"):
        assert tool in taste
    for domain in ("web", "photo", "fashion"):
        assert f"(domains/{domain}.md)" in taste


def test_card_fields_come_from_the_record(store, settings):
    run_distill(store, settings, FakeVLM())
    for card in store.read_cards():
        record = next(r for r in store.read_clusters() if r.id == card.cluster_id)
        assert card.domain == record.domain  # FakeVLM would have said "web" for everything
        assert card.exemplar_pin_ids == record.exemplar_pin_ids[:8]
        assert card.tokens is None


def test_images_and_labels_sent_to_vlm(store, settings):
    vlm = FakeVLM()
    run_distill(store, settings, vlm)
    by_id = {c["prompt"].split("Describe taste cluster ")[1].split(" ")[0]: c
             for c in card_calls(vlm)}
    assert sorted(by_id) == ALL_IDS

    call = by_id["photo-01"]  # 8 targets + 2 from each of 3 neighbors
    assert len(call["images"]) == 8 + 6
    assert [p.name for p in call["images"][:8]] == [f"blue-{i:02d}.jpg" for i in range(8)]
    assert [p.name for p in call["images"][8:]] == [
        "green-00.jpg", "green-01.jpg", "dark-00.jpg", "dark-01.jpg", "red-00.jpg",
        "red-01.jpg"]
    prompt = call["prompt"]
    assert 'Image 1: TARGET photo-01 (board "film photo", title "blue 0")' in prompt
    assert 'Image 8: TARGET photo-01 (board "film photo", title "blue 7")' in prompt
    assert 'Image 9: NEIGHBOR photo-02 (board "film photo", title "green 0")' in prompt
    assert 'Image 11: NEIGHBOR web-01 (board "web inspo", title "dark 0")' in prompt
    assert "Image 14: NEIGHBOR fashion-01" in prompt and "Image 15" not in prompt
    assert "#1e3cc8" in prompt  # measured palette is passed as a fact
    assert call["system"] and "Banned filler" in call["system"]
    assert call["schema"] is CardDraft and "`tokens`" not in prompt

    web = by_id["web-01"]
    assert web["schema"] is WebCardDraft and "`tokens` (web only)" in web["prompt"]
    assert len(web["images"]) == 8 + 4

    lonely = by_id["fashion-01"]
    assert len(lonely["images"]) == 8 and "NEIGHBOR" not in lonely["prompt"].split("legend")[1]

    text_only = [c for c in vlm.calls if c["schema"] in (DomainDraft, CrossDomainDraft)]
    assert len(text_only) == 4 and all(c["images"] == [] for c in text_only)
    final = vlm.calls[-1]
    assert final["schema"] is CrossDomainDraft
    assert "at least two different domains" in final["prompt"]
    assert "single domain is forbidden" in final["prompt"]
    assert all(cid in final["prompt"] for cid in ALL_IDS)


def test_missing_images_are_not_sent(store, settings):
    store.abs("images/blue-00.jpg").unlink()
    vlm = FakeVLM()
    run_distill(store, settings, vlm, only="photo-01")
    call = card_calls(vlm)[0]
    assert [p.name for p in call["images"][:8]] == [f"blue-{i:02d}.jpg" for i in range(1, 9)]
    assert store.read_cards()[0].exemplar_pin_ids[0] == "local:blue-01"


def test_second_run_makes_no_vlm_calls(store, settings):
    run_distill(store, settings, FakeVLM())
    before = (store.profile_dir / "profile.json").read_text()
    vlm = FakeVLM()
    report = run_distill(store, settings, vlm)
    assert vlm.calls == [] and report.vlm_calls == 0
    assert sorted(report.cards_skipped) == ALL_IDS
    assert (store.profile_dir / "profile.json").read_text() == before


def test_changed_exemplars_redo_only_what_depends_on_them(store, settings):
    run_distill(store, settings, FakeVLM())
    records = store.read_clusters()
    target = next(r for r in records if r.id == "photo-02")
    target.exemplar_pin_ids = list(reversed(target.exemplar_pin_ids))
    store.write_clusters(records)
    vlm = FakeVLM()
    report = run_distill(store, settings, vlm)
    assert report.cards_generated == ["photo-02"]
    assert report.domains_generated == ["photo"] and report.profile_generated
    assert [c["schema"] for c in vlm.calls] == [CardDraft, DomainDraft, CrossDomainDraft]


def test_force_redoes_everything(store, settings):
    run_distill(store, settings, FakeVLM())
    vlm = FakeVLM()
    report = run_distill(store, settings, vlm, force=True)
    assert len(vlm.calls) == 9 and sorted(report.cards_generated) == ALL_IDS


def test_only_limits_to_one_cluster(store, settings):
    vlm = FakeVLM()
    report = run_distill(store, settings, vlm, only="web-02")
    assert report.cards_generated == ["web-02"]
    assert len(card_calls(vlm)) == 1
    assert [c.cluster_id for c in store.read_cards()] == ["web-02"]
    assert not (store.profile_dir / "exemplars" / "web-01").exists()

    run_distill(store, settings, FakeVLM())  # fill in the rest
    vlm = FakeVLM()
    report = run_distill(store, settings, vlm, only="photo-01")  # --only always redoes its card
    assert report.cards_generated == ["photo-01"] and len(card_calls(vlm)) == 1
    assert report.domains_generated == ["photo"]


def test_only_unknown_cluster(store, settings):
    with pytest.raises(DistillError, match="Unknown cluster id"):
        run_distill(store, settings, FakeVLM(), only="web-99")


def test_no_clusters(tmp_path, settings):
    from taste.store import Store

    with pytest.raises(DistillError, match="taste cluster"):
        run_distill(Store(tmp_path / "empty").ensure(), settings, FakeVLM())


def test_feedback_rules_and_evidence(store, settings):
    add_feedback(store, "rule", "Never use drop shadows.", n=1)
    add_feedback(store, "rule", "Never use drop shadows.", n=2)
    add_feedback(store, "rule", "Body text is never centered.", n=3)
    add_feedback(store, "dislike", "the gradient hero felt like a SaaS template", "web", n=4)
    add_feedback(store, "like", "grainy overcast street shots", "photo", n=5)
    add_feedback(store, "like", "tight crops everywhere", None, n=6)
    vlm = FakeVLM()
    run_distill(store, settings, vlm)

    profile = json.loads((store.profile_dir / "profile.json").read_text())
    assert profile["rules"] == ["Never use drop shadows.", "Body text is never centered."]
    taste = (store.profile_dir / "TASTE.md").read_text()
    assert "- Never use drop shadows." in taste and "- Body text is never centered." in taste
    assert taste.count("Never use drop shadows.") == 1

    web = next(c for c in vlm.calls if "cluster web-01" in c["prompt"])
    assert "DISLIKE [web]: the gradient hero felt like a SaaS template" in web["prompt"]
    assert "tight crops everywhere" in web["prompt"]
    assert "grainy overcast" not in web["prompt"] and "drop shadows" not in web["prompt"]
    photo_summary = next(c for c in vlm.calls if c["schema"] is DomainDraft
                         and "`photo` domain" in c["prompt"])
    assert "LIKE [photo]: grainy overcast street shots" in photo_summary["prompt"]
    assert "SaaS template" not in photo_summary["prompt"]
    assert "SaaS template" in vlm.calls[-1]["prompt"] and "grainy" in vlm.calls[-1]["prompt"]


def test_new_rule_is_folded_in_without_vlm(store, settings):
    run_distill(store, settings, FakeVLM())
    add_feedback(store, "rule", "No stock photography.")
    add_feedback(store, "dislike", "old note, already folded in")
    add_feedback(store, "dislike", "too much border radius",
                 ts=datetime.now(timezone.utc) + timedelta(hours=1))
    vlm = FakeVLM()
    report = run_distill(store, settings, vlm)
    assert vlm.calls == [] and report.stale_feedback == 1
    assert store.read_profile().rules == ["No stock photography."]
    assert "- No stock photography." in (store.profile_dir / "TASTE.md").read_text()


def test_render_only_uses_no_vlm(store, settings, monkeypatch):
    run_distill(store, settings, FakeVLM())
    p = store.profile_dir
    originals = {f: f.read_text() for f in p.rglob("*.md")}
    assert len(originals) == 5 + 3 + 2
    for f in originals:
        f.unlink()
    card_path = store.card_path("web-01")
    data = json.loads(card_path.read_text())
    data["name"] = "Ink index grids"
    card_path.write_text(json.dumps(data))
    add_feedback(store, "rule", "Rule added later.")

    def boom(_settings):
        raise AssertionError("render-only must not build a VLM")

    monkeypatch.setattr("taste.distill.llm.get_vlm", boom)
    report = run_distill(store, settings, None, render_only=True)
    assert report.vlm_calls == 0
    assert {p / rel for rel in report.files_rendered} == set(originals)
    assert all(f.exists() for f in originals)
    assert "# Ink index grids (`web-01`)" in (p / "clusters" / "web-01.md").read_text()
    assert "Ink index grids" in (p / "TASTE.md").read_text()
    assert "- Rule added later." in (p / "TASTE.md").read_text()
    unchanged = p / "clusters" / "photo-02.md"
    assert unchanged.read_text() == originals[unchanged]


def test_render_only_needs_json(tmp_path, settings):
    from taste.store import Store

    with pytest.raises(DistillError, match="Nothing to render"):
        run_distill(Store(tmp_path / "empty").ensure(), settings, None, render_only=True)


def test_orphans_are_pruned(store, settings):
    run_distill(store, settings, FakeVLM())
    records = [r for r in store.read_clusters() if r.id != "fashion-01"]
    for r in records:
        r.neighbor_cluster_ids = [n for n in r.neighbor_cluster_ids if n != "fashion-01"]
    store.write_clusters(records)
    vlm = FakeVLM()
    report = run_distill(store, settings, vlm)
    p = store.profile_dir
    assert not (p / "clusters" / "fashion-01.json").exists()
    assert not (p / "clusters" / "fashion-01.md").exists()
    assert not (p / "exemplars" / "fashion-01").exists()
    assert not (p / "domains" / "fashion.md").exists()
    assert report.pruned and [c["schema"] for c in vlm.calls] == [CrossDomainDraft]
    assert [d.domain for d in store.read_profile().domains] == ["web", "photo"]
    assert "fashion" not in (p / "TASTE.md").read_text()


def test_scripted_tokens_and_invariants(store, settings):
    report = run_distill(store, settings, ScriptedVLM())
    profile = store.read_profile()
    assert [i.name for i in profile.invariants] == ["Single accent"]
    assert profile.invariants[0].evidence_cluster_ids == ["web-01", "photo-01"]
    assert report.invariants_dropped == ["Web only"]
    taste = (store.profile_dir / "TASTE.md").read_text()
    assert "**Single accent:**" in taste and "`photo-01`" in taste and "Web only" not in taste

    cards = {c.cluster_id: c for c in store.read_cards()}
    assert "junk" not in cards["web-01"].tokens.palette  # non-hex values are dropped
    assert cards["photo-01"].tokens is None

    design = (store.profile_dir / "DESIGN.md").read_text()
    fm = parse_front_matter(design)
    assert fm["version"] == "alpha" and fm["name"]
    # both clusters agree on text color and spacing; background, fonts, radius are variants
    assert fm["colors"]["text"] == "#F2F2F2"
    assert fm["colors"]["name-web-01"] == {"background": "#0C0C0E", "accent": "#FF3B00"}
    assert fm["colors"]["name-web-02"] == {"background": "#F0E4C8"}
    assert fm["typography"]["name-web-01"]["display"] == {"fontFamily": "Neo-grotesque sans"}
    assert fm["typography"]["name-web-02"]["display"] == {"fontFamily": "Old-style serif"}
    assert fm["spacing"] == {"base": "8px"}
    assert fm["rounded"] == {"name-web-01": {"default": "0px"},
                             "name-web-02": {"default": "12px"}}
    md = design.split("\n---\n", 1)[1]
    order = [md.index(h) for h in ("## Overview", "## Colors", "## Typography", "## Layout",
                                   "## Shapes", "## Do's and Don'ts")]
    assert order == sorted(order)
    assert "variant `name-web-01`" in md and "single centered column" in md
