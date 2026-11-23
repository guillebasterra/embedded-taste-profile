import json
from collections import Counter

import pytest
from typer.testing import CliRunner

from taste.cli import app
from taste.facets.read import READINGS_FILE, facets_dir, run_facets
from taste.facets.threads import (effective_count, fit_threads, read_threads, spread_examples,
                                  write_threads)
from taste.facets.describe import describe_threads, read_cards
from taste.fixtures import make_fixture_home
from taste.interfaces import FakeVLM

runner = CliRunner()

STYLES = {
    "grain": ["film grain", "candid snapshot", "soft focus", "warm color cast", "faded"],
    "pixel": ["pixel grid", "dense technical", "halftone dots", "monochrome screen", "glitch"],
    "space": ["negative space", "single figure", "quiet field", "low horizon", "wide margins"],
}


def seed_readings(tmp_path):
    """Fixture pins with readings whose tags follow a style that ignores the board."""
    store = make_fixture_home(tmp_path, n_per_group=6)
    run_facets(store, None)  # measures
    ids = sorted(store.pin_domains())
    styles = list(STYLES)
    rows, truth = [], {}
    for n, pid in enumerate(ids):
        style = styles[n % len(styles)]
        truth[pid] = style
        rows.append({"pin_id": pid, "model": "fake", "reading": {
            "medium": "photo", "subject": "x", "style_tags": STYLES[style][:4 + n % 2],
            "palette": "p", "light": "l", "texture": "t", "line_and_shape": "s",
            "composition": "c", "finish": 1 + n % 3, "mood": ["calm"], "distinctive": "d"}})
    (facets_dir(store) / READINGS_FILE).write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    return store, truth


def test_threads_follow_style_across_boards(tmp_path):
    store, truth = seed_readings(tmp_path)
    found = fit_threads(store, k=3)
    assert [t.id for t in found] == ["t01", "t02", "t03"]
    for t in found:
        styles = Counter(truth[p] for p in t.pins)
        style, count = styles.most_common(1)[0]
        assert count == len(t.pins)  # pure
        assert any(term in " ".join(STYLES[style]) for term in t.terms[:3])
        assert len(t.boards) > 1 and t.spread > 1  # crosses boards
        assert set(t.measures) == {"lightness", "contrast", "saturation", "warmth", "busyness"}
    assert {Counter(truth[p] for p in t.pins).most_common(1)[0][0] for t in found} == set(STYLES)
    assert found == sorted(found, key=lambda t: (-t.spread, -len(t.pins)))

    write_threads(store, found)
    assert read_threads(store) == found


def test_too_few_readings(tmp_path):
    store = make_fixture_home(tmp_path, n_per_group=1)
    with pytest.raises(ValueError, match="taste facets"):
        fit_threads(store, k=16)


def test_effective_count_and_examples():
    assert effective_count(Counter()) == 0.0
    assert effective_count(Counter(a=5)) == pytest.approx(1.0)
    assert effective_count(Counter(a=3, b=3)) == pytest.approx(2.0)

    class T:
        pins = ["p1", "p2", "p3", "p4"]
    boards = {"p1": "A", "p2": "A", "p3": "B", "p4": "C"}
    assert spread_examples(T, boards, 3) == ["p1", "p3", "p4"]
    assert spread_examples(T, boards, 4) == ["p1", "p3", "p4", "p2"]


def test_cli_writes_table_preview_and_cards(tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setenv("TASTE_VLM", "fake")
    store, _ = seed_readings(tmp_path)
    result = runner.invoke(app, ["threads", "--home", str(tmp_path), "--k", "3"])
    assert result.exit_code == 0, result.output
    assert "t01" in result.output and "threads.html" in result.output
    assert "3 thread cards written, 0 reused" in result.output
    page = (facets_dir(store) / "threads.html").read_text()
    assert page.count("<section>") == 3 and "file://" in page
    result = runner.invoke(app, ["threads", "--home", str(tmp_path), "--k", "3"])
    assert "0 thread cards written, 3 reused" in result.output

    result = runner.invoke(app, ["threads", "--home", str(tmp_path), "--k", "3",
                                 "--no-describe"])
    assert result.exit_code == 0 and "thread cards" not in result.output


def test_cards_reuse_by_members_and_clean_thumbnails(tmp_path):
    store, _ = seed_readings(tmp_path)
    found = fit_threads(store, k=3)
    vlm = FakeVLM()
    cards, generated = describe_threads(store, found, vlm)
    assert generated == 3 and [c.thread_id for c in cards] == ["t01", "t02", "t03"]
    card = cards[0]
    assert card.pins == len(found[0].pins) and card.exemplars[0] == found[0].pins[0]
    assert len(card.thumbnails) == 4
    assert all((store.profile_dir / p).exists() for p in card.thumbnails)
    prompt = vlm.calls[0]["prompt"]
    assert "Top terms:" in prompt and "Per-image notes:" in prompt
    assert len(vlm.calls[0]["images"]) == len(card.exemplars) <= 8
    assert read_cards(store) == cards
    md = (store.profile_dir / "threads.md").read_text()
    assert md.count("## t0") == 3 and "**On the web:**" in md and "![](threads/t01-1.jpg)" in md

    _, generated = describe_threads(store, found, vlm)
    assert generated == 0
    _, generated = describe_threads(store, found[:2], vlm, force=True)
    assert generated == 2
    assert not list((store.profile_dir / "threads").glob("t03-*.jpg"))


def test_cli_skips_quietly_on_a_small_home(tmp_path, monkeypatch):
    make_fixture_home(tmp_path, n_per_group=1)
    result = runner.invoke(app, ["threads", "--home", str(tmp_path)])
    assert result.exit_code == 0 and "Skipped threads" in result.output

