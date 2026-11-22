import json
from datetime import datetime, timezone

import numpy as np
import pytest
from typer.testing import CliRunner

from taste.fixtures import make_fixture_home
from taste.models import BoardEntry, ClusterCard, Feedback
from taste.scoring import api
from taste.scoring.evaluate import evaluate, suggest_threshold


def _dislike(store, src, name="bad.jpg", verdict="dislike"):
    dest = store.feedback_dir / name
    dest.write_bytes(src.read_bytes())
    store.append_feedback(Feedback(id=name, ts=datetime.now(timezone.utc), verdict=verdict,
                                   note="too loud", image_path=f"feedback/{name}"))


# ---- search ---------------------------------------------------------------

def test_search_blue_returns_blue_pins_first(store, embedder):
    hits = api.search(store, embedder, "blue")
    assert len(hits) == 8
    assert all(h.pin_id.startswith("local:blue") for h in hits)
    sims = [h.similarity for h in hits]
    assert sims == sorted(sims, reverse=True)
    blue_cluster = {h.cluster_id for h in hits}
    assert len(blue_cluster) == 1 and blue_cluster.pop().startswith("photo-")
    assert hits[0].board == "film photo" and store.abs(hits[0].image_path).exists()


def test_search_k_and_domain(store, embedder):
    assert len(api.search(store, embedder, "blue", k=3)) == 3
    assert api.search(store, embedder, "blue", k=0) == []
    web = api.search(store, embedder, "blue", domain="web", k=50)
    assert len(web) == 16 and all(h.board == "web inspo" for h in web)
    assert api.search(store, embedder, "a warm website", domain="web", k=1)[0].pin_id.startswith(
        "local:cream")
    assert api.search(store, embedder, "blue", domain="art") == []


def test_search_skips_excluded_boards_and_empty_index(store, embedder, tmp_path):
    boards = store.read_boards()
    boards["film photo"] = BoardEntry(domain="photo", include=False)
    store.write_boards(boards)
    assert not any(h.board == "film photo" for h in api.search(store, embedder, "blue", k=50))
    empty = make_fixture_home(tmp_path / "empty", n_per_group=2)
    assert api.search(empty, embedder, "blue") == []


# ---- critique -------------------------------------------------------------

def test_critique_red_beats_magenta_and_noise(store, embedder, images):
    red = api.critique(store, embedder, images / "red.jpg")
    magenta = api.critique(store, embedder, images / "magenta.jpg")
    noise = api.critique(store, embedder, images / "noise.png")
    assert red.score > magenta.score and red.score > noise.score
    assert red.raw_similarity > 0.99 > magenta.raw_similarity
    assert 0 < red.score <= 1 and magenta.score == 0 and noise.score == 0
    assert red.domain == "fashion" and red.nearest[0].cluster_id == "fashion-01"
    assert len(red.nearest) == 3
    assert [m.similarity for m in red.nearest] == sorted(
        (m.similarity for m in red.nearest), reverse=True)
    assert red.nearest[0].name is None and red.nearest_anti == []
    assert any("fashion-01" in n for n in red.notes)
    assert any(n.startswith("Off taste") for n in noise.notes)


def test_critique_score_is_percentile_of_own_pins(store, embedder):
    # A pin's own image scores as the share of pins no closer to their centroid than it is.
    ids, vecs = store.read_embeddings("fake")
    cents = np.array([c.centroid for c in store.read_clusters()], dtype=np.float32)
    ref = (vecs @ cents.T).max(axis=1)
    pin = store.read_pins()[0]
    result = api.critique(store, embedder, store.abs(pin.image_path))
    assert result.score == pytest.approx(float(np.mean(ref <= ref[0] + 1e-6)))
    assert api.percentile_score(0.5, np.array([0.1, 0.4, 0.6, 0.9])) == 0.5


def test_critique_uses_card_names(store, embedder, images):
    store.write_card(ClusterCard(cluster_id="fashion-01", domain="fashion", name="Loud Red",
                                 one_liner="x", principles=[], do=[], avoid=[],
                                 contrast_note="x"))
    result = api.critique(store, embedder, images / "red.jpg")
    assert result.nearest[0].name == "Loud Red" and result.nearest[1].name is None
    assert "Loud Red (fashion-01)" in result.notes[0]


def test_dislike_feedback_populates_nearest_anti(store, embedder, images):
    _dislike(store, images / "magenta.jpg")
    _dislike(store, images / "noise.png", name="liked.png", verdict="like")  # ignored
    store.append_feedback(Feedback(id="gone", ts=datetime.now(timezone.utc), verdict="dislike",
                                   note="x", image_path="feedback/gone.jpg"))  # missing file
    magenta = api.critique(store, embedder, images / "magenta.jpg")
    assert len(magenta.nearest_anti) == 1 and magenta.nearest_anti[0] > 0.999
    assert any("disliked reference" in n and n.startswith("Closer") for n in magenta.notes)
    red = api.critique(store, embedder, images / "red.jpg")
    assert len(red.nearest_anti) == 1 and red.nearest_anti[0] < red.raw_similarity
    assert not any("disliked" in n for n in red.notes)


def test_critique_errors(store, embedder, images, tmp_path):
    with pytest.raises(FileNotFoundError):
        api.critique(store, embedder, images / "nope.jpg")
    bare = make_fixture_home(tmp_path / "bare", n_per_group=2)
    with pytest.raises(RuntimeError, match="taste cluster"):
        api.critique(bare, embedder, images / "red.jpg")


# ---- eval -----------------------------------------------------------------

def test_eval_separates_heldout_from_noise(store, embedder, distractors):
    report = evaluate(store, embedder, distractors, threshold=0.3)
    assert report.auc > 0.9
    assert (report.n_train, report.n_heldout, report.n_distractors) == (32, 8, 20)
    assert report.n_clusters == 5
    assert report.distractor_mean < report.threshold < report.heldout_mean
    assert report.tpr == 1.0 and report.fpr == 0.0
    assert len(store.read_clusters()) == 5  # eval never rewrites clusters.json


def test_suggest_threshold():
    cut = suggest_threshold(np.array([0.9, 0.8, 0.7]), np.array([0.3, 0.5]))
    assert cut == pytest.approx(0.6)


def test_eval_errors(store, embedder, tmp_path):
    (tmp_path / "none").mkdir()
    with pytest.raises(RuntimeError, match="No images"):
        evaluate(store, embedder, tmp_path / "none")


# ---- CLI ------------------------------------------------------------------

def test_cli_end_to_end(tmp_path, monkeypatch, images, distractors):
    from taste.cli import app

    monkeypatch.setenv("TASTE_EMBEDDER", "fake")
    monkeypatch.setenv("COLUMNS", "200")
    home = tmp_path / "clihome"
    store = make_fixture_home(home)
    runner = CliRunner()

    early = runner.invoke(app, ["critique", str(images / "red.jpg"), "--home", str(home)])
    assert early.exit_code == 1 and "taste cluster" in early.output

    for cmd in (["embed"], ["cluster"]):
        r = runner.invoke(app, cmd + ["--home", str(home)])
        assert r.exit_code == 0, r.output

    r = runner.invoke(app, ["search", "blue", "--home", str(home), "-k", "3"])
    assert r.exit_code == 0, r.output
    assert r.output.count("local:blue") == 3 and "local:red" not in r.output
    r = runner.invoke(app, ["search", "blue", "--home", str(home), "--domain", "photo", "--json"])
    assert [h["pin_id"][:10] for h in json.loads(r.output)] == ["local:blue"] * 8

    _dislike(store, images / "magenta.jpg")
    r = runner.invoke(app, ["critique", str(images / "red.jpg"), "--home", str(home)])
    assert r.exit_code == 0, r.output
    assert "fashion-01" in r.output and "On taste" in r.output
    r = runner.invoke(app, ["critique", str(images / "magenta.jpg"), "--home", str(home),
                            "--json"])
    data = json.loads(r.output)
    assert data["score"] == 0 and data["nearest_anti"][0] > 0.999

    r = runner.invoke(app, ["eval", "--distractors", str(distractors), "--home", str(home)])
    assert r.exit_code == 0, r.output
    assert "ROC AUC: 1.000" in r.output and "Suggested accept threshold" in r.output

    monkeypatch.setenv("TASTE_HOME", str(home))  # --home is optional when TASTE_HOME is set
    assert runner.invoke(app, ["search", "red", "-k", "1"]).output.count("local:red") == 1
