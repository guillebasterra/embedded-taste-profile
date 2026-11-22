import io

import pytest
from PIL import Image

from taste.interfaces import FakeEmbedder
from taste.serve import tools
from taste.serve.tools import ToolInputError


# ---- get_taste ---------------------------------------------------------------

def test_get_taste_no_domain_returns_taste_md(built):
    assert tools.get_taste(built) == "# TASTE\n\nTop-level taste summary.\n"


def test_get_taste_domain_summary(built):
    text = tools.get_taste(built, "web")
    assert "DOMAIN-MD" in text
    assert "CARD-MD" not in text and "Warm Editorial" not in text


def test_get_taste_domain_full_includes_cards(built):
    text = tools.get_taste(built, "web", "full")
    assert "DOMAIN-MD" in text
    assert "CARD-MD" in text  # module C's markdown
    assert "Warm Editorial principle A" in text and "Warm Editorial avoid X" in text  # fallback
    assert "Blue Film" not in text  # other domain's card stays out


def test_get_taste_domain_without_markdown_renders_json(built):
    text = tools.get_taste(built, "photo")
    assert "Photo summary text." in text and "photo principle" in text


def test_get_taste_missing_profile_names_the_command(bare):
    msg = tools.get_taste(bare)
    assert "not built" in msg and "taste embed" in msg and str(bare.home) in msg
    assert "not built" in tools.get_taste(bare, "web")


def test_get_taste_next_command_follows_build_state(tmp_path, built):
    from taste.store import Store

    assert "taste ingest" in tools.get_taste(Store(tmp_path / "empty"))
    (built.profile_dir / "TASTE.md").unlink()
    assert "taste distill --render-only" in tools.get_taste(built)


def test_get_taste_domain_not_in_profile(built):
    msg = tools.get_taste(built, "fashion")
    assert "no 'fashion'" in msg and "photo, web" in msg


def test_get_taste_rejects_bad_arguments(built):
    with pytest.raises(ToolInputError, match="Unknown domain"):
        tools.get_taste(built, "cars")
    with pytest.raises(ToolInputError, match="detail"):
        tools.get_taste(built, "web", "everything")


# ---- get_exemplars -----------------------------------------------------------

def test_get_exemplars_returns_text_and_thumbnails(built, scoring):
    reply = tools.get_exemplars(built, "blue grain", domain="photo", k=2)
    assert scoring.calls["search"][0]["query"] == "blue grain"
    assert scoring.calls["search"][0]["domain"] == "photo"
    assert scoring.calls["search"][0]["k"] == 2
    assert isinstance(scoring.calls["search"][0]["embedder"], FakeEmbedder)
    assert "local:blue-00" in reply.text and "0.91" in reply.text
    assert "photo-01 (Blue Film)" in reply.text and "https://example.com/a" in reply.text
    assert [t.pin_id for t in reply.thumbnails] == ["local:blue-00", "local:blue-01"]
    for t in reply.thumbnails:
        with Image.open(io.BytesIO(t.data)) as im:
            assert im.format == "JPEG" and max(im.size) <= 512


def test_thumbnail_is_capped_at_512(tmp_path):
    big = tmp_path / "big.png"
    Image.new("RGBA", (2000, 1000), (10, 20, 30, 255)).save(big)
    with Image.open(io.BytesIO(tools.make_thumbnail(big))) as im:
        assert im.format == "JPEG" and im.size == (512, 256)


def test_get_exemplars_clamps_k_and_survives_missing_image(built, scoring):
    built.abs("images/blue-01.jpg").unlink()
    reply = tools.get_exemplars(built, "blue", k=500)
    assert scoring.calls["search"][0]["k"] == tools.MAX_K
    assert len(reply.thumbnails) == 1 and "image file unavailable" in reply.text


def test_get_exemplars_empty_states(bare, scoring, tmp_path):
    from taste.store import Store

    scoring.state["hits"] = []
    assert "taste embed" in tools.get_exemplars(bare, "blue").text  # no embeddings yet
    assert "taste ingest" in tools.get_exemplars(Store(tmp_path / "empty"), "blue").text
    with pytest.raises(ToolInputError):
        tools.get_exemplars(bare, "  ")


def test_get_exemplars_before_scoring_is_implemented(bare, monkeypatch):
    def boom(*a, **k):
        raise NotImplementedError

    monkeypatch.setattr(tools.scoring_api, "search", boom)
    assert "not available" in tools.get_exemplars(bare, "blue").text


# ---- embedder ----------------------------------------------------------------

def test_embedder_is_lazy_and_cached(built, scoring, monkeypatch):
    built_count = []

    def factory(settings):
        built_count.append(settings.home)
        return FakeEmbedder()

    monkeypatch.setattr(tools, "_get_embedder", factory)
    tools.get_taste(built)
    tools.log_feedback(built, "like", "nice")
    assert built_count == []  # text tools never load it
    tools.get_exemplars(built, "blue")
    tools.critique(built, built.abs("images/dark-00.jpg"))
    assert len(built_count) == 1


# ---- critique ----------------------------------------------------------------

def test_critique_includes_card_principles_and_avoid(built, scoring):
    img = built.abs("images/dark-00.jpg")
    reply = tools.critique(built, str(img))
    assert scoring.calls["critique"][0]["image_path"] == img.resolve()
    assert reply.result.score == 0.42 and reply.card.cluster_id == "web-01"
    assert "0.42" in reply.text and "a bit generic" in reply.text
    assert "Dark Brutalist principle A" in reply.text and "Dark Brutalist avoid X" in reply.text
    assert "web-02 (Warm Editorial)" in reply.text
    data = reply.to_dict()
    assert data["result"]["score"] == 0.42
    assert data["matched_cluster"]["principles"] == ["Dark Brutalist principle A",
                                                     "Dark Brutalist principle B"]
    assert data["matched_cluster"]["avoid"] == ["Dark Brutalist avoid X"]


def test_critique_without_card_points_to_distill(built, scoring):
    built.card_path("web-01").unlink()
    reply = tools.critique(built, built.abs("images/dark-00.jpg"))
    assert reply.card is None and "taste distill" in reply.text


def test_critique_bad_inputs_and_unbuilt_home(bare, scoring, tmp_path):
    with pytest.raises(ToolInputError, match="not found"):
        tools.critique(bare, tmp_path / "nope.png")
    txt = tmp_path / "notes.png"
    txt.write_text("not an image")
    with pytest.raises(ToolInputError, match="Not a readable image"):
        tools.critique(bare, txt)
    reply = tools.critique(bare, bare.abs("images/dark-00.jpg"))
    assert reply.result is None and "taste embed" in reply.text
    assert scoring.calls["critique"] == []


# ---- log_feedback ------------------------------------------------------------

def test_log_feedback_round_trip_with_image(bare, tmp_path):
    shot = tmp_path / "shot.PNG"
    Image.new("RGB", (40, 30), (200, 30, 30)).save(shot, format="PNG")
    fb = tools.log_feedback(bare, " Dislike ", "  too much padding ", shot, "web", "portfolio")
    assert fb.verdict == "dislike" and fb.note == "too much padding"
    assert fb.image_path == f"feedback/{fb.id}.png"
    assert bare.abs(fb.image_path).read_bytes() == shot.read_bytes()
    assert shot.exists()
    assert bare.read_feedback() == [fb]
    assert fb.domain == "web" and fb.context == "portfolio" and fb.ts.tzinfo is not None

    rule = tools.log_feedback(bare, "rule", "never use gradients")
    assert rule.image_path is None and rule.domain is None
    assert [f.id for f in tools.list_feedback(bare)] == [fb.id, rule.id]
    assert fb.id in tools.feedback_confirmation(bare, fb)


@pytest.mark.parametrize("args, match", [
    (("meh", "note"), "verdict"),
    (("like", "   "), "note"),
    (("like", "note", None, "cars"), "Unknown domain"),
    (("like", "note", "/no/such/file.png"), "not found"),
])
def test_log_feedback_rejects_bad_input(bare, args, match):
    with pytest.raises(ToolInputError, match=match):
        tools.log_feedback(bare, *args)
    assert bare.read_feedback() == []
    assert list(bare.feedback_dir.iterdir()) == []


def test_log_feedback_refuses_non_image_files(bare, tmp_path):
    secret = tmp_path / ".env"
    secret.write_text("KEY=value")
    with pytest.raises(ToolInputError, match="Not a readable image"):
        tools.log_feedback(bare, "like", "note", secret)
    assert list(bare.feedback_dir.iterdir()) == []
