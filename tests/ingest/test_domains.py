from __future__ import annotations

import pytest

from taste.fixtures import make_fixture_home
from taste.ingest.domains import guess_domain, update_boards
from taste.models import BoardEntry, Pin
from taste.store import Store


@pytest.mark.parametrize("name,domain", [
    ("web inspo", "web"), ("UI / UX", "web"), ("Sites I love", "web"), ("type", "web"),
    ("Typography", "web"), ("landing-pages", "web"),
    ("film photo", "photo"), ("35mm", "photo"), ("Photography", "photo"), ("photos", "photo"),
    ("fits", "fashion"), ("Outfits", "fashion"), ("what to wear", "fashion"),
    ("raw denim", "fashion"), ("streetwear", "fashion"),
    ("art", "art"), ("Paintings", "art"), ("illustration refs", "art"), ("posters", "art"),
    ("recipes", "other"), ("apartment", "other"), ("building facades", "other"),
    ("party ideas", "other"), ("fruit", "other"), ("", "other"),
])
def test_guess_domain(name, domain):
    assert guess_domain(name) == domain


def test_most_hits_wins_and_ties_follow_order():
    assert guess_domain("film photo posters") == "photo"
    assert guess_domain("web art") == "web"


def test_update_boards_creates_file_from_pins(tmp_path):
    store = Store(tmp_path).ensure()
    store.write_pins([Pin(id=f"local:{i}", source="local", board=b)
                      for i, b in enumerate(["web inspo", "web inspo", "film photo", "misc"])])
    boards, added = update_boards(store)
    assert added == ["web inspo", "film photo", "misc"]
    assert {k: v.domain for k, v in boards.items()} == {
        "web inspo": "web", "film photo": "photo", "misc": "other"}
    assert store.read_boards() == boards and all(e.include for e in boards.values())


def test_update_boards_never_overwrites_user_entries(tmp_path):
    store = Store(tmp_path).ensure()
    store.write_boards({"web inspo": BoardEntry(domain="art", include=False),
                        "gone board": BoardEntry(domain="photo")})
    store.write_pins([Pin(id="local:1", source="local", board="web inspo"),
                      Pin(id="local:2", source="local", board="fits")])
    boards, added = update_boards(store)
    assert added == ["fits"]
    saved = store.read_boards()
    assert saved["web inspo"] == BoardEntry(domain="art", include=False)  # heuristic says web
    assert saved["gone board"] == BoardEntry(domain="photo")  # no pins, still kept
    assert saved["fits"] == BoardEntry(domain="fashion")


def test_update_boards_noop_leaves_file_bytes_alone(tmp_path):
    store = make_fixture_home(tmp_path, n_per_group=1)
    text = "# my notes\n" + store.boards_path.read_text()
    store.boards_path.write_text(text)
    _boards, added = update_boards(store)
    assert added == [] and store.boards_path.read_text() == text
