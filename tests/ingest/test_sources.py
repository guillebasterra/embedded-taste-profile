from __future__ import annotations

import hashlib
import json
import shutil

import httpx
import pytest
import respx
from PIL import Image

from taste.ingest import sources
from taste.ingest.errors import IngestError
from taste.ingest.pinterest import PinterestClient
from taste.store import Store

from .conftest import API, image_bytes, raw_pin


class StaticToken:
    def access_token(self):
        return "T"

    def refresh(self):
        return False


def page(items, bookmark=None):
    return httpx.Response(200, json={"items": items, "bookmark": bookmark})


def mock_account():
    """Two boards; b1 has a section and paginated pins, b2 is flat."""
    respx.get(f"{API}/boards").mock(return_value=page(
        [{"id": "b1", "name": "Web Inspo"}, {"id": "b2", "name": "fits"}]))
    respx.get(f"{API}/boards/b1/sections").mock(return_value=page([{"id": "s1", "name": "Type"}]))
    respx.get(f"{API}/boards/b2/sections").mock(return_value=page([]))

    def b1_pins(request):
        if request.url.params.get("bookmark") == "p2":
            return page([raw_pin("3", "b1", "s1")])
        return page([raw_pin("1", "b1"), raw_pin("2", "b1")], "p2")

    respx.get(f"{API}/boards/b1/pins").mock(side_effect=b1_pins)
    respx.get(f"{API}/boards/b1/sections/s1/pins").mock(
        return_value=page([raw_pin("3", "b1", "s1"), raw_pin("4", "b1", "s1")]))
    respx.get(f"{API}/boards/b2/pins").mock(return_value=page([raw_pin("5", "b2")]))


def client_for(http, **kw):
    return PinterestClient(StaticToken(), http, sleep=lambda s: None, **kw)


def test_largest_image_by_area_and_missing_media():
    assert sources.largest_image(raw_pin("1"))["url"].endswith("/1200x/1.jpg")
    for media in (None, {}, {"images": None}, {"images": {}}, {"images": {"x": {"width": 1}}},
                  "video"):
        assert sources.largest_image({"id": "1", "media": media}) is None
    assert sources.largest_image({"id": "1"}) is None


def test_pin_from_api_maps_fields():
    pin = sources.pin_from_api(raw_pin("77", "b1", "s1", title="", alt_text="a chair"),
                               "Web Inspo", "b1", {"s1": "Type"})
    assert pin.id == "pin:77" and pin.source == "pinterest_api"
    assert (pin.board, pin.board_id, pin.section) == ("Web Inspo", "b1", "Type")
    assert pin.title is None and pin.alt_text == "a chair"
    assert pin.link == "https://example.com/77"
    assert (pin.width, pin.height) == (1200, 900) and pin.image_path is None
    assert pin.created_at.year == 2024
    bare = sources.pin_from_api({"id": "8", "created_at": "garbage"}, "B")
    assert bare.image_url is None and bare.created_at is None and bare.section is None


@respx.mock
def test_fetch_api_walks_boards_sections_and_pages():
    mock_account()
    seen = []
    with httpx.Client() as http:
        result = sources.fetch_api(client_for(http),
                                   on_board=lambda b, pins, ok: seen.append((b["id"], len(pins), ok)))
    assert [p.id for p in result.pins] == ["pin:1", "pin:2", "pin:3", "pin:4", "pin:5"]
    by_id = {p.id: p for p in result.pins}
    assert by_id["pin:1"].section is None
    assert by_id["pin:3"].section == "Type" and by_id["pin:4"].section == "Type"
    assert by_id["pin:5"].board == "fits"
    assert result.boards_done == ["b1", "b2"] and not result.stopped and not result.limit_hit
    assert seen == [("b1", 4, True), ("b2", 1, True)]


@respx.mock
def test_fetch_api_board_filter_and_unknown_board():
    mock_account()
    with httpx.Client() as http:
        by_name = sources.fetch_api(client_for(http), boards=["FITS"])
        by_id = sources.fetch_api(client_for(http), boards=["b2"])
        with pytest.raises(IngestError, match="No such board: nope"):
            sources.fetch_api(client_for(http), boards=["nope"])
    assert [p.id for p in by_name.pins] == ["pin:5"] == [p.id for p in by_id.pins]


@respx.mock
def test_fetch_api_limit():
    mock_account()
    with httpx.Client() as http:
        result = sources.fetch_api(client_for(http), limit=3)
    assert [p.id for p in result.pins] == ["pin:1", "pin:2", "pin:3"]
    assert result.limit_hit and result.boards_done == []


@respx.mock
def test_fetch_api_rate_limit_keeps_partial_progress():
    mock_account()
    respx.get(f"{API}/boards/b1/sections/s1/pins").mock(
        return_value=httpx.Response(429, headers={"Retry-After": "86400"}))
    seen = []
    with httpx.Client() as http:
        result = sources.fetch_api(client_for(http),
                                   on_board=lambda b, pins, ok: seen.append((b["id"], len(pins), ok)))
    assert result.stopped and result.retry_after == 86400.0
    assert [p.id for p in result.pins] == ["pin:1", "pin:2", "pin:3"]
    assert result.boards_done == [] and seen == [("b1", 3, False)]


@respx.mock
def test_fetch_api_skips_boards_done_earlier():
    mock_account()
    with httpx.Client() as http:
        result = sources.fetch_api(client_for(http), skip_board_ids={"b1"})
    assert [p.id for p in result.pins] == ["pin:5"] and result.boards_skipped == 1
    assert not respx.get(f"{API}/boards/b1/pins").called


# ---- public -----------------------------------------------------------------------

RSS = """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"><channel><title>Brutal Web</title>
<link>https://www.pinterest.com/someone/brutal-web/</link>
<item><title>Grid study</title><link>https://www.pinterest.com/pin/111/</link>
<description>&lt;a href="https://www.pinterest.com/pin/111/"&gt;&lt;img src="https://i.pinimg.com/236x/aa/bb/cc/one.jpg"&gt;&lt;/a&gt;A &amp;amp; B grid</description>
<pubDate>Mon, 02 Jun 2025 10:00:00 +0000</pubDate><guid>https://www.pinterest.com/pin/111/</guid></item>
<item><title></title><link>https://www.pinterest.com/pin/222/</link>
<description>&lt;img src='https://i.pinimg.com/236x/dd/two.jpg'&gt;</description></item>
<item><title>dupe</title><link>https://www.pinterest.com/pin/111/</link><description/></item>
<item><title>not a pin</title><link>https://www.pinterest.com/ideas/</link></item>
</channel></rss>"""


def test_rss_url_validation():
    assert sources.rss_url("https://www.pinterest.com/someone/brutal-web/?x=1") == \
        "https://www.pinterest.com/someone/brutal-web.rss"
    assert sources.rss_url("https://es.pinterest.com/a/b") == "https://es.pinterest.com/a/b.rss"
    for bad in ("https://example.com/a/b/", "https://www.pinterest.com/someone/", "nonsense"):
        with pytest.raises(IngestError):
            sources.rss_url(bad)


def test_parse_board_rss():
    pins, fallbacks = sources.parse_board_rss(RSS, "brutal-web")
    assert [p.id for p in pins] == ["pub:111", "pub:222"]
    one, two = pins
    assert one.source == "public_board" and one.board == "Brutal Web"
    assert one.title == "Grid study" and one.description == "A & B grid"
    assert one.image_url == "https://i.pinimg.com/736x/aa/bb/cc/one.jpg"
    assert one.link == "https://www.pinterest.com/pin/111/" and one.created_at.year == 2025
    assert two.title is None and two.created_at is None
    assert fallbacks == {"pub:111": ["https://i.pinimg.com/236x/aa/bb/cc/one.jpg"],
                         "pub:222": ["https://i.pinimg.com/236x/dd/two.jpg"]}


def test_parse_board_rss_rejects_html_and_entities():
    with pytest.raises(IngestError):
        sources.parse_board_rss("<html><body>login</body></html>", "b")
    with pytest.raises(IngestError):
        sources.parse_board_rss("<body><p>unclosed", "b")
    with pytest.raises(IngestError):
        sources.parse_board_rss('<!DOCTYPE x [<!ENTITY a "b">]><rss><channel/></rss>', "b")


@respx.mock
def test_fetch_public_and_errors():
    feed = respx.get("https://www.pinterest.com/someone/brutal-web.rss").mock(
        return_value=httpx.Response(200, text=RSS))
    respx.get("https://www.pinterest.com/someone/secret.rss").mock(
        return_value=httpx.Response(404))
    with httpx.Client() as http:
        pins, fallbacks = sources.fetch_public(
            http, "https://www.pinterest.com/someone/brutal-web/", limit=1)
        with pytest.raises(IngestError, match="HTTP 404"):
            sources.fetch_public(http, "https://www.pinterest.com/someone/secret/")
    assert feed.called and [p.id for p in pins] == ["pub:111"] and list(fallbacks) == ["pub:111"]


# ---- local ------------------------------------------------------------------------

def make_folder(root):
    (root / "web inspo" / "dark").mkdir(parents=True)
    (root / "fits").mkdir()
    (root / ".hidden").mkdir()
    files = {
        "web inspo/a.jpg": image_bytes((10, 10, 10), (30, 20)),
        "web inspo/dark/b.png": image_bytes((20, 20, 20), (16, 16), "PNG"),
        "fits/c.JPG": image_bytes((200, 30, 30), (11, 13)),
        "fits/c-copy.jpg": image_bytes((200, 30, 30), (11, 13)),  # byte-identical duplicate
        "loose.webp": image_bytes((1, 2, 3), (9, 9), "WEBP"),
        "fits/broken.jpg": b"not an image",
        "fits/notes.txt": b"hello",
        ".hidden/x.jpg": image_bytes(),
    }
    for rel, data in files.items():
        (root / rel).write_bytes(data)
    return files


def test_fetch_local(tmp_path, home):
    root = tmp_path / "moodboard"
    root.mkdir()
    files = make_folder(root)
    store = Store(home).ensure()
    result = sources.fetch_local(store, root)
    by_title = {p.title: p for p in result.pins}
    assert set(by_title) == {"a", "b", "c-copy", "loose"}  # sorted walk: c-copy wins over c
    a = by_title["a"]
    digest = hashlib.sha1(files["web inspo/a.jpg"]).hexdigest()[:16]
    assert a.id == f"local:{digest}" and a.source == "local"
    assert (a.board, a.section, a.width, a.height) == ("web inspo", None, 30, 20)
    assert a.image_path == f"images/local-{digest}.jpg"
    assert store.abs(a.image_path).read_bytes() == files["web inspo/a.jpg"]
    assert (by_title["b"].board, by_title["b"].section) == ("web inspo", "dark")
    assert by_title["b"].image_path.endswith(".png")
    assert by_title["loose"].board == "moodboard"
    assert result.invalid == ["fits/broken.jpg"]
    assert (result.copied, result.already) == (4, 0)
    with Image.open(store.abs(by_title["loose"].image_path)) as im:
        assert im.size == (9, 9)

    again = sources.fetch_local(store, root)
    assert (again.copied, again.already) == (0, 4)
    assert [p.id for p in again.pins] == [p.id for p in result.pins]
    assert len(sources.fetch_local(store, root, limit=2).pins) == 2


def test_fetch_local_missing_dir(tmp_path, home):
    with pytest.raises(IngestError, match="Not a directory"):
        sources.fetch_local(Store(home), tmp_path / "nope")


def test_fetch_local_reads_gallery_dl_sidecars(tmp_path, home):
    # trimmed from a real gallery-dl --write-metadata sidecar
    board = tmp_path / "pinterest" / "me" / "Agent Crowd"
    board.mkdir(parents=True)
    (board / "pinterest_e1_abc.jpg").write_bytes(image_bytes((30, 90, 40), (12, 20)))
    (board / "pinterest_e1_abc.jpg.json").write_text(json.dumps({
        "id": "e1_abc", "title": "", "grid_title": "Tadpoles", "description": "",
        "alt_text": None, "auto_alt_text": "green bubbles with small yellow birds in them",
        "link": None, "url": "https://i.pinimg.com/originals/33/ab/ba/33ab.jpg",
        "created_at": "Sat, 21 Nov 2026 16:44:58 +0000",
        "board": {"id": "569635121558272071", "name": "Agent Crowd"}}))
    (board / "plain.jpg").write_bytes(image_bytes((1, 1, 1)))
    (board / "broken.jpg").write_bytes(image_bytes((2, 2, 2)))
    (board / "broken.jpg.json").write_text("{not json")
    pins = {p.title: p for p in sources.fetch_local(Store(home).ensure(),
                                                    tmp_path / "pinterest" / "me").pins}
    assert set(pins) == {"Tadpoles", "plain", "broken"}
    tad = pins["Tadpoles"]
    assert tad.id.startswith("local:") and tad.board == "Agent Crowd"
    assert tad.board_id == "569635121558272071"
    assert tad.alt_text == "green bubbles with small yellow birds in them"
    assert tad.description is None and tad.link is None
    assert tad.image_url.endswith("33ab.jpg") and tad.created_at.year == 2026
    assert pins["plain"].board_id is None and pins["plain"].alt_text is None


def _heic_folder(tmp_path):
    root = tmp_path / "boards"
    (root / "b").mkdir(parents=True)
    (root / "b" / "x.heic").write_bytes(b"pretend heic")
    (root / "b" / "clip.m4v").write_bytes(b"video")
    (root / "b" / "x.heic.json").write_text("{}")
    (root / "b" / "y.jpg.part").write_bytes(b"partial")
    (root / "b" / "ok.jpg").write_bytes(image_bytes((9, 9, 9)))
    return root


def test_fetch_local_converts_heic_and_reports_unsupported(tmp_path, home, monkeypatch):
    def fake_convert(src, dest):
        Image.new("RGB", (7, 5), (1, 2, 3)).save(dest, "JPEG")

    monkeypatch.setattr(sources, "heif_converter", lambda: fake_convert)
    result = sources.fetch_local(Store(home).ensure(), _heic_folder(tmp_path))
    heic = next(p for p in result.pins if p.title == "x")
    digest = hashlib.sha1(b"pretend heic").hexdigest()[:16]
    assert heic.id == f"local:{digest}" and heic.image_path == f"images/local-{digest}.jpg"
    assert (heic.width, heic.height) == (7, 5)
    assert result.unsupported == {".m4v": 1} and result.copied == 2


def test_fetch_local_without_converter_skips_heic(tmp_path, home, monkeypatch):
    monkeypatch.setattr(sources, "heif_converter", lambda: None)
    result = sources.fetch_local(Store(home).ensure(), _heic_folder(tmp_path))
    assert [p.title for p in result.pins] == ["ok"]
    assert result.unsupported == {".heic": 1, ".m4v": 1}


def test_fetch_local_failed_conversion_is_invalid(tmp_path, home, monkeypatch):
    def broken(src, dest):
        dest.write_bytes(b"half")
        raise RuntimeError("sips failed")

    monkeypatch.setattr(sources, "heif_converter", lambda: broken)
    store = Store(home).ensure()
    result = sources.fetch_local(store, _heic_folder(tmp_path))
    assert result.invalid == ["b/x.heic"] and [p.title for p in result.pins] == ["ok"]
    assert len(list(store.images_dir.iterdir())) == 1  # the half-written jpg was removed


@pytest.mark.skipif(shutil.which("sips") is None, reason="needs macOS sips")
def test_heif_converter_with_real_sips(tmp_path):
    import subprocess

    src_jpg = tmp_path / "a.jpg"
    Image.new("RGB", (40, 30), (200, 10, 10)).save(src_jpg)
    heic = tmp_path / "a.heic"
    subprocess.run(["sips", "-s", "format", "heic", str(src_jpg), "--out", str(heic)],
                   check=True, capture_output=True)
    out = tmp_path / "out.jpg"
    sources.heif_converter()(heic, out)
    with Image.open(out) as im:
        assert im.format == "JPEG" and im.size == (40, 30)
