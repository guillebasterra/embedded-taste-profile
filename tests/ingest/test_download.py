from __future__ import annotations

import httpx
import respx

from taste.ingest.download import download_images
from taste.ingest.pipeline import persist
from taste.models import Pin
from taste.store import Store

from .conftest import image_bytes

CDN = "https://i.pinimg.com"


def pin(pid: str, **kw) -> Pin:
    return Pin(id=f"pin:{pid}", source="pinterest_api", board="b",
               image_url=f"{CDN}/1200x/{pid}.jpg", **kw)


@respx.mock
def test_downloads_verifies_and_records_size(home):
    store = Store(home).ensure()
    respx.get(f"{CDN}/1200x/1.jpg").mock(
        return_value=httpx.Response(200, content=image_bytes(size=(40, 30))))
    respx.get(f"{CDN}/1200x/2.jpg").mock(  # extension comes from the bytes, not the URL
        return_value=httpx.Response(200, content=image_bytes(size=(8, 9), fmt="PNG")))
    with httpx.Client() as http:
        results = download_images([pin("1"), pin("2")], store, http, sleep=lambda s: None)
    assert [r.status for r in results] == ["downloaded", "downloaded"]
    one, two = (r.pin for r in results)
    assert (one.image_path, one.width, one.height) == ("images/pin-1.jpg", 40, 30)
    assert (two.image_path, two.width, two.height) == ("images/pin-2.png", 8, 9)
    assert store.abs(one.image_path).exists() and not list(store.images_dir.glob("*.part"))


@respx.mock
def test_retries_transient_failures(home):
    store = Store(home).ensure()
    route = respx.get(f"{CDN}/1200x/1.jpg").mock(side_effect=[
        httpx.Response(503), httpx.ConnectError("reset"),
        httpx.Response(200, content=image_bytes())])
    sleeps: list[float] = []
    with httpx.Client() as http:
        [res] = download_images([pin("1")], store, http, sleep=sleeps.append)
    assert res.status == "downloaded" and route.call_count == 3 and len(sleeps) == 2


@respx.mock
def test_404_and_garbage_fail_without_leaving_files(home):
    store = Store(home).ensure()
    gone = respx.get(f"{CDN}/1200x/1.jpg").mock(return_value=httpx.Response(404))
    respx.get(f"{CDN}/1200x/2.jpg").mock(
        return_value=httpx.Response(200, content=b"<html>not an image</html>"))
    with httpx.Client() as http:
        results = download_images([pin("1"), pin("2"), pin("3", ).model_copy(
            update={"image_url": None})], store, http, sleep=lambda s: None)
    assert [r.status for r in results] == ["failed"] * 3
    assert gone.call_count == 1  # 404 is not retried
    assert results[0].error == "HTTP 404" and "valid image" in results[1].error
    assert all(r.pin.image_path is None for r in results)
    assert list(store.images_dir.iterdir()) == []


@respx.mock
def test_skips_existing_files_and_refetches_corrupt_ones(home):
    store = Store(home).ensure()
    (store.images_dir / "pin-1.jpg").write_bytes(image_bytes(size=(12, 34)))
    (store.images_dir / "pin-2.jpg").write_bytes(b"truncated")
    r1 = respx.get(f"{CDN}/1200x/1.jpg").mock(return_value=httpx.Response(500))
    r2 = respx.get(f"{CDN}/1200x/2.jpg").mock(
        return_value=httpx.Response(200, content=image_bytes(size=(5, 6))))
    with httpx.Client() as http:
        a, b = download_images([pin("1"), pin("2")], store, http, sleep=lambda s: None)
    assert a.status == "skipped" and (a.pin.width, a.pin.height) == (12, 34)
    assert r1.call_count == 0
    assert b.status == "downloaded" and (b.pin.width, b.pin.height) == (5, 6)
    assert r2.call_count == 1


@respx.mock
def test_fallback_url_is_used_when_the_first_fails(home):
    store = Store(home).ensure()
    respx.get(f"{CDN}/1200x/1.jpg").mock(return_value=httpx.Response(403))
    respx.get(f"{CDN}/236x/1.jpg").mock(return_value=httpx.Response(200, content=image_bytes()))
    with httpx.Client() as http:
        [res] = download_images([pin("1")], store, http, sleep=lambda s: None,
                                fallbacks={"pin:1": [f"{CDN}/236x/1.jpg"]})
    assert res.status == "downloaded" and res.pin.image_url == f"{CDN}/236x/1.jpg"


@respx.mock
def test_persist_is_incremental(home):
    store = Store(home).ensure()
    routes = [respx.get(f"{CDN}/1200x/{i}.jpg").mock(
        return_value=httpx.Response(200, content=image_bytes(size=(20 + i, 10))))
        for i in (1, 2, 3)]
    with httpx.Client() as http:
        first = persist(store, [pin("1"), pin("2")], http)
        assert (first.new, first.downloaded, first.already, first.total) == (2, 2, 0, 2)
        second = persist(store, [pin("1", title="renamed"), pin("2"), pin("3")], http)
    assert (second.new, second.downloaded, second.already, second.total) == (1, 1, 2, 3)
    assert [r.call_count for r in routes] == [1, 1, 1]
    by_id = {p.id: p for p in store.read_pins()}
    assert by_id["pin:1"].title == "renamed"  # metadata refreshes, image and size are kept
    assert (by_id["pin:1"].image_path, by_id["pin:1"].width) == ("images/pin-1.jpg", 21)


@respx.mock
def test_persist_records_pins_without_images_and_failures(home):
    store = Store(home).ensure()
    respx.get(f"{CDN}/1200x/1.jpg").mock(return_value=httpx.Response(404))
    no_image = Pin(id="pin:9", source="pinterest_api", board="b")
    with httpx.Client() as http:
        stats = persist(store, [pin("1"), no_image], http)
    assert (stats.failed, stats.no_image, stats.total) == (1, 1, 2)
    assert stats.errors == ["pin:1: HTTP 404"]
    assert store.pin_domains() == {}  # neither counts as an image pin downstream
