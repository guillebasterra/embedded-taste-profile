from __future__ import annotations

import json
import stat

import httpx
import respx
from typer.testing import CliRunner

from taste.cli import app
from taste.ingest import oauth
from taste.models import BoardEntry
from taste.store import Store

from .conftest import API, image_bytes, raw_pin
from .test_sources import RSS, make_folder, page

runner = CliRunner()
CDN = "https://i.pinimg.com"


def run(*args):
    return runner.invoke(app, [str(a) for a in args])


def test_commands_are_registered():
    out = run("--help").output
    assert all(cmd in out for cmd in ("auth", "ingest", "domains"))
    for cmd in ("auth", "ingest", "domains"):
        assert "--home" in run(cmd, "--help").output


# ---- auth -------------------------------------------------------------------------

def test_auth_without_app_credentials(home):
    result = run("auth", "--home", home, "--no-browser")
    assert result.exit_code == 1 and "PINTEREST_APP_ID" in result.output


def test_auth_env_token_short_circuits(home, monkeypatch):
    monkeypatch.setenv("PINTEREST_ACCESS_TOKEN", "ENV-TOKEN-VALUE")
    monkeypatch.setattr(oauth, "wait_for_callback",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no listener")))
    result = run("auth", "--home", home)
    assert result.exit_code == 0 and "Nothing to do" in result.output
    assert "ENV-TOKEN-VALUE" not in result.output
    assert not (home / "auth" / "pinterest.json").exists()


@respx.mock
def test_auth_full_flow_saves_tokens_and_prints_no_secrets(home, monkeypatch):
    home.mkdir()
    (home / ".env").write_text("PINTEREST_APP_ID=app123\nPINTEREST_APP_SECRET=s3cr3t-value\n")
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
    captured = {}

    def fake_wait(state, timeout=300.0, on_ready=None, **kw):
        captured["state"] = state
        on_ready(8085)
        return "one-time-code"

    monkeypatch.setattr(oauth, "wait_for_callback", fake_wait)
    token_route = respx.post(oauth.TOKEN_URL).mock(return_value=httpx.Response(200, json={
        "access_token": "ACCESS-XYZ", "refresh_token": "REFRESH-XYZ", "expires_in": 3600,
        "refresh_token_expires_in": 7200}))

    result = run("auth", "--home", home)
    assert result.exit_code == 0, result.output
    assert "https://www.pinterest.com/oauth/?client_id=app123" in result.output
    assert f"state={captured['state']}" in result.output
    assert opened and opened[0].startswith("https://www.pinterest.com/oauth/")
    assert b"code=one-time-code" in token_route.calls.last.request.content
    for secret in ("ACCESS-XYZ", "REFRESH-XYZ", "s3cr3t-value", "one-time-code"):
        assert secret not in result.output
    path = home / "auth" / "pinterest.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text())["access_token"] == "ACCESS-XYZ"

    opened.clear()
    assert run("auth", "--home", home, "--no-browser").exit_code == 0
    assert opened == []


@respx.mock
def test_auth_reports_state_mismatch_and_exchange_failure(home, monkeypatch):
    monkeypatch.setenv("PINTEREST_APP_ID", "app123")
    monkeypatch.setenv("PINTEREST_APP_SECRET", "s3cr3t-value")

    def bad_state(state, **kw):
        return oauth.parse_callback("/callback?code=c&state=forged", state)

    monkeypatch.setattr(oauth, "wait_for_callback", bad_state)
    result = run("auth", "--home", home, "--no-browser")
    assert result.exit_code == 1 and "state mismatch" in result.output

    monkeypatch.setattr(oauth, "wait_for_callback", lambda state, **kw: "c")
    respx.post(oauth.TOKEN_URL).mock(return_value=httpx.Response(401, json={"message": "bad"}))
    result = run("auth", "--home", home, "--no-browser")
    assert result.exit_code == 1 and "401" in result.output
    assert "s3cr3t-value" not in result.output
    assert not (home / "auth" / "pinterest.json").exists()


# ---- ingest: local ----------------------------------------------------------------

def test_ingest_local_end_to_end_then_domains(tmp_path, home, monkeypatch):
    root = tmp_path / "moodboard"
    root.mkdir()
    make_folder(root)
    monkeypatch.setenv("TASTE_HOME", str(home))  # --home default comes from the environment

    first = run("ingest", "--source", "local", "--path", root)
    assert first.exit_code == 0, first.output
    assert "4 pins seen, 4 new, 4 images saved" in first.output
    assert "Skipped 1 file(s)" in first.output
    store = Store(home)
    pins = store.read_pins()
    assert len(pins) == 4 and all(p.id.startswith("local:") for p in pins)
    assert all(store.abs(p.image_path).exists() and p.width and p.height for p in pins)
    assert {p.board for p in pins} == {"web inspo", "fits", "moodboard"}

    second = run("ingest", "--source", "local", "--path", root)
    assert "4 pins seen, 0 new, 0 images saved, 4 already on disk" in second.output
    assert len(store.read_pins()) == 4 and len(list(store.images_dir.iterdir())) == 4

    result = run("domains")
    assert result.exit_code == 0, result.output
    assert "3 board(s) added" in result.output and "boards.toml" in result.output
    assert {k: v.domain for k, v in store.read_boards().items()} == {
        "web inspo": "web", "fits": "fashion", "moodboard": "other"}
    assert len(store.pin_domains()) == 4

    boards = store.read_boards()
    boards["moodboard"] = BoardEntry(domain="art", include=False)
    store.write_boards(boards)
    again = run("domains")
    assert "0 board(s) added, 3 kept" in again.output
    assert store.read_boards()["moodboard"] == BoardEntry(domain="art", include=False)


def test_ingest_argument_errors(home):
    assert run("ingest", "--home", home, "--source", "local").exit_code == 1
    assert run("ingest", "--home", home, "--source", "public").exit_code == 1
    assert run("ingest", "--home", home, "--source", "local", "--path",
               home / "missing").exit_code == 1
    assert run("ingest", "--home", home, "--source", "ftp").exit_code == 2


def test_domains_with_no_pins(home):
    result = run("domains", "--home", home)
    assert result.exit_code == 0 and "taste ingest" in result.output


# ---- ingest: api ------------------------------------------------------------------

def mock_api(b2_pins=None):
    respx.get(f"{API}/boards").mock(return_value=page(
        [{"id": "b1", "name": "Web Inspo"}, {"id": "b2", "name": "fits"}]))
    respx.get(f"{API}/boards/b1/sections").mock(return_value=page([]))
    respx.get(f"{API}/boards/b2/sections").mock(return_value=page([]))
    b1 = respx.get(f"{API}/boards/b1/pins").mock(
        return_value=page([raw_pin("1", "b1"), raw_pin("2", "b1")]))
    b2 = respx.get(f"{API}/boards/b2/pins").mock(
        return_value=b2_pins or page([raw_pin("3", "b2"), raw_pin("4", "b2", media=None)]))
    images = respx.get(url__regex=rf"{CDN}/1200x/\d+\.jpg").mock(
        side_effect=lambda request: httpx.Response(200, content=image_bytes(size=(64, 48))))
    return b1, b2, images


def test_ingest_api_needs_login(home):
    result = run("ingest", "--home", home)
    assert result.exit_code == 1 and "taste auth" in result.output


@respx.mock
def test_ingest_api_end_to_end_and_incremental_rerun(home, monkeypatch):
    monkeypatch.setenv("PINTEREST_ACCESS_TOKEN", "ENV-TOKEN-VALUE")
    b1, b2, images = mock_api()

    first = run("ingest", "--home", home)
    assert first.exit_code == 0, first.output
    assert "4 pins seen, 4 new, 3 images saved" in first.output
    assert "1 pin(s) have no image" in first.output
    assert "ENV-TOKEN-VALUE" not in first.output
    assert b1.calls.last.request.headers["Authorization"] == "Bearer ENV-TOKEN-VALUE"
    store = Store(home)
    by_id = {p.id: p for p in store.read_pins()}
    assert set(by_id) == {"pin:1", "pin:2", "pin:3", "pin:4"}
    assert by_id["pin:1"].image_url.endswith("/1200x/1.jpg")
    assert (by_id["pin:1"].image_path, by_id["pin:1"].width, by_id["pin:1"].height) == \
        ("images/pin-1.jpg", 64, 48)
    assert by_id["pin:4"].image_path is None
    assert images.call_count == 3
    assert not (home / "ingest_state.json").exists()

    b2.mock(return_value=page([raw_pin("3", "b2"), raw_pin("5", "b2")]))
    second = run("ingest", "--home", home)
    assert second.exit_code == 0, second.output
    assert "1 new, 1 images saved, 3 already on disk" in second.output
    assert images.call_count == 4  # only the new pin's image was fetched
    assert len(store.read_pins()) == 5


@respx.mock
def test_ingest_api_refreshes_saved_token_on_401(home, monkeypatch):
    monkeypatch.setenv("PINTEREST_APP_ID", "app123")
    monkeypatch.setenv("PINTEREST_APP_SECRET", "s3cr3t-value")
    store = Store(home).ensure()
    oauth.save_tokens(store, oauth.TokenSet("STALE", "REFRESH-1", expires_at=None))
    mock_api()
    respx.get(f"{API}/boards").mock(side_effect=lambda request: (
        page([{"id": "b1", "name": "Web Inspo"}])
        if request.headers["Authorization"] == "Bearer FRESH" else httpx.Response(401, json={})))
    refresh = respx.post(oauth.TOKEN_URL).mock(return_value=httpx.Response(
        200, json={"access_token": "FRESH", "expires_in": 3600}))

    result = run("ingest", "--home", home, "--board", "web inspo")
    assert result.exit_code == 0, result.output
    assert refresh.call_count == 1
    assert oauth.load_tokens(store).access_token == "FRESH"
    assert "FRESH" not in result.output and "STALE" not in result.output
    assert len(store.read_pins()) == 2


@respx.mock
def test_ingest_api_daily_cap_stops_cleanly_and_resumes(home, monkeypatch):
    monkeypatch.setenv("PINTEREST_ACCESS_TOKEN", "ENV-TOKEN-VALUE")
    b1, b2, images = mock_api(b2_pins=httpx.Response(429, headers={"Retry-After": "43200"}))

    first = run("ingest", "--home", home)
    assert first.exit_code == 3, first.output
    assert "rate limit reached" in first.output and "Progress is saved" in first.output
    assert "about 12 hours" in first.output
    store = Store(home)
    assert {p.id for p in store.read_pins()} == {"pin:1", "pin:2"}
    assert all(p.image_path for p in store.read_pins())
    assert json.loads((home / "ingest_state.json").read_text()) == {"done_board_ids": ["b1"]}

    b2.mock(return_value=page([raw_pin("3", "b2")]))
    second = run("ingest", "--home", home)
    assert second.exit_code == 0, second.output
    assert "Skipped 1 board(s)" in second.output
    assert b1.call_count == 1  # the finished board cost no requests on resume
    assert {p.id for p in store.read_pins()} == {"pin:1", "pin:2", "pin:3"}
    assert not (home / "ingest_state.json").exists()


@respx.mock
def test_ingest_api_limit_and_max_requests(home, monkeypatch):
    monkeypatch.setenv("PINTEREST_ACCESS_TOKEN", "ENV-TOKEN-VALUE")
    mock_api()
    limited = run("ingest", "--home", home, "--limit", "1")
    assert limited.exit_code == 0 and "1 pins seen" in limited.output
    assert len(Store(home).read_pins()) == 1
    capped = run("ingest", "--home", home, "--max-requests", "3")  # boards, sections, b1 pins
    assert capped.exit_code == 3 and "rate limit" in capped.output
    assert len(Store(home).read_pins()) == 2


# ---- ingest: public ---------------------------------------------------------------

@respx.mock
def test_ingest_public_board(home):
    respx.get("https://www.pinterest.com/someone/brutal-web.rss").mock(
        return_value=httpx.Response(200, text=RSS))
    respx.get(f"{CDN}/736x/aa/bb/cc/one.jpg").mock(
        return_value=httpx.Response(200, content=image_bytes(size=(73, 60))))
    respx.get(f"{CDN}/736x/dd/two.jpg").mock(return_value=httpx.Response(403))
    small = respx.get(f"{CDN}/236x/dd/two.jpg").mock(
        return_value=httpx.Response(200, content=image_bytes(size=(23, 20))))

    result = run("ingest", "--home", home, "--source", "public", "--url",
                 "https://www.pinterest.com/someone/brutal-web/")
    assert result.exit_code == 0, result.output
    assert "partial" in result.output and "2 images saved" in result.output
    by_id = {p.id: p for p in Store(home).read_pins()}
    assert set(by_id) == {"pub:111", "pub:222"}
    assert (by_id["pub:111"].width, by_id["pub:111"].image_path) == (73, "images/pub-111.jpg")
    assert by_id["pub:222"].width == 23 and small.call_count == 1
    assert by_id["pub:111"].board == "Brutal Web"
