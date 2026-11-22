from __future__ import annotations

import base64
import json
import stat
import threading
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import respx

from taste.config import PINTEREST_REDIRECT_URI, PINTEREST_SCOPES
from taste.ingest import oauth
from taste.ingest.errors import AuthError
from taste.store import Store

from .conftest import make_settings

TOKEN_BODY = {"access_token": "ACCESS-1", "refresh_token": "REFRESH-1", "token_type": "bearer",
              "expires_in": 2592000, "refresh_token_expires_in": 31536000, "scope": "pins:read"}


def test_auth_url_has_every_parameter():
    url = oauth.build_auth_url("app123", "st4te")
    parts = urlsplit(url)
    assert f"{parts.scheme}://{parts.netloc}{parts.path}" == "https://www.pinterest.com/oauth/"
    q = parse_qs(parts.query)
    assert q == {"client_id": ["app123"], "redirect_uri": [PINTEREST_REDIRECT_URI],
                 "response_type": ["code"], "scope": [",".join(PINTEREST_SCOPES)],
                 "state": ["st4te"]}


def test_state_is_random():
    assert oauth.new_state() != oauth.new_state()
    assert len(oauth.new_state()) >= 24


def test_parse_callback_checks_state():
    assert oauth.parse_callback("/callback?code=abc&state=good", "good") == "abc"
    with pytest.raises(AuthError, match="state mismatch"):
        oauth.parse_callback("/callback?code=abc&state=evil", "good")
    with pytest.raises(AuthError, match="state mismatch"):
        oauth.parse_callback("/callback?code=abc", "good")
    with pytest.raises(AuthError, match="denied"):
        oauth.parse_callback("/callback?error=access_denied&state=good", "good")
    with pytest.raises(AuthError, match="no authorization code"):
        oauth.parse_callback("/callback?state=good", "good")


def _listen(state: str, **kw):
    """Run the listener on a free port; returns (port, result dict, thread)."""
    ready = threading.Event()
    box: dict = {}

    def run():
        try:
            box["code"] = oauth.wait_for_callback(
                state, port=0, timeout=10,
                on_ready=lambda p: (box.__setitem__("port", p), ready.set()), **kw)
        except Exception as exc:
            box["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert ready.wait(5)
    return box, thread


def test_callback_listener_returns_code():
    box, thread = _listen("good")
    with httpx.Client(trust_env=False) as c:
        base = f"http://127.0.0.1:{box['port']}"
        assert c.get(f"{base}/favicon.ico").status_code == 404  # stray requests are ignored
        resp = c.get(f"{base}/callback", params={"code": "the-code", "state": "good"})
    thread.join(5)
    assert resp.status_code == 200 and "the-code" not in resp.text
    assert box["code"] == "the-code"


def test_callback_listener_rejects_wrong_state():
    box, thread = _listen("good")
    with httpx.Client(trust_env=False) as c:
        resp = c.get(f"http://127.0.0.1:{box['port']}/callback",
                     params={"code": "x", "state": "forged"})
    thread.join(5)
    assert resp.status_code == 400
    assert isinstance(box["error"], AuthError)


def test_callback_listener_times_out():
    with pytest.raises(AuthError, match="Timed out"):
        oauth.wait_for_callback("s", port=0, timeout=0.05)


@respx.mock
def test_exchange_code_sends_basic_auth_and_form():
    route = respx.post(oauth.TOKEN_URL).mock(return_value=httpx.Response(200, json=TOKEN_BODY))
    with httpx.Client() as http:
        tokens = oauth.exchange_code(http, "app123", "s3cr3t", "the-code", now=1000.0)
    req = route.calls.last.request
    assert req.headers["Authorization"] == "Basic " + base64.b64encode(b"app123:s3cr3t").decode()
    assert req.headers["Content-Type"].startswith("application/x-www-form-urlencoded")
    assert parse_qs(req.content.decode()) == {
        "grant_type": ["authorization_code"], "code": ["the-code"],
        "redirect_uri": [PINTEREST_REDIRECT_URI]}
    assert tokens.access_token == "ACCESS-1" and tokens.refresh_token == "REFRESH-1"
    assert tokens.expires_at == 1000.0 + 2592000
    assert tokens.refresh_expires_at == 1000.0 + 31536000
    assert "ACCESS-1" not in repr(tokens) and "REFRESH-1" not in repr(tokens)


@respx.mock
def test_exchange_failure_is_an_auth_error_without_secrets():
    respx.post(oauth.TOKEN_URL).mock(
        return_value=httpx.Response(400, json={"code": 283, "message": "Invalid code"}))
    with httpx.Client() as http, pytest.raises(AuthError) as err:
        oauth.exchange_code(http, "app123", "s3cr3t", "the-code")
    assert "400" in str(err.value) and "Invalid code" in str(err.value)
    assert "s3cr3t" not in str(err.value) and "the-code" not in str(err.value)


@respx.mock
def test_refresh_keeps_old_refresh_token_when_not_rotated():
    route = respx.post(oauth.TOKEN_URL).mock(return_value=httpx.Response(
        200, json={"access_token": "ACCESS-2", "expires_in": 100}))
    old = oauth.TokenSet("ACCESS-1", "REFRESH-1", expires_at=5.0, refresh_expires_at=9999.0)
    with httpx.Client() as http:
        new = oauth.refresh_tokens(http, "app123", "s3cr3t", old, now=50.0)
    assert parse_qs(route.calls.last.request.content.decode()) == {
        "grant_type": ["refresh_token"], "refresh_token": ["REFRESH-1"]}
    assert route.calls.last.request.headers["Authorization"].startswith("Basic ")
    assert (new.access_token, new.refresh_token) == ("ACCESS-2", "REFRESH-1")
    assert new.expires_at == 150.0 and new.refresh_expires_at == 9999.0


def test_token_file_is_mode_600_and_round_trips(home):
    store = Store(home)
    path = oauth.save_tokens(store, oauth.TokenSet("A", "R", 10.0, 20.0, "pins:read"))
    assert path == home / "auth" / "pinterest.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert oauth.load_tokens(store) == oauth.TokenSet("A", "R", 10.0, 20.0, "pins:read")
    path.chmod(0o644)  # a rewrite tightens it again
    oauth.save_tokens(store, oauth.TokenSet("B"))
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text())["access_token"] == "B"


def test_load_tokens_missing_and_corrupt(home):
    store = Store(home).ensure()
    assert oauth.load_tokens(store) is None
    oauth.token_path(store).write_text("{nope")
    with pytest.raises(AuthError):
        oauth.load_tokens(store)


@respx.mock
def test_token_manager_refreshes_expired_token_and_saves(home):
    store = Store(home)
    oauth.save_tokens(store, oauth.TokenSet("OLD", "REFRESH-1", expires_at=1000.0))
    route = respx.post(oauth.TOKEN_URL).mock(return_value=httpx.Response(
        200, json={"access_token": "NEW", "refresh_token": "REFRESH-2", "expires_in": 500}))
    with httpx.Client() as http:
        tm = oauth.TokenManager(make_settings(home), store, http, now=lambda: 2000.0)
        assert tm.access_token() == "NEW"
        assert tm.access_token() == "NEW"  # still valid: no second refresh
    assert route.call_count == 1
    saved = oauth.load_tokens(store)
    assert (saved.access_token, saved.refresh_token, saved.expires_at) == \
        ("NEW", "REFRESH-2", 2500.0)
    assert stat.S_IMODE(oauth.token_path(store).stat().st_mode) == 0o600


@respx.mock
def test_token_manager_valid_token_needs_no_request(home):
    store = Store(home)
    oauth.save_tokens(store, oauth.TokenSet("OK", "R", expires_at=5000.0))
    with httpx.Client() as http:
        assert oauth.TokenManager(make_settings(home), store, http,
                                  now=lambda: 100.0).access_token() == "OK"


def test_token_manager_env_token_short_circuits(home):
    with httpx.Client() as http:
        tm = oauth.TokenManager(make_settings(home, pinterest_access_token="ENV"),
                                Store(home), http)
        assert tm.access_token() == "ENV"
        assert tm.refresh() is False
    assert not (home / "auth" / "pinterest.json").exists()


def test_token_manager_not_logged_in(home):
    with httpx.Client() as http, pytest.raises(AuthError, match="taste auth"):
        oauth.TokenManager(make_settings(home), Store(home), http).access_token()


def test_token_manager_expired_without_refresh_token(home):
    store = Store(home)
    oauth.save_tokens(store, oauth.TokenSet("OLD", None, expires_at=1.0))
    with httpx.Client() as http, pytest.raises(AuthError, match="cannot be refreshed"):
        oauth.TokenManager(make_settings(home), store, http, now=lambda: 99.0).access_token()
