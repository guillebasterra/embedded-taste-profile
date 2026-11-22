"""Pinterest OAuth: authorization URL, localhost callback, token exchange/refresh, token file.

Nothing in here prints or logs a secret, a code or a token.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx

from taste.config import PINTEREST_REDIRECT_URI, PINTEREST_SCOPES, Settings
from taste.ingest.errors import AuthError
from taste.store import Store

AUTH_URL = "https://www.pinterest.com/oauth/"
TOKEN_URL = "https://api.pinterest.com/v5/oauth/token"
TOKEN_FILE = "pinterest.json"
EXPIRY_SKEW = 60  # refresh this many seconds before the access token expires


@dataclass
class TokenSet:
    access_token: str
    refresh_token: str | None = None
    expires_at: float | None = None  # unix seconds
    refresh_expires_at: float | None = None
    scope: str | None = None

    def __repr__(self) -> str:  # never leak tokens through a traceback or a debug print
        return f"TokenSet(expires_at={self.expires_at}, has_refresh={bool(self.refresh_token)})"

    def expired(self, now: float | None = None) -> bool:
        if self.expires_at is None:
            return False
        return (now if now is not None else time.time()) >= self.expires_at - EXPIRY_SKEW


# ---- authorization URL + callback -------------------------------------------------

def new_state() -> str:
    return secrets.token_urlsafe(24)


def build_auth_url(app_id: str, state: str, redirect_uri: str = PINTEREST_REDIRECT_URI,
                   scopes: tuple[str, ...] = PINTEREST_SCOPES) -> str:
    query = urlencode({
        "client_id": app_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": ",".join(scopes),
        "state": state,
    })
    return f"{AUTH_URL}?{query}"


def parse_callback(url_or_path: str, expected_state: str) -> str:
    """Return the authorization code from a redirect, after checking `state`."""
    params = parse_qs(urlsplit(url_or_path).query)
    if "error" in params:
        detail = (params.get("error_description") or params["error"])[0]
        raise AuthError(f"Pinterest denied the authorization: {detail}")
    state = (params.get("state") or [""])[0]
    if not state or not secrets.compare_digest(state, expected_state):
        raise AuthError("OAuth state mismatch: the redirect did not come from this login attempt.")
    code = (params.get("code") or [""])[0]
    if not code:
        raise AuthError("The redirect carried no authorization code.")
    return code


def wait_for_callback(expected_state: str, redirect_uri: str = PINTEREST_REDIRECT_URI,
                      timeout: float = 300.0, port: int | None = None,
                      on_ready: Callable[[int], None] | None = None) -> str:
    """Serve one OAuth redirect on localhost and return the authorization code."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    target = urlsplit(redirect_uri)
    callback_path = target.path or "/"
    outcome: dict[str, object] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if urlsplit(self.path).path != callback_path:
                self.send_error(404)
                return
            try:
                outcome["code"] = parse_callback(self.path, expected_state)
                status, text = 200, "Pinterest is connected. You can close this tab."
            except AuthError as exc:
                outcome["error"] = exc
                status, text = 400, "Authorization failed. Check the terminal."
            body = f"<html><body><p>{text}</p></body></html>".encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:  # the request line contains the code
            return

    bind_port = port if port is not None else (target.port or 80)
    try:
        server = HTTPServer(("127.0.0.1", bind_port), Handler)
    except OSError as exc:
        raise AuthError(f"Could not listen on port {bind_port} for the OAuth redirect "
                        f"({exc.strerror or exc}).") from None
    try:
        if on_ready:
            on_ready(server.server_address[1])
        deadline = time.monotonic() + timeout
        while not outcome:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AuthError("Timed out waiting for the Pinterest redirect.")
            server.timeout = min(remaining, 1.0)
            server.handle_request()
    finally:
        server.server_close()
    if "error" in outcome:
        raise outcome["error"]  # type: ignore[misc]
    return str(outcome["code"])


# ---- token endpoint ---------------------------------------------------------------

def _token_request(http: httpx.Client, app_id: str, app_secret: str, data: dict[str, str],
                   now: float | None = None) -> TokenSet:
    try:
        resp = http.post(TOKEN_URL, data=data, auth=(app_id, app_secret))
    except httpx.HTTPError as exc:
        raise AuthError(f"Could not reach the Pinterest token endpoint "
                        f"({type(exc).__name__}).") from None
    if resp.status_code != 200:
        message = ""
        try:
            message = str(resp.json().get("message") or "")[:200]
        except Exception:
            pass
        raise AuthError(f"Pinterest token endpoint returned {resp.status_code}"
                        + (f": {message}" if message else ""))
    body = resp.json()
    if not body.get("access_token"):
        raise AuthError("Pinterest token endpoint returned no access token.")
    t = now if now is not None else time.time()

    def at(key: str) -> float | None:
        return t + float(body[key]) if body.get(key) is not None else None

    return TokenSet(
        access_token=body["access_token"],
        refresh_token=body.get("refresh_token"),
        expires_at=at("expires_in"),
        refresh_expires_at=at("refresh_token_expires_in"),
        scope=body.get("scope"),
    )


def exchange_code(http: httpx.Client, app_id: str, app_secret: str, code: str,
                  redirect_uri: str = PINTEREST_REDIRECT_URI, now: float | None = None) -> TokenSet:
    return _token_request(http, app_id, app_secret, {
        "grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri}, now)


def refresh_tokens(http: httpx.Client, app_id: str, app_secret: str, current: TokenSet,
                   now: float | None = None) -> TokenSet:
    if not current.refresh_token:
        raise AuthError("No refresh token on file. Run `taste auth` again.")
    new = _token_request(http, app_id, app_secret, {
        "grant_type": "refresh_token", "refresh_token": current.refresh_token}, now)
    if not new.refresh_token:  # the refresh grant does not always rotate the refresh token
        new.refresh_token = current.refresh_token
        new.refresh_expires_at = current.refresh_expires_at
    return new


# ---- token file -------------------------------------------------------------------

def token_path(store: Store) -> Path:
    return store.auth_dir / TOKEN_FILE


def save_tokens(store: Store, tokens: TokenSet) -> Path:
    path = token_path(store)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    tmp = path.with_suffix(".json.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(asdict(tokens), f, indent=2)
    os.chmod(tmp, 0o600)  # in case a stale tmp file existed with wider permissions
    os.replace(tmp, path)
    return path


def load_tokens(store: Store) -> TokenSet | None:
    path = token_path(store)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
        return TokenSet(**{k: raw.get(k) for k in TokenSet.__dataclass_fields__})
    except (ValueError, TypeError):
        raise AuthError(f"{path} is unreadable. Run `taste auth` again.") from None


class TokenManager:
    """Hands out a valid access token; refreshes and re-saves when it expires."""

    def __init__(self, settings: Settings, store: Store, http: httpx.Client,
                 now: Callable[[], float] = time.time):
        self.settings, self.store, self.http, self.now = settings, store, http, now
        self._tokens: TokenSet | None = None

    @property
    def from_env(self) -> bool:
        return bool(self.settings.pinterest_access_token)

    def access_token(self) -> str:
        if self.from_env:
            return self.settings.pinterest_access_token  # type: ignore[return-value]
        if self._tokens is None:
            self._tokens = load_tokens(self.store)
        if self._tokens is None or not self._tokens.access_token:
            raise AuthError("Not logged in to Pinterest. Run `taste auth` "
                            "(or set PINTEREST_ACCESS_TOKEN).")
        if self._tokens.expired(self.now()) and not self.refresh():
            raise AuthError("The Pinterest token expired and cannot be refreshed. "
                            "Run `taste auth` again.")
        return self._tokens.access_token

    def refresh(self) -> bool:
        """Try a refresh grant. False when there is nothing to refresh with."""
        if self.from_env:
            return False
        if self._tokens is None:
            self._tokens = load_tokens(self.store)
        s = self.settings
        if (self._tokens is None or not self._tokens.refresh_token
                or not s.pinterest_app_id or not s.pinterest_app_secret):
            return False
        self._tokens = refresh_tokens(self.http, s.pinterest_app_id, s.pinterest_app_secret,
                                      self._tokens, self.now())
        save_tokens(self.store, self._tokens)
        return True
