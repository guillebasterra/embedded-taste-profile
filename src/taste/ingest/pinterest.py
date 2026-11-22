"""Minimal Pinterest API v5 client: bearer auth, bookmark pagination, 401/429/5xx handling."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Iterator, Protocol

import httpx

from taste.ingest.errors import ApiError, AuthError, RateLimitStop

API_BASE = "https://api.pinterest.com/v5"
MAX_PAGE_SIZE = 250


class TokenProvider(Protocol):
    def access_token(self) -> str: ...
    def refresh(self) -> bool: ...


def retry_after_seconds(resp: httpx.Response) -> float | None:
    """Seconds the server asks us to wait: Retry-After (seconds or HTTP date), else
    X-RateLimit-Reset (seconds until the window resets)."""
    raw = resp.headers.get("Retry-After")
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            try:
                when = parsedate_to_datetime(raw)
                if when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
                return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError):
                pass
    raw = resp.headers.get("X-RateLimit-Reset")
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    return None


class PinterestClient:
    def __init__(self, tokens: TokenProvider, http: httpx.Client, *,
                 sleep: Callable[[float], None] = time.sleep, max_retries: int = 4,
                 max_wait: float = 120.0, backoff: float = 2.0,
                 max_requests: int | None = None):
        self.tokens, self.http, self.sleep = tokens, http, sleep
        self.max_retries, self.max_wait, self.backoff = max_retries, max_wait, backoff
        self.max_requests = max_requests
        self.requests_made = 0

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{API_BASE}{path}"
        refreshed = False
        attempt = 0
        while True:
            if self.max_requests is not None and self.requests_made >= self.max_requests:
                raise RateLimitStop(None)
            token = self.tokens.access_token()
            self.requests_made += 1
            try:
                resp = self.http.get(url, params=params,
                                     headers={"Authorization": f"Bearer {token}"})
            except httpx.TransportError as exc:
                if attempt >= self.max_retries:
                    raise ApiError(0, f"network error ({type(exc).__name__})") from None
                self.sleep(min(self.backoff * 2 ** attempt, self.max_wait))
                attempt += 1
                continue
            status = resp.status_code
            if status == 200:
                return resp.json()
            if status == 401:
                if not refreshed and self.tokens.refresh():
                    refreshed = True
                    continue
                raise AuthError("Pinterest rejected the access token. Run `taste auth` again.")
            if status == 429:
                wait = retry_after_seconds(resp)
                if attempt >= self.max_retries or (wait is not None and wait > self.max_wait):
                    raise RateLimitStop(wait)
                self.sleep(wait if wait is not None
                           else min(self.backoff * 2 ** attempt, self.max_wait))
                attempt += 1
                continue
            if status >= 500 and attempt < self.max_retries:
                self.sleep(min(self.backoff * 2 ** attempt, self.max_wait))
                attempt += 1
                continue
            message = ""
            try:
                message = str(resp.json().get("message") or "")[:200]
            except Exception:
                pass
            raise ApiError(status, message)

    def paginate(self, path: str, params: dict[str, Any] | None = None) -> Iterator[dict]:
        """Yield items across pages, following `bookmark` until it is null."""
        query: dict[str, Any] = {"page_size": MAX_PAGE_SIZE, **(params or {})}
        seen: set[str] = set()
        while True:
            page = self.get(path, query)
            yield from page.get("items") or []
            bookmark = page.get("bookmark")
            if not bookmark or bookmark in seen:  # a repeated bookmark would loop forever
                return
            seen.add(bookmark)
            query = {**query, "bookmark": bookmark}

    # ---- endpoints ------------------------------------------------------------
    def boards(self) -> Iterator[dict]:
        """All boards; falls back to public ones when the token lacks the secret-board scope."""
        try:
            first = self.get("/boards", {"page_size": MAX_PAGE_SIZE, "privacy": "ALL"})
        except ApiError as exc:
            if getattr(exc, "status", None) not in (400, 401, 403):
                raise
            yield from self.paginate("/boards")
            return
        yield from first.get("items") or []
        bookmark = first.get("bookmark")
        if bookmark:
            yield from self.paginate("/boards", {"privacy": "ALL", "bookmark": bookmark})

    def sections(self, board_id: str) -> Iterator[dict]:
        return self.paginate(f"/boards/{board_id}/sections")

    def board_pins(self, board_id: str) -> Iterator[dict]:
        return self.paginate(f"/boards/{board_id}/pins")

    def section_pins(self, board_id: str, section_id: str) -> Iterator[dict]:
        return self.paginate(f"/boards/{board_id}/sections/{section_id}/pins")
