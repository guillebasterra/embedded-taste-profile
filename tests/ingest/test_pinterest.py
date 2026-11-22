from __future__ import annotations

import httpx
import pytest
import respx

from taste.ingest.errors import ApiError, AuthError, RateLimitStop
from taste.ingest.pinterest import PinterestClient, retry_after_seconds

from .conftest import API


class Tokens:
    def __init__(self, can_refresh=True):
        self.token, self.can_refresh, self.refreshes = "T1", can_refresh, 0

    def access_token(self):
        return self.token

    def refresh(self):
        if not self.can_refresh:
            return False
        self.refreshes += 1
        self.token = f"T{self.refreshes + 1}"
        return True


def make_client(http, tokens=None, **kw):
    sleeps: list[float] = []
    client = PinterestClient(tokens or Tokens(), http, sleep=sleeps.append, **kw)
    return client, sleeps


@respx.mock
def test_pagination_follows_bookmarks():
    def handler(request):
        bookmark = request.url.params.get("bookmark")
        assert request.url.params["page_size"] == "250"
        assert request.url.params["privacy"] == "ALL"
        assert request.headers["Authorization"] == "Bearer T1"
        pages = {None: (["a", "b"], "bm1"), "bm1": (["c"], "bm2"), "bm2": (["d"], None)}
        ids, nxt = pages[bookmark]
        return httpx.Response(200, json={"items": [{"id": i} for i in ids], "bookmark": nxt})

    route = respx.get(f"{API}/boards").mock(side_effect=handler)
    with httpx.Client() as http:
        client, _ = make_client(http)
        assert [b["id"] for b in client.boards()] == ["a", "b", "c", "d"]
    assert route.call_count == 3 and client.requests_made == 3


@respx.mock
def test_pagination_stops_on_repeated_bookmark():
    route = respx.get(f"{API}/boards/b1/pins").mock(
        return_value=httpx.Response(200, json={"items": [{"id": "x"}], "bookmark": "same"}))
    with httpx.Client() as http:
        client, _ = make_client(http)
        assert len(list(client.board_pins("b1"))) == 2
    assert route.call_count == 2


@respx.mock
def test_401_refreshes_once_then_retries():
    route = respx.get(f"{API}/boards").mock(side_effect=[
        httpx.Response(401, json={"code": 2, "message": "Authentication failed."}),
        httpx.Response(200, json={"items": [{"id": "a"}], "bookmark": None})])
    tokens = Tokens()
    with httpx.Client() as http:
        client, _ = make_client(http, tokens)
        assert [b["id"] for b in client.boards()] == ["a"]
    assert tokens.refreshes == 1
    assert route.calls[0].request.headers["Authorization"] == "Bearer T1"
    assert route.calls[1].request.headers["Authorization"] == "Bearer T2"


@respx.mock
def test_401_twice_gives_up_after_one_refresh():
    route = respx.get(f"{API}/boards").mock(return_value=httpx.Response(401, json={}))
    tokens = Tokens()
    with httpx.Client() as http, pytest.raises(AuthError, match="taste auth"):
        client, _ = make_client(http, tokens)
        list(client.boards())
    assert tokens.refreshes == 1 and route.call_count == 2


@respx.mock
def test_401_with_static_token_fails_fast():
    route = respx.get(f"{API}/boards").mock(return_value=httpx.Response(401, json={}))
    with httpx.Client() as http, pytest.raises(AuthError):
        client, _ = make_client(http, Tokens(can_refresh=False))
        list(client.boards())
    assert route.call_count == 1


@respx.mock
def test_429_honors_retry_after_then_succeeds():
    route = respx.get(f"{API}/boards").mock(side_effect=[
        httpx.Response(429, headers={"Retry-After": "7"}, json={"message": "slow down"}),
        httpx.Response(200, json={"items": [{"id": "a"}], "bookmark": None})])
    with httpx.Client() as http:
        client, sleeps = make_client(http)
        assert [b["id"] for b in client.boards()] == ["a"]
    assert sleeps == [7.0] and route.call_count == 2


@respx.mock
def test_429_without_header_backs_off_exponentially_then_stops():
    route = respx.get(f"{API}/boards").mock(return_value=httpx.Response(429, json={}))
    with httpx.Client() as http:
        client, sleeps = make_client(http, max_retries=3, backoff=2.0)
        with pytest.raises(RateLimitStop) as err:
            list(client.boards())
    assert sleeps == [2.0, 4.0, 8.0] and route.call_count == 4
    assert err.value.retry_after is None


@respx.mock
def test_429_with_long_wait_is_the_daily_cap_and_does_not_sleep():
    route = respx.get(f"{API}/boards").mock(
        return_value=httpx.Response(429, headers={"X-RateLimit-Reset": "40000"}, json={}))
    with httpx.Client() as http:
        client, sleeps = make_client(http)
        with pytest.raises(RateLimitStop) as err:
            list(client.boards())
    assert sleeps == [] and route.call_count == 1 and err.value.retry_after == 40000.0


@respx.mock
def test_max_requests_budget_stops_before_calling():
    route = respx.get(f"{API}/boards").mock(
        return_value=httpx.Response(200, json={"items": [{"id": "a"}], "bookmark": "more"}))
    with httpx.Client() as http:
        client, _ = make_client(http, max_requests=1)
        with pytest.raises(RateLimitStop):
            list(client.boards())
    assert route.call_count == 1


@respx.mock
def test_5xx_retries_and_4xx_raises_with_message():
    respx.get(f"{API}/boards").mock(side_effect=[
        httpx.Response(503), httpx.Response(200, json={"items": [], "bookmark": None})])
    respx.get(f"{API}/boards/nope/pins").mock(
        return_value=httpx.Response(404, json={"code": 40, "message": "Board not found."}))
    with httpx.Client() as http:
        client, sleeps = make_client(http)
        assert list(client.boards()) == []
        assert sleeps == [2.0]
        with pytest.raises(ApiError, match="404: Board not found"):
            list(client.board_pins("nope"))


@respx.mock
def test_transport_error_retries():
    respx.get(f"{API}/boards").mock(side_effect=[
        httpx.ConnectError("boom"), httpx.Response(200, json={"items": [], "bookmark": None})])
    with httpx.Client() as http:
        client, sleeps = make_client(http)
        assert list(client.boards()) == []
    assert len(sleeps) == 1


def test_retry_after_parsing():
    assert retry_after_seconds(httpx.Response(429, headers={"Retry-After": "12"})) == 12.0
    assert retry_after_seconds(httpx.Response(429, headers={"X-RateLimit-Reset": "30"})) == 30.0
    assert retry_after_seconds(httpx.Response(429)) is None
    assert retry_after_seconds(httpx.Response(429, headers={"Retry-After": "soon"})) is None
    past = retry_after_seconds(
        httpx.Response(429, headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}))
    assert past == 0.0
