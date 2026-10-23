import httpx
import pytest
import respx

from rag.collect.http import HttpError, RateLimiter

URL = "https://api.example.org/items"
OTHER_URL = "https://other.example.org/items"


def test_rate_limiter_spaces_requests_per_host(fake_clock) -> None:
    limiter = RateLimiter(
        {"a": 3.0}, default_interval_s=1.0, clock=fake_clock.clock, sleep=fake_clock.sleep
    )
    limiter.wait("a")  # first request to a host never waits
    fake_clock.now += 1.0
    limiter.wait("a")  # 1 s elapsed of 3 s -> waits 2 s
    limiter.wait("b")  # a different host has its own clock
    assert fake_clock.sleeps == [2.0]


def test_rate_limiter_does_not_wait_when_interval_already_passed(fake_clock) -> None:
    limiter = RateLimiter(
        {}, default_interval_s=1.0, clock=fake_clock.clock, sleep=fake_clock.sleep
    )
    limiter.wait("a")
    fake_clock.now += 5.0
    limiter.wait("a")
    assert fake_clock.sleeps == []


@respx.mock
def test_requests_to_same_host_are_spaced(make_client, fake_clock) -> None:
    respx.get(URL).mock(return_value=httpx.Response(200, text="ok"))
    with make_client() as client:
        client.request_text("GET", URL, params={"page": 1})
        client.request_text("GET", URL, params={"page": 2})
    assert fake_clock.sleeps == [3.0]


@respx.mock
def test_second_identical_request_is_served_from_cache(make_client) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(200, text="body"))
    with make_client() as first:
        assert first.request_text("GET", URL, params={"q": "bpe"}) == "body"
    # A new client (as in a rerun) reads the same on-disk cache.
    with make_client() as second:
        assert second.request_text("GET", URL, params={"q": "bpe"}) == "body"
        assert second.counts == {"cache_hit": 1}
    assert route.call_count == 1


@respx.mock
def test_different_params_are_different_cache_entries(make_client) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(200, text="body"))
    with make_client() as client:
        client.request_text("GET", URL, params={"q": "bpe"})
        client.request_text("GET", URL, params={"q": "unigram"})
    assert route.call_count == 2


@respx.mock
def test_retries_429_with_backoff_then_succeeds(make_client, fake_clock) -> None:
    route = respx.get(URL).mock(
        side_effect=[httpx.Response(429), httpx.Response(503), httpx.Response(200, text="ok")]
    )
    with make_client() as client:
        assert client.request_text("GET", URL) == "ok"
    assert route.call_count == 3
    # Backoff 2 s, then the limiter tops up to its 3 s gap since the last
    # request (1 s more); the 4 s backoff already exceeds the gap.
    assert fake_clock.sleeps == [2.0, 1.0, 4.0]


@respx.mock
def test_retry_after_header_is_honoured(make_client, fake_clock) -> None:
    respx.get(URL).mock(
        side_effect=[httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200)]
    )
    with make_client() as client:
        client.request_text("GET", URL)
    assert fake_clock.sleeps == [7.0]


@respx.mock
def test_gives_up_after_max_attempts(make_client) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(500))
    with make_client() as client, pytest.raises(HttpError, match="after 3 attempts"):
        client.request_text("GET", URL)
    assert route.call_count == 3


@respx.mock
def test_timeouts_are_retried(make_client) -> None:
    respx.get(URL).mock(side_effect=[httpx.ConnectTimeout("slow"), httpx.Response(200, text="ok")])
    with make_client() as client:
        assert client.request_text("GET", URL) == "ok"


@respx.mock
def test_non_retryable_status_fails_immediately(make_client) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(404))
    with make_client() as client, pytest.raises(HttpError, match="404"):
        client.request_text("GET", URL)
    assert route.call_count == 1


@respx.mock
def test_failed_responses_are_not_cached(make_client) -> None:
    route = respx.get(URL).mock(side_effect=[httpx.Response(404), httpx.Response(200, text="ok")])
    with make_client() as client:
        with pytest.raises(HttpError):
            client.request_text("GET", URL)
        assert client.request_text("GET", URL) == "ok"
    assert route.call_count == 2


@respx.mock
def test_user_agent_includes_contact_email_when_set(make_client, monkeypatch) -> None:
    monkeypatch.setenv("TEST_CONTACT_EMAIL", "someone@example.org")
    route = respx.get(URL).mock(return_value=httpx.Response(200))
    with make_client() as client:
        client.request_text("GET", URL)
    assert (
        route.calls.last.request.headers["User-Agent"]
        == "test-agent/0.0 (mailto:someone@example.org)"
    )


@respx.mock
def test_user_agent_without_contact_email(make_client, monkeypatch) -> None:
    monkeypatch.delenv("TEST_CONTACT_EMAIL", raising=False)
    route = respx.get(URL).mock(return_value=httpx.Response(200))
    with make_client() as client:
        client.request_text("GET", URL)
    assert route.calls.last.request.headers["User-Agent"] == "test-agent/0.0"


@respx.mock
def test_host_headers_sent_only_to_their_host(make_client) -> None:
    api = respx.get(URL).mock(return_value=httpx.Response(200))
    other = respx.get(OTHER_URL).mock(return_value=httpx.Response(200))
    with make_client(host_headers={"api.example.org": {"x-api-key": "secret"}}) as client:
        client.request_text("GET", URL)
        client.request_text("GET", OTHER_URL)
    assert api.calls.last.request.headers["x-api-key"] == "secret"
    assert "x-api-key" not in other.calls.last.request.headers


@respx.mock
def test_post_json_round_trip(make_client) -> None:
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=[{"id": 1}]))
    with make_client() as client:
        assert client.post_json(URL, body={"ids": ["a"]}) == [{"id": 1}]
        assert client.post_json(URL, body={"ids": ["a"]}) == [{"id": 1}]
    assert route.call_count == 1
    assert route.calls.last.request.content == b'{"ids":["a"]}'


@respx.mock
def test_rejected_body_is_retried_and_not_cached(make_client) -> None:
    route = respx.get(URL).mock(
        side_effect=[httpx.Response(200, text="bad"), httpx.Response(200, text="good")]
    )
    with make_client() as client:
        assert client.request_text("GET", URL, accept=lambda body: body == "good") == "good"
        assert client.request_text("GET", URL) == "good"  # the good body was cached
    assert route.call_count == 2
