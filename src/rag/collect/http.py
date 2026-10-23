"""A small, polite HTTP client: per-host rate limit, response cache, retries.

Every external source goes through `PoliteClient`, so the rules each source
asks for (arXiv: one request every 3 s) are enforced in one place, and every
successful response is cached on disk so reruns make no network calls.
"""

import hashlib
import json
import logging
import os
import time
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from rag.collect.config import HttpConfig
from rag.common.io import write_json

logger = logging.getLogger(__name__)

# 429: rate limited. 5xx: server trouble that usually clears on a retry.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class HttpError(Exception):
    """A request failed permanently (non-retryable status, or retries exhausted)."""


class RateLimiter:
    """Keeps at least `min_interval_s[host]` seconds between requests to each host.

    `clock` and `sleep` are injectable so tests can check spacing without waiting.
    """

    def __init__(
        self,
        min_interval_s: Mapping[str, float],
        default_interval_s: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._intervals = dict(min_interval_s)
        self._default = default_interval_s
        self._clock = clock
        self._sleep = sleep
        self._last_request: dict[str, float] = {}

    def wait(self, host: str) -> None:
        interval = self._intervals.get(host, self._default)
        last = self._last_request.get(host)
        if last is not None:
            remaining = interval - (self._clock() - last)
            if remaining > 0:
                self._sleep(remaining)
        # Stamp after waiting, i.e. when the request actually goes out.
        self._last_request[host] = self._clock()


class ResponseCache:
    """Successful response bodies on disk, one JSON file per distinct request.

    The key covers method, URL, query params and JSON body, but not headers,
    so an API key never influences (or leaks into) the cache.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def key(method: str, url: str, params: Mapping[str, Any] | None, body: Any) -> str:
        canonical = json.dumps(
            {"method": method.upper(), "url": url, "params": params or {}, "body": body},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _path(self, host: str, key: str) -> Path:
        return self.root / host / f"{key}.json"

    def get(self, host: str, key: str) -> str | None:
        path = self._path(host, key)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))["body"]

    def put(self, host: str, key: str, request: dict[str, Any], body: str) -> None:
        # The request is stored next to the body so a cache file can be read
        # and understood on its own.
        write_json(self._path(host, key), {"request": request, "body": body})


class PoliteClient:
    def __init__(
        self,
        config: HttpConfig,
        cache_dir: Path,
        host_headers: Mapping[str, Mapping[str, str]] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._config = config
        self._sleep = sleep
        self._limiter = RateLimiter(
            config.min_interval_s, config.default_min_interval_s, clock=clock, sleep=sleep
        )
        self._cache = ResponseCache(cache_dir)
        # Extra headers for one host only, e.g. the Semantic Scholar API key.
        self._host_headers = {host: dict(h) for host, h in (host_headers or {}).items()}
        self._http = httpx.Client(
            headers={"User-Agent": user_agent(config)},
            timeout=config.timeout_s,
            follow_redirects=True,
            transport=transport,
        )
        # Counts network calls vs cache hits, so a rerun can show it made no calls.
        self.counts: Counter[str] = Counter()

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "PoliteClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def request_text(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_body: Any = None,
    ) -> str:
        """Return the response body, from the cache if this exact request was made before."""
        host = urlsplit(url).netloc
        key = ResponseCache.key(method, url, params, json_body)
        cached = self._cache.get(host, key)
        if cached is not None:
            self.counts["cache_hit"] += 1
            logger.debug("cache hit %s %s", method, url)
            return cached
        response = self._send(method, url, params=params, json_body=json_body)
        body = response.text
        request = {"method": method.upper(), "url": url, "params": params, "body": json_body}
        self._cache.put(host, key, request, body)
        return body

    def get_json(self, url: str, *, params: Mapping[str, Any] | None = None) -> Any:
        return json.loads(self.request_text("GET", url, params=params))

    def post_json(self, url: str, *, params: Mapping[str, Any] | None = None, body: Any) -> Any:
        return json.loads(self.request_text("POST", url, params=params, json_body=body))

    def fetch_bytes(self, url: str) -> bytes:
        """Uncached binary GET (PDFs): the downloader stores the file itself."""
        return self._send("GET", url).content

    def _send(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_body: Any = None,
    ) -> httpx.Response:
        host = urlsplit(url).netloc
        headers = self._host_headers.get(host, {})
        last_problem = ""
        for attempt in range(1, self._config.max_attempts + 1):
            self._limiter.wait(host)
            self.counts["network"] += 1
            try:
                response = self._http.request(
                    method, url, params=params, json=json_body, headers=headers
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_problem = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code < 400:
                    return response
                if response.status_code not in RETRYABLE_STATUS:
                    raise HttpError(f"{method} {url} -> HTTP {response.status_code}")
                last_problem = f"HTTP {response.status_code}"
                retry_after = _retry_after_s(response)
                if retry_after is not None and attempt < self._config.max_attempts:
                    logger.warning(
                        "%s from %s; waiting %.0fs (Retry-After)", last_problem, host, retry_after
                    )
                    self._sleep(retry_after)
                    continue
            if attempt < self._config.max_attempts:
                delay = self._config.backoff_base_s * 2 ** (attempt - 1)
                logger.warning(
                    "%s on %s (attempt %d/%d); retrying in %.0fs",
                    last_problem,
                    url,
                    attempt,
                    self._config.max_attempts,
                    delay,
                )
                self._sleep(delay)
        raise HttpError(
            f"{method} {url} failed after {self._config.max_attempts} attempts: {last_problem}"
        )


def user_agent(config: HttpConfig) -> str:
    email = os.environ.get(config.contact_email_env, "").strip()
    return f"{config.user_agent} (mailto:{email})" if email else config.user_agent


def _retry_after_s(response: httpx.Response) -> float | None:
    # Only the delta-seconds form; the HTTP-date form is rare for APIs and
    # falls back to exponential backoff.
    value = response.headers.get("Retry-After", "")
    try:
        return max(0.0, float(value))
    except ValueError:
        return None
