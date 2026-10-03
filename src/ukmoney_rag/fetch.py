"""Polite, cache-first HTTP fetching.

Design rules:
* Identify ourselves (User-Agent), rate-limit, retry only transient failures.
* Respect robots.txt for HTML sources, and fail *closed* if it cannot be read.
* Never try to get around bot protection: a challenge page is a hard error.
* Raw responses are cached on disk under data/raw/<source.name>.<ext>, so re-runs are
  offline, deterministic, and you can drop in manually saved pages.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tenacity.wait import wait_base

from ukmoney_rag.models import Source

logger = logging.getLogger(__name__)

USER_AGENT = (
    "ukmoney-rag/0.1 (portfolio project; contact: 318276113+nuwork911@users.noreply.github.com)"
)

_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_CHALLENGE_MARKERS = ("<title>Just a moment...</title>", "/cdn-cgi/challenge-platform/")


class FetchError(Exception):
    """A source could not be fetched."""


class BlockedError(FetchError):
    """Access denied by the site (403, bot challenge, or robots.txt)."""


class _Retryable(Exception):
    """Internal: transient HTTP status worth retrying."""


@dataclass(frozen=True)
class FetchResult:
    content: str
    fetched_at: datetime
    from_cache: bool


def govuk_api_url(page_url: str) -> str:
    """Map https://www.gov.uk/<path> to its Content API equivalent."""
    path = urlparse(page_url).path.rstrip("/")
    return f"https://www.gov.uk/api/content{path}"


def looks_like_challenge(body: str) -> bool:
    head = body[:20_000]
    return any(marker in head for marker in _CHALLENGE_MARKERS)


class Fetcher:
    def __init__(
        self,
        cache_dir: Path,
        *,
        client: httpx.Client | None = None,
        offline: bool = False,
        refresh: bool = False,
        min_interval_s: float = 1.0,
        retry_wait: wait_base | None = None,
        user_agent: str = USER_AGENT,
    ) -> None:
        self._cache_dir = cache_dir
        self._owns_client = client is None
        self._client = client or httpx.Client(
            headers={"User-Agent": user_agent},
            timeout=httpx.Timeout(20.0),
            follow_redirects=True,
        )
        self._offline = offline
        self._refresh = refresh
        self._min_interval_s = min_interval_s
        self._retry_wait = retry_wait or wait_exponential(multiplier=1, min=1, max=10)
        self._user_agent = user_agent
        self._last_request = 0.0
        self._robots: dict[str, RobotFileParser] = {}
        self._blocked_origins: dict[str, str] = {}

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def cache_path(self, source: Source) -> Path:
        suffix = "json" if source.kind == "govuk_api" else "html"
        return self._cache_dir / f"{source.name}.{suffix}"

    def fetch(self, source: Source) -> FetchResult:
        path = self.cache_path(source)
        if path.exists() and (self._offline or not self._refresh):
            logger.info("cache hit: %s", source.name)
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            return FetchResult(path.read_text(encoding="utf-8"), mtime, from_cache=True)
        if self._offline:
            raise FetchError(f"{source.name}: not in cache and --offline is set")

        if source.kind == "html":
            self._check_robots(source.url)
            url = source.url
        else:
            url = govuk_api_url(source.url)

        logger.info("fetching: %s", url)
        response = self._request(url)
        if response.status_code in (401, 403):
            raise BlockedError(f"HTTP {response.status_code} for {url} (access denied)")
        if response.status_code >= 400:
            raise FetchError(f"HTTP {response.status_code} for {url}")
        if looks_like_challenge(response.text):
            raise BlockedError(f"bot-protection challenge page returned for {url}")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(response.text, encoding="utf-8")
        return FetchResult(response.text, datetime.now(UTC), from_cache=False)

    # -- internals ---------------------------------------------------------

    def _throttle(self) -> None:
        wait = self._min_interval_s - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _request(self, url: str) -> httpx.Response:
        self._throttle()
        try:
            for attempt in Retrying(
                stop=stop_after_attempt(3),
                wait=self._retry_wait,
                retry=retry_if_exception_type((httpx.TransportError, _Retryable)),
                reraise=True,
            ):
                with attempt:
                    response = self._client.get(url)
                    if response.status_code in _RETRY_STATUS:
                        raise _Retryable(f"HTTP {response.status_code}")
        except (httpx.TransportError, _Retryable) as exc:
            raise FetchError(f"giving up on {url}: {exc}") from exc
        return response

    def _check_robots(self, url: str) -> None:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin in self._blocked_origins:  # don't keep knocking on a door that is closed
            raise BlockedError(self._blocked_origins[origin])
        parser = self._robots.get(origin)
        if parser is None:
            parser = RobotFileParser()
            response = self._request(f"{origin}/robots.txt")
            if response.status_code == 200 and not looks_like_challenge(response.text):
                parser.parse(response.text.splitlines())
            elif response.status_code == 404:
                parser.parse([])  # no robots.txt means no restrictions
            else:
                message = (
                    f"cannot read robots.txt at {origin} (HTTP {response.status_code}); "
                    "refusing to fetch"
                )
                self._blocked_origins[origin] = message
                raise BlockedError(message)
            self._robots[origin] = parser
        if not parser.can_fetch(self._user_agent, url):
            raise BlockedError(f"robots.txt disallows {url}")
