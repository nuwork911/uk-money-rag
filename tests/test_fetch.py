from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from tenacity import wait_none

from ukmoney_rag.fetch import BlockedError, Fetcher, FetchError, govuk_api_url
from ukmoney_rag.models import Source

Handler = Callable[[httpx.Request], httpx.Response]


def _fetcher(tmp_path: Path, handler: Handler, **kwargs: object) -> Fetcher:
    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    return Fetcher(tmp_path, client=client, min_interval_s=0, retry_wait=wait_none(), **kwargs)  # type: ignore[arg-type]


def test_govuk_api_url() -> None:
    expected = "https://www.gov.uk/api/content/state-pension"
    assert govuk_api_url("https://www.gov.uk/state-pension") == expected
    assert govuk_api_url("https://www.gov.uk/state-pension/") == expected
    assert govuk_api_url("https://www.gov.uk/state-pension?x=1") == expected


def test_fetch_hits_api_url_then_serves_from_cache(tmp_path: Path, govuk_source: Source) -> None:
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, text='{"ok": true}')

    fetcher = _fetcher(tmp_path, handler)
    first = fetcher.fetch(govuk_source)
    second = fetcher.fetch(govuk_source)

    assert urls == ["https://www.gov.uk/api/content/sample-guide"]  # second call: no network
    assert (first.from_cache, second.from_cache) == (False, True)
    assert (tmp_path / "sample-guide.json").read_text() == '{"ok": true}'


def test_retries_transient_errors_then_succeeds(tmp_path: Path, govuk_source: Source) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] < 3 else httpx.Response(200, text="{}")

    assert _fetcher(tmp_path, handler).fetch(govuk_source).content == "{}"
    assert calls["n"] == 3


def test_gives_up_after_repeated_failures_and_does_not_cache(
    tmp_path: Path, govuk_source: Source
) -> None:
    fetcher = _fetcher(tmp_path, lambda r: httpx.Response(500))
    with pytest.raises(FetchError, match="giving up"):
        fetcher.fetch(govuk_source)
    assert not (tmp_path / "sample-guide.json").exists()


def test_404_is_not_retried(tmp_path: Path, govuk_source: Source) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404)

    with pytest.raises(FetchError, match="404"):
        _fetcher(tmp_path, handler).fetch(govuk_source)
    assert calls["n"] == 1


def test_offline_cache_miss_raises(tmp_path: Path, govuk_source: Source) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("network must not be touched in offline mode")

    with pytest.raises(FetchError, match="offline"):
        _fetcher(tmp_path, handler, offline=True).fetch(govuk_source)


def test_bot_challenge_page_is_a_hard_error_and_never_cached(
    tmp_path: Path, html_source: Source
) -> None:
    challenge = "<html><head><title>Just a moment...</title></head></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /")
        return httpx.Response(200, text=challenge)

    with pytest.raises(BlockedError, match="challenge"):
        _fetcher(tmp_path, handler).fetch(html_source)
    assert not (tmp_path / "sample-page.html").exists()


def test_robots_disallow_blocks_fetch(tmp_path: Path, html_source: Source) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /en/")
        raise AssertionError("page must not be requested when robots.txt disallows it")

    with pytest.raises(BlockedError, match=r"robots\.txt disallows"):
        _fetcher(tmp_path, handler).fetch(html_source)


def test_unreadable_robots_fails_closed(tmp_path: Path, html_source: Source) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403)

    with pytest.raises(BlockedError, match=r"cannot read robots\.txt"):
        _fetcher(tmp_path, handler).fetch(html_source)


def test_blocked_origin_is_only_probed_once(tmp_path: Path) -> None:
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        return httpx.Response(403)

    fetcher = _fetcher(tmp_path, handler)
    for name in ("a", "b", "c"):
        source = Source(name=name, kind="html", url=f"https://blocked.test/{name}", licence="l")
        with pytest.raises(BlockedError):
            fetcher.fetch(source)
    assert hits["n"] == 1  # robots.txt requested once, not once per source
