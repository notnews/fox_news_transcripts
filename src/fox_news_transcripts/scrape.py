"""Fetch transcript pages listed in a URL file and append parsed records to JSONL.

Raw HTML is kept gzipped under ``--html-dir`` so the corpus can be re-parsed
without refetching. JSONL is the checkpoint: append-only and flushed per row,
so an interrupted run resumes by skipping URLs already present.
"""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import requests
import scrapelib

from fox_news_transcripts import __version__
from fox_news_transcripts.discover import load_seen_urls
from fox_news_transcripts.parsers import kind_from_url, parse_transcript, uid_from_url

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)

WAYBACK_AVAILABLE = "https://archive.org/wayback/available"


@dataclass(slots=True)
class ScrapeSummary:
    """Counts reported at the end of a run."""

    urls: int = 0
    written: int = 0
    skipped: int = 0
    empty_text: int = 0
    from_wayback: int = 0
    failed: list[str] = field(default_factory=list)


def make_session(
    requests_per_minute: int, retries: int, timeout: float
) -> scrapelib.Scraper:
    """Rate-limited, retrying session. Fox serves a 403 to non-browser agents."""
    session = scrapelib.Scraper(
        requests_per_minute=requests_per_minute,
        retry_attempts=retries,
        retry_wait_seconds=10,
    )
    session.timeout = timeout
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0 Safari/537.36 "
        f"fox-news-transcripts/{__version__}"
    )
    return session


def wayback_url(session: scrapelib.Scraper, url: str) -> str | None:
    """Closest archived copy of ``url``, or None."""
    data = session.get(WAYBACK_AVAILABLE, params={"url": url}).json()
    snapshot = (data.get("archived_snapshots") or {}).get("closest") or {}
    if snapshot.get("available"):
        timestamp, original = snapshot["timestamp"], url
        return f"https://web.archive.org/web/{timestamp}id_/{original}"
    return None


def fetch_html(
    session: scrapelib.Scraper, url: str, *, wayback: bool
) -> tuple[str, str]:
    """Return the page HTML and whether it came from the Wayback Machine."""
    try:
        response = session.get(url)
        return response.text, response.url
    except scrapelib.HTTPError:
        if not wayback:
            raise
        archived = wayback_url(session, url)
        if archived is None:
            raise
        log.info("live fetch failed, using %s", archived)
        return session.get(archived).text, archived


def scrape(
    urls_file: Path,
    out: Path,
    html_dir: Path,
    *,
    requests_per_minute: int = 60,
    retries: int = 3,
    timeout: float = 30.0,
    wayback: bool = True,
    limit: int | None = None,
    session: scrapelib.Scraper | None = None,
) -> ScrapeSummary:
    """Fetch every URL in ``urls_file`` not already in ``out``."""
    session = session or make_session(requests_per_minute, retries, timeout)
    seen = load_seen_urls(out)
    summary = ScrapeSummary()
    if seen:
        log.info("resuming: %d transcripts already in %s", len(seen), out)
    out.parent.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)

    with urls_file.open(encoding="utf-8") as handle:
        urls = [json.loads(line)["url"] for line in handle if line.strip()]
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    with out.open("a", encoding="utf-8") as handle:
        for url in urls:
            summary.urls += 1
            if url in seen:
                summary.skipped += 1
                continue
            if kind_from_url(url) != "transcript":
                summary.skipped += 1
                continue
            if limit is not None and summary.written + len(summary.failed) >= limit:
                break
            try:
                html, source_url = fetch_html(session, url, wayback=wayback)
                transcript = parse_transcript(html, url, datetime.now(UTC))
                if not transcript.text:
                    raise ValueError("empty or unrecognized transcript page")
            except (
                requests.RequestException,
                scrapelib.HTTPError,
                OSError,
                ValueError,
            ):
                log.exception("failed: %s", url)
                summary.failed.append(url)
                continue
            html_path = html_dir / f"{uid_from_url(url)}.html.gz"
            part = html_path.with_suffix(".gz.part")
            with gzip.open(part, "wt", encoding="utf-8") as raw:
                raw.write(html)
            part.replace(html_path)
            record = transcript.to_record()
            archived = source_url.startswith("https://web.archive.org/")
            record["from_wayback"] = archived
            record["source_url"] = source_url
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            seen.add(url)
            summary.written += 1
            summary.from_wayback += int(archived)
            summary.empty_text += int(transcript.wordcount == 0)
    log.info(
        "done: %d urls, %d written, %d skipped, %d empty text, "
        "%d via wayback, %d failed",
        summary.urls,
        summary.written,
        summary.skipped,
        summary.empty_text,
        summary.from_wayback,
        len(summary.failed),
    )
    for url in summary.failed:
        log.warning("failed: %s", url)
    return summary
