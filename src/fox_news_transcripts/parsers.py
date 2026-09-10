"""Parse a foxnews.com transcript page into a record.

Fox News renders every article, however old, in its current template, so one
parser covers the whole 2003--2025 range: ``h1.headline``, ``time[datetime]``,
``div.article-body`` whose introductory paragraph is the dek and whose remaining
``<p>`` elements are the transcript. Video pages under ``/video/`` share the
search API's results but have no ``article-body``; they parse to an empty text
with ``kind = "video"`` so a caller can count rather than silently keep them.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from bs4 import BeautifulSoup, Tag

RUSH_NOTICE = re.compile(r"^this is a rush transcript", re.IGNORECASE)
STAFF_NOTICE = re.compile(r"^this article was written by fox news staff", re.IGNORECASE)


@dataclass(slots=True)
class Transcript:
    """One transcript page."""

    url: str
    kind: str
    title: str | None
    dek: str | None
    published_at: datetime | None
    modified_at: datetime | None
    author: str | None
    section: str | None
    uid: str
    wordcount: int
    text: str
    scraped_at: datetime

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict with ISO-formatted timestamps."""
        record = asdict(self)
        for key in ("published_at", "modified_at", "scraped_at"):
            value = record[key]
            record[key] = value.isoformat() if value else None
        return record


def normalize_space(text: str) -> str:
    """Collapse runs of whitespace, including non-breaking spaces, to one space."""
    return " ".join(text.split())


def _text(tag: Tag | None) -> str | None:
    if tag is None:
        return None
    text = normalize_space(tag.get_text(" "))
    return text or None


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _json_ld(soup: BeautifulSoup) -> dict[str, Any]:
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except json.JSONDecodeError:
            continue
        candidates = data if isinstance(data, list) else [data]
        for item in candidates:
            if isinstance(item, dict) and "datePublished" in item:
                return item
    return {}


def uid_from_url(url: str) -> str:
    """Last path segment, which Fox keeps unique per article."""
    return urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1]


def kind_from_url(url: str) -> str:
    """Classify the URL as a transcript, video, or other article."""
    path = urlsplit(url).path
    if "/video/" in path:
        return "video"
    if "/transcript" in path:
        return "transcript"
    return "article"


def parse_transcript(
    html: str, url: str, scraped_at: datetime | None = None
) -> Transcript:
    """Parse one page.

    Args:
        html: Page markup.
        url: Where it was fetched from; stored verbatim.
        scraped_at: Fetch time; defaults to now in UTC.

    Returns:
        The parsed record. Pages without an ``article-body`` yield empty text.
    """
    soup = BeautifulSoup(html, "html.parser")
    ld = _json_ld(soup)
    headline = soup.find("h1")
    time_tag = soup.find("time")
    body = soup.find("div", class_="article-body")

    dek = None
    lines: list[str] = []
    if isinstance(body, Tag):
        paragraphs = body.find_all("p")
        has_notice = any(
            RUSH_NOTICE.match(p.get_text(" ", strip=True)) for p in paragraphs
        )
        after_notice = False
        for para in paragraphs:
            line = normalize_space(para.get_text(" "))
            if RUSH_NOTICE.match(line):
                after_notice = True
                continue
            if not line or STAFF_NOTICE.match(line):
                continue
            if (
                dek is None
                and has_notice
                and not after_notice
                and para is paragraphs[0]
            ):
                dek = line
                continue
            lines.append(line)

    text = "\n".join(lines)
    published = _parse_dt(ld.get("datePublished")) or _parse_dt(
        time_tag.get("datetime") if isinstance(time_tag, Tag) else None
    )
    return Transcript(
        url=url,
        kind=kind_from_url(url) if body is not None or "/video/" in url else "article",
        title=_text(headline),
        dek=dek,
        published_at=published,
        modified_at=_parse_dt(ld.get("dateModified")),
        author=_text(soup.find("span", class_="byline-text")),
        section=_text(soup.find("span", class_="eyebrow")),
        uid=uid_from_url(url),
        wordcount=len(text.split()),
        text=text,
        scraped_at=scraped_at or datetime.now(UTC),
    )
