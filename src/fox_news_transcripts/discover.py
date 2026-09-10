"""List transcript URLs through Fox News' article-search API.

``https://www.foxnews.com/api/article-search`` pages through a category or a
tag ten to thirty results at a time. The transcript category alone stops at
around ten thousand results, so the 2025 collection also walked every
``fox-news/shows/<show>/transcript`` tag it saw in the category results. Tag
queries return video pages as well as transcripts; the record keeps the API's
``category`` so a caller can filter.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    import scrapelib

from fox_news_transcripts.checkpoint import prepare_checkpoint

log = logging.getLogger(__name__)

API_URL = "https://www.foxnews.com/api/article-search"
TRANSCRIPT_CATEGORY = "fox-news/transcript"


def load_seen_urls(path: Path) -> set[str]:
    """URLs already present in a JSONL file, if it exists."""
    if not path.exists():
        return set()
    prepare_checkpoint(path)
    with path.open(encoding="utf-8") as handle:
        return {json.loads(line)["url"] for line in handle if line.strip()}


def absolute_url(url: str) -> str:
    """The API has returned both absolute URLs and bare paths over the years."""
    if url.startswith(("http://", "https://")):
        return url
    return "https://www.foxnews.com/" + url.lstrip("/")


def to_record(item: dict[str, Any], query: str, fetched_at: str) -> dict[str, Any]:
    """Keep the API fields that matter and record which query found the item."""
    category = item.get("category") or {}
    return {
        "url": absolute_url(item["url"]),
        "article_id": item.get("articleId"),
        "title": item.get("title"),
        "description": item.get("description"),
        "published_at": item.get("publicationDate"),
        "last_published_at": item.get("lastPublishedDate"),
        "category": category.get("name"),
        "category_url": category.get("url"),
        "query": query,
        "fetched_at": fetched_at,
    }


def iter_query(
    session: scrapelib.Scraper,
    search_by: str,
    value: str,
    *,
    page_size: int = 30,
    max_pages: int | None = None,
):
    """Yield one category or tag query until its results are exhausted."""
    if page_size < 1 or (max_pages is not None and max_pages < 1):
        raise ValueError("page size and max pages must be positive")
    start = 0
    pages = 0
    seen_pages = set()
    while max_pages is None or pages < max_pages:
        params = {
            "searchBy": search_by,
            "values": value,
            "size": page_size,
            "from": start,
        }
        items = session.get(API_URL, params=params).json()
        if not isinstance(items, list):
            raise ValueError("expected article-search result array")
        if not items:
            return
        signature = tuple(item["url"] for item in items)
        if signature in seen_pages:
            raise ValueError("article-search repeated a page")
        seen_pages.add(signature)
        yield from items
        start += page_size
        pages += 1


def show_tags(items: list[dict[str, Any]]) -> list[str]:
    """Extract transcript show tags from category results."""
    tags: dict[str, None] = {}
    for item in items:
        url = (item.get("category") or {}).get("url") or ""
        if "/category/shows/" in url:
            tags["fox-news" + url.split("/category", 1)[1]] = None
    return list(tags)


def discover(
    session: scrapelib.Scraper,
    out: Path,
    *,
    page_size: int = 30,
    max_pages: int | None = None,
    include_shows: bool = True,
) -> dict[str, int]:
    """Append newly seen URLs to ``out`` and return counts per query."""
    seen = load_seen_urls(out)
    fetched_at = datetime.now(UTC).isoformat()
    counts: dict[str, int] = {}
    out.parent.mkdir(parents=True, exist_ok=True)

    def run(search_by: str, value: str) -> list[dict[str, Any]]:
        items = iter_query(
            session, search_by, value, page_size=page_size, max_pages=max_pages
        )
        new = 0
        observed = []
        with out.open("a", encoding="utf-8") as handle:
            for item in items:
                observed.append(item)
                record = to_record(item, f"{search_by}:{value}", fetched_at)
                if record["url"] in seen:
                    continue
                seen.add(record["url"])
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                new += 1
        counts[value] = new
        log.info("%s: %d items, %d new", value, len(observed), new)
        return observed

    category_items = run("categories", TRANSCRIPT_CATEGORY)
    if include_shows:
        for tag in show_tags(category_items):
            run("tags", tag)
    return counts
