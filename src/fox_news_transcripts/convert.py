"""Build the Parquet deliverable from scraper JSONL or the 2025 Dataverse files.

The 2025 release is a URL list (``foxnews-transcript-urls-2025.csv.gz``) plus a
tarball of ``.txt`` files named after the URL's last path segment. Those two
join on that segment.
"""

from __future__ import annotations

import ast
import csv
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA = pa.schema(
    [
        pa.field("url", pa.string(), nullable=False),
        pa.field("uid", pa.string()),
        pa.field("kind", pa.string()),
        pa.field("title", pa.string()),
        pa.field("dek", pa.string()),
        pa.field("published_at", pa.timestamp("us", tz="UTC")),
        pa.field("author", pa.string()),
        pa.field("section", pa.string()),
        pa.field("wordcount", pa.int32()),
        pa.field("text", pa.string()),
        pa.field("source", pa.string(), nullable=False),
    ]
)

csv.field_size_limit(sys.maxsize)


def _dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
        return (
            parsed.replace(tzinfo=UTC)
            if parsed.tzinfo is None
            else parsed.astimezone(UTC)
        )
    except ValueError:
        return None


def _blank_to_none(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def rows_from_jsonl(path: Path) -> list[dict[str, Any]]:
    """Scraper output → schema records."""
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            records.append(
                {
                    "url": row["url"],
                    "uid": row.get("uid"),
                    "kind": row.get("kind"),
                    "title": row.get("title"),
                    "dek": row.get("dek"),
                    "published_at": _dt(row.get("published_at")),
                    "author": row.get("author"),
                    "section": row.get("section"),
                    "wordcount": row.get("wordcount"),
                    "text": row.get("text"),
                    "source": path.name,
                }
            )
    return records


def rows_from_legacy(urls_csv: Path, text_dir: Path) -> list[dict[str, Any]]:
    """The 2025 Dataverse URL list joined to its extracted ``.txt`` files."""
    import gzip

    opener = gzip.open if urls_csv.suffix == ".gz" else open
    records = []
    with opener(urls_csv, "rt", encoding="utf-8", newline="") as handle:  # type: ignore[operator]
        for row in csv.DictReader(handle):
            url = _blank_to_none(row.get("url"))
            if not url:
                continue
            uid = url.rstrip("/").rsplit("/", 1)[-1]
            txt = text_dir / f"{uid}.txt"
            text = (
                txt.read_text(encoding="utf-8", errors="replace")
                if txt.exists()
                else None
            )
            records.append(
                {
                    "url": url,
                    "uid": uid,
                    "kind": "video" if "/video/" in url else "transcript",
                    "title": _blank_to_none(row.get("title")),
                    "dek": _blank_to_none(row.get("description")),
                    "published_at": _dt(row.get("publicationDate")),
                    "author": None,
                    "section": _blank_to_none(
                        (ast.literal_eval(row["category"]).get("name"))
                        if row.get("category", "").startswith("{")
                        else None
                    ),
                    "wordcount": len(text.split()) if text is not None else None,
                    "text": text,
                    "source": urls_csv.name,
                }
            )
    return records


def dedupe(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """First occurrence of a URL wins."""
    seen: set[str] = set()
    kept = []
    dropped = 0
    for record in records:
        if record["url"] in seen:
            dropped += 1
            continue
        seen.add(record["url"])
        kept.append(record)
    return kept, dropped


def write_parquet(records: list[dict[str, Any]], out: Path) -> pa.Table:
    """Write under :data:`SCHEMA`, sorted by publication time."""
    records.sort(
        key=lambda r: (
            r["published_at"] is None,
            r["published_at"] or datetime.min.replace(tzinfo=UTC),
        )
    )
    table = pa.Table.from_pylist(records, schema=SCHEMA)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, compression="zstd", row_group_size=50_000)
    return table


def describe(table: pa.Table, dropped: int) -> str:
    """Rows per source and kind, duplicates dropped, rows per year."""
    lines = [f"rows: {table.num_rows}  duplicates dropped: {dropped}"]
    for column in ("source", "kind"):
        lines.append(f"per {column}:")
        lines.extend(
            f"  {k}: {v}"
            for k, v in sorted(Counter(table.column(column).to_pylist()).items())
        )
    years = Counter(
        d.year if d else None for d in table.column("published_at").to_pylist()
    )
    lines.append("per year:")
    lines.extend(
        f"  {y or 'missing'}: {n}"
        for y, n in sorted(years.items(), key=lambda kv: (kv[0] is None, kv[0]))
    )
    return "\n".join(lines)
