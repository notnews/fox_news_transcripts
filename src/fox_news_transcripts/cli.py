"""Discovery, transcript collection, conversion, and upload commands."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from fox_news_transcripts import convert, discover, scrape, upload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fox-news-transcripts")
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser(
        "discover", help="list transcript URLs via the article-search API"
    )
    d.add_argument("--out", type=Path, default=Path("data/urls.jsonl"))
    d.add_argument("--page-size", type=int, default=30)
    d.add_argument("--max-pages", type=int, help="pages per query; default unlimited")
    d.add_argument("--no-shows", action="store_true", help="skip per-show tag queries")
    d.add_argument("--rpm", type=int, default=60)

    s = sub.add_parser("scrape", help="fetch and parse pages listed in a URL file")
    s.add_argument("--urls", type=Path, default=Path("data/urls.jsonl"))
    s.add_argument("--out", type=Path, default=Path("data/transcripts.jsonl"))
    s.add_argument("--html-dir", type=Path, default=Path("data/html"))
    s.add_argument("--limit", type=int)
    s.add_argument(
        "--no-wayback", action="store_true", help="do not fall back to archive.org"
    )
    s.add_argument("--rpm", type=int, default=60)
    s.add_argument("--retries", type=int, default=3)
    s.add_argument("--timeout", type=float, default=30.0)

    p = sub.add_parser(
        "to-parquet", help="combine JSONL and the 2025 release into Parquet"
    )
    p.add_argument("inputs", type=Path, nargs="*", help="scraper JSONL files")
    p.add_argument(
        "--legacy-urls", type=Path, help="foxnews-transcript-urls-2025.csv.gz"
    )
    p.add_argument(
        "--legacy-text-dir", type=Path, help="unpacked fnc_transcripts_text_2025"
    )
    p.add_argument("--out", type=Path, required=True)

    u = sub.add_parser("upload", help="add a file to the Dataverse dataset")
    u.add_argument("file", type=Path)
    u.add_argument("--doi", default=upload.DATASET_DOI)
    return parser


def _logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return an exit status."""
    args = _build_parser().parse_args(argv)
    if args.command == "discover":
        _logging(Path("data/discover.log"))
        session = scrape.make_session(args.rpm, 3, 30.0)
        counts = discover.discover(
            session,
            args.out,
            page_size=args.page_size,
            max_pages=args.max_pages,
            include_shows=not args.no_shows,
        )
        for query, n in counts.items():
            sys.stdout.write(f"{n:>7}  {query}\n")
        return 0
    if args.command == "scrape":
        _logging(Path("data/scrape.log"))
        summary = scrape.scrape(
            args.urls,
            args.out,
            args.html_dir,
            requests_per_minute=args.rpm,
            retries=args.retries,
            timeout=args.timeout,
            wayback=not args.no_wayback,
            limit=args.limit,
        )
        return 1 if summary.failed else 0
    if args.command == "to-parquet":
        records = []
        for path in args.inputs:
            records.extend(convert.rows_from_jsonl(path))
        if args.legacy_urls:
            if not args.legacy_text_dir:
                sys.stderr.write("--legacy-text-dir is required with --legacy-urls\n")
                return 2
            records.extend(
                convert.rows_from_legacy(args.legacy_urls, args.legacy_text_dir)
            )
        records, dropped = convert.dedupe(records)
        table = convert.write_parquet(records, args.out)
        sys.stdout.write(convert.describe(table, dropped) + "\n")
        return 0
    if args.command == "upload":
        sys.stdout.write(upload.upload(args.file, args.doi) + "\n")
        return 0
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
