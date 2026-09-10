import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from fox_news_transcripts import convert, discover, parsers, scrape

FIXTURES = Path(__file__).parent / "fixtures"
URL = "https://www.foxnews.com/transcript/example"
PAGE = (FIXTURES / "transcript_2024_fns.html").read_text()


def test_parser_and_video():
    row = parsers.parse_transcript(PAGE, URL)
    assert "SHANNON BREAM" in row.text
    assert row.title
    assert row.kind == "transcript"
    video = parsers.parse_transcript(
        (FIXTURES / "video_2026_hannity.html").read_text(),
        "https://www.foxnews.com/video/1",
    )
    assert video.kind == "video"
    assert video.wordcount == 0


def test_pagination_repetition_is_failure():
    session = SimpleNamespace(
        get=lambda *a, **kw: SimpleNamespace(json=lambda: [{"url": URL}])
    )
    with pytest.raises(ValueError, match="repeated"):
        list(discover.iter_query(session, "categories", "transcript"))


def test_scrape_limit_applies_to_new_records_and_resumes(tmp_path):
    urls = tmp_path / "urls.jsonl"
    urls.write_text(
        json.dumps({"url": URL}) + "\n" + json.dumps({"url": URL + "-2"}) + "\n"
    )
    out = tmp_path / "out.jsonl"
    html = tmp_path / "html"
    session = SimpleNamespace(get=lambda url: SimpleNamespace(text=PAGE, url=url))
    assert scrape.scrape(urls, out, html, limit=1, session=session).written == 1
    assert scrape.scrape(urls, out, html, limit=1, session=session).written == 1
    assert len(out.read_text().splitlines()) == 2
    assert not list(html.glob("*.part"))
    rows, dropped = convert.dedupe(convert.rows_from_jsonl(out))
    assert dropped == 0
    convert.write_parquet(rows, tmp_path / "out.parquet")
    assert pq.read_table(tmp_path / "out.parquet").schema == convert.SCHEMA


def test_legacy_missing_text_and_quoted_category(tmp_path):
    import csv

    source = tmp_path / "urls.csv"
    with source.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["url", "category"])
        writer.writeheader()
        writer.writerow({"url": URL, "category": repr({"name": "America's News"})})
    row = convert.rows_from_legacy(source, tmp_path)[0]
    assert row["section"] == "America's News"
    assert row["text"] is None
    assert row["wordcount"] is None


def test_first_speaker_is_preserved_without_intro_notice():
    page = (
        '<div class="article-body"><p>HOST: First sentence.</p>'
        "<p>GUEST: Reply.</p></div>"
    )
    row = parsers.parse_transcript(page, URL)
    assert row.text.startswith("HOST: First sentence.")
    assert row.dek is None
