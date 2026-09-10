# Fox News Transcripts 2003–2025

[![CI](https://github.com/notnews/fox_news_transcripts/actions/workflows/ci.yml/badge.svg)](https://github.com/notnews/fox_news_transcripts/actions/workflows/ci.yml)
[![Data](https://img.shields.io/badge/data-Dataverse-blue)](https://doi.org/10.7910/DVN/Q2KIES)
[![Code license](https://img.shields.io/badge/code-MIT-green)](LICENSE)

Discovery, scraping, and conversion tools for Fox News transcript pages. Search results can also contain video pages; the new transcript collector excludes those URLs.

## Data

| Release | Files | Coverage | Count | DOI |
|---|---|---|---|---|
| Earlier collection | Parsed transcripts and raw HTML | 2003 onward | Approximately 24k, historical README claim | [Q2KIES](https://doi.org/10.7910/DVN/Q2KIES) |
| 2025 collection | `foxnews-transcript-urls-2025.csv.gz`, `fnc_transcripts_text_2025` text archive, raw HTML | 2003–2025 | See Dataverse; the old year table mixes discovery results and cannot establish transcript counts | [Q2KIES](https://doi.org/10.7910/DVN/Q2KIES) |

Historical counts below describe published releases or the local files identified in the table, not a new full collection. Dataverse metadata requests returned HTTP 403 during cleanup on 2026-09-10; unverified release claims remain labeled as historical documentation.

## Column dictionary

| Parquet columns | Type | Meaning |
|---|---|---|
| `url`, `uid` | string | Source URL and final path segment |
| `kind` | string | Transcript, video, or article; historical conversion retains source kinds |
| `title`, `dek`, `author`, `section` | string | Page or discovery metadata when available |
| `published_at` | UTC timestamp | Page publication timestamp, not necessarily the episode air date |
| `text`, `wordcount` | string, int32 | Transcript and whitespace word count; null for missing historical text files |
| `source` | string | Input filename |

Scrape JSONL also carries modification/fetch times and the resolved `source_url`, including the Wayback timestamp when used.

## Coverage and known gaps

Category discovery alone may be capped by the source; show-tag queries broaden coverage but do not prove completeness. Discovery retains the API category so mixed source kinds can be inspected. Repeated result pages cause an error rather than an endless loop. The collector skips non-transcript URLs, and empty or unrecognized transcript pages remain failures eligible for retry.

Historical text files join on the URL's final path segment. Publication dates can differ from dates in episode headlines. Preserve those distinctions in downstream analysis. The 2025 full corpus is not present locally and has not been reconverted.

## How collected

| Era | Method |
|---|---|
| Earlier collection | Discover transcript links and download HTML |
| 2025 | Article-search category plus show-tag queries; HTML-to-text notebooks |
| Cleanup | Streaming discovery checkpoint, bounded transcript fetches, raw gzip HTML, pure parsing and typed Parquet |

The pre-cleanup implementation is preserved at [1dbb84c](https://github.com/notnews/fox_news_transcripts/tree/1dbb84c). New fetches write checkpoints under `data/`; reruns skip successful records and retry failures. Pure parsers read saved responses without accessing the network. Fixture provenance is in [tests/fixtures/SOURCES.md](tests/fixtures/SOURCES.md).

An interrupted, unterminated final JSONL record is removed before resuming; complete records are preserved. A valid final record missing only its newline is retained. Malformed complete lines remain errors.

## Usage

Python 3.12 or later and [uv](https://docs.astral.sh/uv/) are required.

```sh
uv sync --frozen --group dev
uv run fox-news-transcripts discover --max-pages 1 --no-shows
uv run fox-news-transcripts scrape --limit 5
uv run fox-news-transcripts to-parquet data/transcripts.jsonl --out data/transcripts.parquet
uv run fox-news-transcripts to-parquet --legacy-urls data/foxnews-transcript-urls-2025.csv.gz --legacy-text-dir data/fnc_transcripts_text_2025 --out data/legacy.parquet
uv run fox-news-transcripts upload data/transcripts.parquet
```

Run `make check` for Ruff, formatting, pytest, and pre-commit. `make ci-docker` runs lint and tests in standard Python 3.12 and 3.14 Docker images. CI uses the same lockfile and checks. Large inputs and generated data belong under ignored `data/`, not in Git.

The `upload` command reads `DATAVERSE_API_TOKEN` from the environment and adds the specified file to Dataverse. It does not publish a dataset version. Cleanup does not upload or replace any remote data.

## Citation

Use [CITATION.cff](CITATION.cff) and cite the relevant [Dataverse release](https://doi.org/10.7910/DVN/Q2KIES), including its version and DOI.

## License

Code is [MIT licensed](LICENSE). News text, abstracts, and archived pages retain their owners' rights; a code license does not grant rights to those materials. Consult the terms of the linked data release.

## 🔗 Adjacent Repositories

- [notnews/msnbc_transcripts](https://github.com/notnews/msnbc_transcripts) — MSNBC Transcripts: 2008--2022
- [notnews/cnn_transcripts](https://github.com/notnews/cnn_transcripts) — CNN Transcripts 2000--2025
- [notnews/stanford_tv_news](https://github.com/notnews/stanford_tv_news) — Stanford Cable TV News Dataset
- [notnews/nbc_transcripts](https://github.com/notnews/nbc_transcripts) — NBC-hosted MSNBC transcripts 2008--2014
- [notnews/archive_news_cc](https://github.com/notnews/archive_news_cc) — Closed Caption Transcripts of News Videos from archive.org 2014--2023
