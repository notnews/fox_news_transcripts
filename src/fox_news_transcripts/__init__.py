"""Scraper and provenance record for the Fox News Transcripts 2003--2025 corpus."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fox-news-transcripts")
except PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "0.0.0"
