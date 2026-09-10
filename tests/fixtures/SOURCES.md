# Fixture sources

Captured during the interrupted cleanup on 2026-09-10; retained pages were trimmed further during continuation.

- `api_category_page.json`: https://www.foxnews.com/api/article-search?searchBy=categories&values=fox-news/transcript&size=3&from=0; metadata excerpt.
- `api_tag_page.json`: article-search API, show-tag query; metadata excerpt. Exact original query was not saved.
- `transcript_2024_fns.html`: Fox News Sunday 2024 transcript captured by the previous session; headline/date and four short paragraph excerpts retained. Source URL confirmed during live smoke: https://www.foxnews.com/transcript/fox-news-sunday-october-20-2024. The exact original capture time was not saved.
- `video_2026_hannity.html`: Hannity video page from the same session, headline/date excerpt; exact capture URL was not saved.

Synthetic pagination, retry, and conversion cases are constructed in tests.
