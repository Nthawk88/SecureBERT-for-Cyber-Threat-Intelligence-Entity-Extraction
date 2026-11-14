"""Configurable base spider with robust fallback extraction.

CSS selectors on security blogs change frequently, so each source defines its
*preferred* selectors but the base spider always falls back to generic
heuristics (``<article>``, Open Graph meta tags, ``<time datetime>``, paragraph
aggregation). This keeps spiders resilient when site markup shifts.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional

import scrapy
from dateutil import parser as date_parser

from src.collection.items import ArticleItem

_PAGE_RE = re.compile(r"/page/\d+/?$")


class CTIBaseSpider(scrapy.Spider):
    # -- to be overridden per source -----------------------------------------
    source_name: str = "base"
    # CSS selectors that yield article hyperlinks on listing pages.
    listing_link_selectors: List[str] = []
    # Substrings that an article URL path is expected to contain (best-effort).
    article_url_hints: List[str] = []
    # Preferred content/title/date selectors (with generic fallbacks below).
    title_selectors: List[str] = []
    content_selectors: List[str] = []
    date_selectors: List[str] = []
    # Optional "next page" selector for pagination through the listing.
    next_page_selector: Optional[str] = None
    # If > 0, also seed "{start}/page/2..N+1" listing pages on startup (helps
    # server-rendered Ghost/WordPress blogs reach the per-source article target).
    paginate_pages: int = 0
    # URL substrings that indicate a NON-article (nav/section/listing) page.
    exclude_url_patterns: List[str] = [
        "/category/", "/categories/", "/tag/", "/tags/", "/topic/", "/topics/",
        "/page/", "/author/", "/authors/", "/search", "/feed", "/rss",
        "/about", "/contact", "/privacy", "/subscribe", "/newsletter",
        "/login", "/signin", "/sign-in", "#", "/products", "/services",
        "/webinar", "/category/", "/all-tags", "/events", "/podcast",
    ]

    # generic fallbacks (applied when source selectors miss) -----------------
    _FALLBACK_TITLE = [
        "meta[property='og:title']::attr(content)",
        "meta[name='twitter:title']::attr(content)",
        "h1.post-full-title::text", "h1.article-title::text", "h1.entry-title::text",
        "article h1::text",
        "title::text",
        "h1::text",  # bare h1 last: often site nav, not the article title
    ]
    _FALLBACK_CONTENT = ["article p", "main p", "div[class*='content'] p", "div[class*='article'] p", "p"]
    _FALLBACK_DATE = [
        "meta[property='article:published_time']::attr(content)",
        "time::attr(datetime)",
        "meta[name='date']::attr(content)",
        "time::text",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved = 0
        self._visited_listings: set = set()

    def start_requests(self):
        urls = list(self.start_urls)
        if self.paginate_pages:
            for base in self.start_urls:
                b = base if base.endswith("/") else base + "/"
                for n in range(2, self.paginate_pages + 2):
                    urls.append(f"{b}page/{n}/")
        for url in urls:
            yield scrapy.Request(url, callback=self.parse, dont_filter=False)

    # -- settings-derived limits ---------------------------------------------
    @classmethod
    def update_settings(cls, settings):
        super().update_settings(settings)

    def _limit(self) -> int:
        return int(self.settings.get("CTI_ARTICLES_PER_SOURCE", 20))

    @property
    def min_content_chars(self) -> int:
        return int(self.settings.get("CTI_MIN_CONTENT_CHARS", 200))

    @property
    def output_dir(self) -> str:
        return self.settings.get("CTI_OUTPUT_DIR", "data/scraped/raw")

    def _recency_cutoff(self) -> datetime:
        months = int(self.settings.get("CTI_RECENCY_MONTHS", 6))
        return datetime.now(timezone.utc) - timedelta(days=months * 30)

    # -- listing page ---------------------------------------------------------
    def parse(self, response):
        links: List[str] = []
        for sel in self.listing_link_selectors:
            links.extend(response.css(sel).getall())
        if not links:
            # generic fallback: any anchor whose path looks like an article
            links = response.css("a::attr(href)").getall()

        start = set(self.start_urls)
        roots = {u.rstrip("/") for u in self.start_urls}
        seen = set()
        for href in links:
            if not href:
                continue
            url = response.urljoin(href).split("#")[0]
            if url in seen:
                continue
            seen.add(url)
            # skip listing/start/root pages and known non-article sections
            if url in start or url.rstrip("/") in roots:
                continue
            if any(p in url for p in self.exclude_url_patterns):
                continue
            if self.article_url_hints and not any(h in url for h in self.article_url_hints):
                continue
            if self._saved >= self._limit():
                return
            yield scrapy.Request(url, callback=self.parse_article)

        # pagination: follow next-page / "/page/N/" listing links generically
        if self._saved < self._limit():
            self._visited_listings.add(response.url)
            yield from self._follow_pagination(response)

    def _follow_pagination(self, response):
        """Yield requests for the next listing page(s) using several heuristics."""
        candidates: List[str] = []
        if self.next_page_selector:
            candidates += response.css(self.next_page_selector).getall()
        candidates += response.css("a[rel='next']::attr(href)").getall()
        candidates += response.css("link[rel='next']::attr(href)").getall()
        candidates += response.css("a.older-posts::attr(href)").getall()
        candidates += response.css("a.next::attr(href)").getall()
        # any anchor that looks like a numbered pagination link
        for href in response.css("a::attr(href)").getall():
            if href and _PAGE_RE.search(href.split("#")[0]):
                candidates.append(href)

        seen = set()
        for href in candidates:
            if not href:
                continue
            url = response.urljoin(href).split("#")[0]
            if url in seen or url in self._visited_listings:
                continue
            seen.add(url)
            self._visited_listings.add(url)
            yield scrapy.Request(url, callback=self.parse)

    # -- article page ---------------------------------------------------------
    def _first(self, response, selectors: Iterable[str]) -> Optional[str]:
        for sel in selectors:
            val = response.css(sel).get()
            if val and val.strip():
                return val.strip()
        return None

    @staticmethod
    def _node_text(node) -> str:
        """All descendant text of a node, whitespace-collapsed (tags stripped)."""
        parts = [t.strip() for t in node.css("::text").getall() if t and t.strip()]
        return " ".join(parts)

    def _join(self, response, selectors: Iterable[str]) -> str:
        for sel in selectors:
            parts = [self._node_text(n) for n in response.css(sel)]
            parts = [p for p in parts if p]
            # require a couple of paragraphs to consider it real body content
            if len("\n".join(parts)) > 200 or len(parts) >= 3:
                return "\n".join(parts)
        # last resort: any paragraph text
        parts = [self._node_text(n) for n in response.css("p")]
        parts = [p for p in parts if p]
        return "\n".join(parts)

    def _parse_date(self, raw: Optional[str]) -> Optional[datetime]:
        if not raw:
            return None
        try:
            dt = date_parser.parse(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except (ValueError, OverflowError):
            return None

    def parse_article(self, response):
        if self._saved >= self._limit():
            return

        title = self._first(response, self.title_selectors + self._FALLBACK_TITLE)
        content_parts = self.content_selectors + self._FALLBACK_CONTENT
        content = self._join(response, content_parts)
        raw_date = self._first(response, self.date_selectors + self._FALLBACK_DATE)
        published = self._parse_date(raw_date)

        # recency filter (skip if we *know* it's older than the window)
        if published is not None and published < self._recency_cutoff():
            self.logger.debug(f"Skipping old article ({published.date()}): {response.url}")
            return

        item = ArticleItem(
            url=response.url,
            title=title or "",
            content=content or "",
            source=self.source_name,
            published_at=published.isoformat() if published else None,
            scraped_at=datetime.now(timezone.utc).isoformat(),
            word_count=len((content or "").split()),
        )
        self._saved += 1
        yield item
