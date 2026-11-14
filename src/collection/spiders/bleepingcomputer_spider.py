"""BleepingComputer security news spider."""

from __future__ import annotations

from src.collection.spiders.base_spider import CTIBaseSpider


class BleepingComputerSpider(CTIBaseSpider):
    name = "bleepingcomputer"
    source_name = "bleepingcomputer"
    allowed_domains = ["www.bleepingcomputer.com", "bleepingcomputer.com"]
    start_urls = ["https://www.bleepingcomputer.com/news/security/"]

    listing_link_selectors = [
        "div.bc_latest_news_text h4 a::attr(href)",
        "h4 a::attr(href)",
        "article a::attr(href)",
    ]
    article_url_hints = ["/news/"]
    title_selectors = []  # rely on the meta-first fallback chain
    content_selectors = ["div.articleBody p", "div.article_section p", "article p"]
    date_selectors = [
        "meta[property='article:published_time']::attr(content)",
        "time::attr(datetime)",
        "li.cz-news-date::text",
    ]
    next_page_selector = "a.pagination-next::attr(href)"
