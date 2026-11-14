"""Securelist (Kaspersky) spider.

Server-rendered alternative to the Mandiant/Google Cloud blog, which is
JavaScript-rendered and therefore not extractable with plain Scrapy (rendering
JS is out of scope for this research). Securelist is a high-quality clear-web
threat-intelligence source. Use it instead of Mandiant if you want a 5th working
source; this is a methodological choice to document in the thesis.
"""

from __future__ import annotations

from src.collection.spiders.base_spider import CTIBaseSpider


class SecurelistSpider(CTIBaseSpider):
    name = "securelist"
    source_name = "securelist"
    allowed_domains = ["securelist.com"]
    start_urls = ["https://securelist.com/"]

    listing_link_selectors = [
        "h3.c-card__title a::attr(href)",
        "article a::attr(href)",
        "h3 a::attr(href)",
    ]
    # Securelist article URLs look like https://securelist.com/<slug>/<id>/
    article_url_hints = ["securelist.com/"]
    title_selectors = ["h1.c-article__title::text"]
    content_selectors = ["div.js-reading-content p", "div.c-article__content p", "article p"]
    date_selectors = [
        "meta[property='article:published_time']::attr(content)",
        "time::attr(datetime)",
    ]
