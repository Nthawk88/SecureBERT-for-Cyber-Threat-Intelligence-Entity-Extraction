"""Palo Alto Unit 42 spider."""

from __future__ import annotations

from src.collection.spiders.base_spider import CTIBaseSpider


class Unit42Spider(CTIBaseSpider):
    name = "unit42"
    source_name = "unit42"
    allowed_domains = ["unit42.paloaltonetworks.com"]
    start_urls = ["https://unit42.paloaltonetworks.com/"]
    paginate_pages = 4  # seed /page/2../page/5 to reach the target

    listing_link_selectors = [
        "article a::attr(href)",
        "h2 a::attr(href)",
        "a.card-title::attr(href)",
    ]
    article_url_hints = ["unit42.paloaltonetworks.com/"]
    title_selectors = ["h1.article-title::text"]
    content_selectors = ["div.article-content p", "article p"]
    date_selectors = [
        "meta[property='article:published_time']::attr(content)",
        "time::attr(datetime)",
    ]
    next_page_selector = "a.next::attr(href)"
