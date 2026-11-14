"""Cisco Talos Intelligence blog spider."""

from __future__ import annotations

from src.collection.spiders.base_spider import CTIBaseSpider


class TalosSpider(CTIBaseSpider):
    name = "talos"
    source_name = "talos"
    allowed_domains = ["blog.talosintelligence.com"]
    start_urls = ["https://blog.talosintelligence.com/"]
    paginate_pages = 4  # Ghost blog; seed /page/2../page/5 to reach the target

    listing_link_selectors = [
        "h2.post-title a::attr(href)",
        "article a::attr(href)",
        "a.post-card-title::attr(href)",
    ]
    article_url_hints = ["blog.talosintelligence.com/"]
    title_selectors = ["h1.post-title::text"]
    content_selectors = ["div.post-content p", "article p"]
    date_selectors = [
        "meta[property='article:published_time']::attr(content)",
        "time.published::attr(datetime)",
    ]
    next_page_selector = "a.pagination-next::attr(href)"
