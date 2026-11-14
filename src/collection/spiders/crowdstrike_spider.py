"""CrowdStrike blog spider."""

from __future__ import annotations

from src.collection.spiders.base_spider import CTIBaseSpider


class CrowdStrikeSpider(CTIBaseSpider):
    name = "crowdstrike"
    source_name = "crowdstrike"
    allowed_domains = ["www.crowdstrike.com", "crowdstrike.com"]
    start_urls = ["https://www.crowdstrike.com/blog/"]

    listing_link_selectors = [
        "a.bg-card__link::attr(href)",
        "h3 a::attr(href)",
        "article a::attr(href)",
    ]
    article_url_hints = ["/blog/"]
    title_selectors = ["h1.bg-hero__title::text"]
    content_selectors = ["div.bg-body p", "div[class*='content'] p", "article p"]
    date_selectors = [
        "meta[property='article:published_time']::attr(content)",
        "time::attr(datetime)",
    ]
