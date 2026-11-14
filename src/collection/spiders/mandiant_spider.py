"""Mandiant / Google Cloud threat-intelligence blog spider."""

from __future__ import annotations

from src.collection.spiders.base_spider import CTIBaseSpider


class MandiantSpider(CTIBaseSpider):
    name = "mandiant"
    source_name = "mandiant"
    allowed_domains = ["cloud.google.com", "www.mandiant.com"]
    start_urls = ["https://cloud.google.com/blog/topics/threat-intelligence"]

    listing_link_selectors = [
        "a.blog-postcard__link::attr(href)",
        "article a::attr(href)",
        "h2 a::attr(href)",
    ]
    article_url_hints = ["/blog/"]
    title_selectors = []  # rely on the meta-first fallback chain
    content_selectors = ["div.article-body p", "div[class*='body'] p", "article p"]
    date_selectors = [
        "meta[property='article:published_time']::attr(content)",
        "time::attr(datetime)",
    ]
