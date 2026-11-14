"""Scrapy item for a scraped clear-web article."""

from __future__ import annotations

import scrapy


class ArticleItem(scrapy.Item):
    url = scrapy.Field()
    title = scrapy.Field()
    content = scrapy.Field()
    source = scrapy.Field()
    published_at = scrapy.Field()   # article publish date (ISO), if found
    scraped_at = scrapy.Field()     # crawl timestamp (ISO)
    word_count = scrapy.Field()
