"""Scrapy settings, driven by ``config/config.yaml`` (compliance + politeness).

Ethical crawling per the proposal: obey robots.txt, rate-limit 2-5s with
randomisation, and present a transparent academic User-Agent.
"""

from __future__ import annotations

from src.utils.config import load_config

_cfg = load_config()
_s = _cfg.scraping

BOT_NAME = "cti_crawler"
SPIDER_MODULES = ["src.collection.spiders"]
NEWSPIDER_MODULE = "src.collection.spiders"

# -- compliance & politeness --------------------------------------------------
USER_AGENT = _s.get("user_agent", "CTI Research Crawler (academic; contact@university.edu)")
ROBOTSTXT_OBEY = bool(_s.get("robotstxt_obey", True))
DOWNLOAD_DELAY = float(_s.get("download_delay", 3))
RANDOMIZE_DOWNLOAD_DELAY = bool(_s.get("randomize_download_delay", True))
CONCURRENT_REQUESTS = int(_s.get("concurrent_requests", 4))
CONCURRENT_REQUESTS_PER_DOMAIN = int(_s.get("concurrent_requests_per_domain", 1))

# -- autothrottle (adapts to server load, stays polite) -----------------------
AUTOTHROTTLE_ENABLED = bool(_s.get("autothrottle_enabled", True))
AUTOTHROTTLE_START_DELAY = float(_s.get("download_delay_min", 2))
AUTOTHROTTLE_MAX_DELAY = float(_s.get("download_delay_max", 5))
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0

# -- output pipeline ----------------------------------------------------------
ITEM_PIPELINES = {
    "src.collection.pipelines.QualityControlPipeline": 200,
    "src.collection.pipelines.JsonPerArticlePipeline": 300,
}

# -- misc ---------------------------------------------------------------------
HTTPCACHE_ENABLED = False
LOG_LEVEL = "INFO"
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
FEED_EXPORT_ENCODING = "utf-8"

# Custom keys consumed by spiders/pipelines (not native Scrapy settings).
CTI_MIN_CONTENT_CHARS = int(_s.get("min_content_chars", 200))
CTI_ARTICLES_PER_SOURCE = int(_s.get("articles_per_source", 20))
CTI_RECENCY_MONTHS = int(_s.get("recency_months", 6))
CTI_OUTPUT_DIR = _s.get("output_dir", "data/scraped/raw")
