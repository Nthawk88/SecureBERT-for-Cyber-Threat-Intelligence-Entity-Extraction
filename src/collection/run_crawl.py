"""Run one or all clear-web spiders.

Usage (from project root):
    python -m src.collection.run_crawl --source unit42
    python -m src.collection.run_crawl --all
    python -m src.collection.run_crawl --source talos --limit 3 --timeout 90

Spiders write one JSON file per article into ``data/scraped/raw/<source>/``.
Remember: this corpus is for the Phase 3 qualitative case study ONLY.
"""

from __future__ import annotations

import argparse
from typing import Dict, Type

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from scrapy.settings import Settings

from src.collection import settings as cti_settings
from src.collection.spiders.base_spider import CTIBaseSpider
from src.collection.spiders.bleepingcomputer_spider import BleepingComputerSpider
from src.collection.spiders.crowdstrike_spider import CrowdStrikeSpider
from src.collection.spiders.mandiant_spider import MandiantSpider
from src.collection.spiders.securelist_spider import SecurelistSpider
from src.collection.spiders.talos_spider import TalosSpider
from src.collection.spiders.unit42_spider import Unit42Spider

SPIDERS: Dict[str, Type[CTIBaseSpider]] = {
    Unit42Spider.name: Unit42Spider,
    TalosSpider.name: TalosSpider,
    MandiantSpider.name: MandiantSpider,            # JS-rendered; see RUNBOOK
    SecurelistSpider.name: SecurelistSpider,        # verified clear-web alternative
    CrowdStrikeSpider.name: CrowdStrikeSpider,
    BleepingComputerSpider.name: BleepingComputerSpider,
}


def _settings() -> Settings:
    s = Settings()
    s.setmodule(cti_settings, priority="project")
    return s


def main() -> None:
    parser = argparse.ArgumentParser(description="CTI clear-web crawler")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", choices=sorted(SPIDERS), help="run a single source")
    group.add_argument("--all", action="store_true", help="run every source")
    parser.add_argument("--limit", type=int, default=None,
                        help="override articles per source (useful for quick tests)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="hard stop after N seconds (CLOSESPIDER_TIMEOUT)")
    args = parser.parse_args()

    settings = _settings()
    if args.limit is not None:
        settings.set("CTI_ARTICLES_PER_SOURCE", args.limit, priority="cmdline")
        settings.set("CLOSESPIDER_ITEMCOUNT", args.limit * (len(SPIDERS) if args.all else 1),
                     priority="cmdline")
    if args.timeout is not None:
        settings.set("CLOSESPIDER_TIMEOUT", args.timeout, priority="cmdline")

    process = CrawlerProcess(settings)
    if args.all:
        for spider_cls in SPIDERS.values():
            process.crawl(spider_cls)
    else:
        process.crawl(SPIDERS[args.source])
    process.start()


if __name__ == "__main__":
    main()
