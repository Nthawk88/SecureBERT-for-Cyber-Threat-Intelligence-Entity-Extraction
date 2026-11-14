"""Scrapy item pipelines: quality-control filtering + JSON-per-article output."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from scrapy.exceptions import DropItem

from src.utils.config import project_root


class QualityControlPipeline:
    """Drop short / empty articles before they reach disk (QC filter)."""

    def open_spider(self, spider):
        self.min_chars = int(getattr(spider, "min_content_chars", 200))

    def process_item(self, item, spider):
        content = (item.get("content") or "").strip()
        if not content:
            raise DropItem(f"Empty content: {item.get('url')}")
        if len(content) < self.min_chars:
            raise DropItem(
                f"Too short ({len(content)} < {self.min_chars} chars): {item.get('url')}"
            )
        item["content"] = content
        item["word_count"] = len(content.split())
        return item


class JsonPerArticlePipeline:
    """Write one JSON file per article into ``data/scraped/raw/<source>/``."""

    def open_spider(self, spider):
        out_dir = getattr(spider, "output_dir", "data/scraped/raw")
        self.base = project_root() / out_dir
        self.base.mkdir(parents=True, exist_ok=True)
        self.count = 0

    def _slug(self, url: str) -> str:
        h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
        tail = re.sub(r"[^a-zA-Z0-9]+", "-", url.split("//", 1)[-1])[-50:].strip("-")
        return f"{tail}-{h}" if tail else h

    def process_item(self, item, spider):
        source = item.get("source", "unknown")
        source_dir = self.base / source
        source_dir.mkdir(parents=True, exist_ok=True)
        path = source_dir / f"{self._slug(item['url'])}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(dict(item), fh, ensure_ascii=False, indent=2)
        self.count += 1
        spider.logger.info(f"[{source}] saved {self.count}: {item['url']}")
        return item
