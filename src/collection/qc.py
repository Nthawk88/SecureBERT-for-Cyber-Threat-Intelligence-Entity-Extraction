"""Quality control for scraped clear-web articles (Phase 1 -> Phase 3 gate).

Validates the raw JSON corpus before it is used in the qualitative case study:
  * minimum content length,
  * presence of title + parseable content,
  * recency window,
  * duplicate detection,
  * per-source counts vs target.

Run:  python -m src.collection.qc
Outputs a console report and a machine-readable summary JSON, plus a manual
review checklist the analyst should sign off on.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

from src.utils.config import load_config


MANUAL_CHECKLIST = [
    "Content is the full article body (not a truncated preview/teaser).",
    "Title matches the page and is not boilerplate ('Home', 'Blog', etc.).",
    "No navigation/cookie/footer boilerplate dominates the content.",
    "Article is genuinely security-related (threat report / advisory).",
    "Published date falls within the last 6 months.",
    "Encoding is clean (no mojibake / stray HTML entities).",
]


def _load_articles(raw_dir: Path) -> List[Dict]:
    articles: List[Dict] = []
    for path in raw_dir.rglob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            data["_path"] = str(path)
            articles.append(data)
        except (json.JSONDecodeError, OSError) as exc:
            articles.append({"_path": str(path), "_error": str(exc)})
    return articles


def run_qc(config=None) -> Dict:
    cfg = config or load_config()
    raw_dir = cfg.path("paths.scraped_raw")
    min_chars = int(cfg.get("scraping.min_content_chars", 200))
    target_per_source = int(cfg.get("scraping.articles_per_source", 20))
    recency_months = int(cfg.get("scraping.recency_months", 6))
    cutoff = datetime.now(timezone.utc) - timedelta(days=recency_months * 30)

    articles = _load_articles(raw_dir)
    per_source: Counter = Counter()
    issues: List[Dict] = []
    seen_urls: Dict[str, str] = {}
    seen_hashes: Dict[int, str] = {}
    passed = 0

    for art in articles:
        path = art.get("_path", "?")
        if "_error" in art:
            issues.append({"path": path, "issue": f"unreadable JSON: {art['_error']}"})
            continue

        source = art.get("source", "unknown")
        per_source[source] += 1
        url = art.get("url", "")
        content = (art.get("content") or "").strip()
        title = (art.get("title") or "").strip()
        local_issues: List[str] = []

        if not title:
            local_issues.append("missing title")
        if len(content) < min_chars:
            local_issues.append(f"content too short ({len(content)} < {min_chars})")

        # duplicate detection
        if url and url in seen_urls:
            local_issues.append(f"duplicate URL of {seen_urls[url]}")
        elif url:
            seen_urls[url] = path
        chash = hash(content)
        if content and chash in seen_hashes:
            local_issues.append(f"duplicate content of {seen_hashes[chash]}")
        elif content:
            seen_hashes[chash] = path

        # recency (only when a date is present)
        pub = art.get("published_at")
        if pub:
            try:
                dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt < cutoff:
                    local_issues.append(f"older than {recency_months} months ({dt.date()})")
            except ValueError:
                local_issues.append(f"unparseable published_at: {pub}")

        if local_issues:
            issues.append({"path": path, "url": url, "issues": local_issues})
        else:
            passed += 1

    summary = {
        "raw_dir": str(raw_dir),
        "total_articles": sum(per_source.values()),
        "passed": passed,
        "flagged": len(issues),
        "per_source": dict(per_source),
        "target_per_source": target_per_source,
        "sources_below_target": {
            s: c for s, c in per_source.items() if c < target_per_source
        },
        "issues": issues,
        "manual_checklist": MANUAL_CHECKLIST,
    }
    return summary


def main() -> None:
    cfg = load_config()
    summary = run_qc(cfg)

    print("=" * 60)
    print("CLEAR-WEB SCRAPING QUALITY CONTROL")
    print("=" * 60)
    print(f"Raw dir        : {summary['raw_dir']}")
    print(f"Total articles : {summary['total_articles']}")
    print(f"Passed         : {summary['passed']}")
    print(f"Flagged        : {summary['flagged']}")
    print("Per source     :")
    for source, count in sorted(summary["per_source"].items()):
        flag = "  (below target)" if count < summary["target_per_source"] else ""
        print(f"    {source:18s} {count}{flag}")
    if summary["issues"]:
        print("\nFlagged items:")
        for item in summary["issues"][:50]:
            print(f"  - {item.get('url') or item['path']}: "
                  f"{item.get('issues') or item.get('issue')}")

    print("\nManual review checklist (sign off before Phase 3):")
    for i, check in enumerate(MANUAL_CHECKLIST, 1):
        print(f"  [ ] {i}. {check}")

    out_path = cfg.path("paths.scraped_processed") / "qc_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"\nSummary written to {out_path}")


if __name__ == "__main__":
    main()
