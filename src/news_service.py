from __future__ import annotations

import html
import os
import re
import time
import textwrap
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote_plus

import requests


DEFAULT_NEWS_QUERY = '"Formula 1" OR F1 when:7d'
DEFAULT_NEWS_FEED_URL = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
DEFAULT_NEWS_TIMEOUT_SECONDS = 12.0
DEFAULT_NEWS_CACHE_TTL_SECONDS = 900

GENERIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
    "you",
    "your",
    "f1",
    "formula",
    "race",
    "races",
    "grand",
    "prix",
    "season",
    "driver",
    "drivers",
    "team",
    "teams",
}

TOPIC_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    (
        "2026 regulation talks",
        ("regulation", "rules", "rule", "tweak", "energy", "hybrid", "cost cap", "power unit"),
    ),
    (
        "driver and engineer moves",
        ("join", "leave", "departure", "engineer", "switch", "contract", "future", "move"),
    ),
    (
        "team pace and form",
        ("pace", "performance", "battle", "championship", "title", "winner", "loser", "standing"),
    ),
    ("Red Bull", ("red bull",)),
    ("Ferrari", ("ferrari",)),
    ("Mercedes", ("mercedes",)),
    ("McLaren", ("mclaren",)),
]


@dataclass(frozen=True)
class NewsArticleSummary:
    title: str
    summary: str
    source: str
    url: str
    published_at: str | None


class Formula1NewsService:
    def __init__(
        self,
        query: str | None = None,
        feed_url_template: str | None = None,
        timeout_seconds: float | None = None,
        cache_ttl_seconds: int | None = None,
    ) -> None:
        self.query = query or os.getenv("F1_NEWS_QUERY", DEFAULT_NEWS_QUERY)
        self.feed_url_template = feed_url_template or os.getenv("F1_NEWS_FEED_URL", DEFAULT_NEWS_FEED_URL)
        self.timeout_seconds = float(
            timeout_seconds if timeout_seconds is not None else os.getenv("F1_NEWS_TIMEOUT_SECONDS", DEFAULT_NEWS_TIMEOUT_SECONDS)
        )
        self.cache_ttl_seconds = int(
            cache_ttl_seconds if cache_ttl_seconds is not None else os.getenv("F1_NEWS_CACHE_TTL_SECONDS", DEFAULT_NEWS_CACHE_TTL_SECONDS)
        )
        self._session = requests.Session()
        self._cache: dict[int, tuple[float, dict[str, Any]]] = {}

    def build_digest(self, count: int = 6) -> dict[str, Any]:
        count = max(5, min(7, int(count)))
        cached = self._cache.get(count)
        if cached is not None:
            cached_at, payload = cached
            if time.time() - cached_at < self.cache_ttl_seconds:
                return payload

        try:
            payload = self._build_digest_uncached(count)
        except Exception as exc:
            if cached is not None:
                return cached[1]
            raise RuntimeError("Unable to load the latest Formula 1 news feed right now.") from exc

        self._cache[count] = (time.time(), payload)
        return payload

    def _build_digest_uncached(self, count: int) -> dict[str, Any]:
        items = self._fetch_feed_items()
        if not items:
            raise RuntimeError("No Formula 1 news articles were found in the feed.")

        selected = items[:count]
        articles = [self._summarize_item(item) for item in selected]
        topic_counts = self._collect_topics(selected)
        top_topics = [topic for topic, _ in topic_counts.most_common(4)]

        if not top_topics:
            top_topics = ["Formula 1 headlines"]

        return {
            "query": self.query,
            "count": len(articles),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "overall_summary": self._compose_overall_summary(top_topics, articles),
            "top_topics": top_topics,
            "articles": [asdict(article) for article in articles],
        }

    def _fetch_feed_items(self) -> list[dict[str, Any]]:
        feed_url = self.feed_url_template.format(query=quote_plus(self.query))
        response = self._session.get(
            feed_url,
            timeout=self.timeout_seconds,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; F1NewsDigest/1.0)",
                "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
            },
        )
        response.raise_for_status()

        root = ET.fromstring(response.text)
        channel = root.find("channel")
        if channel is None:
            return []

        seen: set[str] = set()
        items: list[dict[str, Any]] = []
        for item in channel.findall("item"):
            title = self._clean_text(item.findtext("title"))
            if not title:
                continue

            normalized_title = self._normalize_text(title).casefold()
            if normalized_title in seen:
                continue
            seen.add(normalized_title)

            source_node = item.find("source")
            source = self._clean_text(source_node.text if source_node is not None else "") or "Google News"
            description = self._clean_text(item.findtext("description"))
            link = self._clean_text(item.findtext("link"))
            published_at = self._parse_pubdate(self._clean_text(item.findtext("pubDate")))

            items.append(
                {
                    "title": self._normalize_headline(title, source),
                    "description": description,
                    "source": source,
                    "url": link,
                    "published_at": published_at,
                }
            )

        return items

    def _summarize_item(self, item: dict[str, Any]) -> NewsArticleSummary:
        description = item.get("description") or ""
        title = item.get("title") or ""
        source = item.get("source") or "Google News"
        summary_seed = description if len(description) > len(title) + 20 else title
        summary = textwrap.shorten(summary_seed, width=150, placeholder="...")
        return NewsArticleSummary(
            title=title,
            summary=summary,
            source=source,
            url=item.get("url") or "",
            published_at=item.get("published_at"),
        )

    def _collect_topics(self, items: list[dict[str, Any]]) -> Counter[str]:
        topic_counts: Counter[str] = Counter()
        combined_titles = " ".join(item.get("title", "") for item in items).casefold()

        for label, patterns in TOPIC_PATTERNS:
            if any(pattern in combined_titles for pattern in patterns):
                topic_counts[label] += sum(1 for item in items if any(pattern in item.get("title", "").casefold() for pattern in patterns))

        return topic_counts

    def _compose_overall_summary(self, topics: list[str], articles: list[NewsArticleSummary]) -> str:
        if topics:
            if len(topics) == 1:
                topic_text = topics[0]
            elif len(topics) == 2:
                topic_text = f"{topics[0]} and {topics[1]}"
            else:
                topic_text = f"{', '.join(topics[:-1])}, and {topics[-1]}"

            return (
                f"Recent Formula 1 coverage is centered on {topic_text}, with the latest headlines "
                f"spanning {self._headline_mix(articles)}."
            )

        return f"Recent Formula 1 coverage spans {self._headline_mix(articles)}."

    def _headline_mix(self, articles: list[NewsArticleSummary]) -> str:
        names = [article.title for article in articles[:3] if article.title]
        if not names:
            return "driver moves, regulation talks, and team updates"
        if len(names) == 1:
            return names[0].rstrip(".")
        if len(names) == 2:
            return f"{names[0].rstrip('.')}, plus {names[1].rstrip('.')}"
        return f"{names[0].rstrip('.')}, {names[1].rstrip('.')}, and {names[2].rstrip('.')}"

    @staticmethod
    def _clean_text(value: str | None) -> str:
        if not value:
            return ""
        return Formula1NewsService._normalize_text(html.unescape(value))

    @staticmethod
    def _normalize_text(value: str) -> str:
        value = re.sub(r"<[^>]+>", " ", value)
        value = re.sub(r"\s+", " ", value)
        return value.strip()

    @staticmethod
    def _normalize_headline(title: str, source: str) -> str:
        headline = Formula1NewsService._normalize_text(title)
        source_suffix = f" - {source}".casefold()
        if headline.casefold().endswith(source_suffix):
            headline = headline[: -len(source_suffix)].rstrip()

        if headline.count(":") and len(headline.split(":", 1)[0].split()) <= 4:
            prefix, suffix = headline.split(":", 1)
            if prefix.isupper() or prefix.upper() == prefix:
                headline = suffix.strip()

        return headline

    @staticmethod
    def _parse_pubdate(value: str) -> str | None:
        if not value:
            return None

        try:
            parsed = parsedate_to_datetime(value)
        except Exception:
            return None

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed.astimezone(timezone.utc).isoformat()
