"""
Optional structured-facts lookup: if the topic matches a known entry in a
local CSV, the pipeline treats those fields as ground truth and tells the
writer node to never deviate from them (see production/nodes/strategist.py
and writer.py for how the "known facts vs. needs research" split drives the
prompt). If nothing matches, the pipeline just relies on web search — the
lookup is a shortcut for facts you already have, not a requirement.

Swap data/topic_facts.example.csv for your own domain: product specs,
pricing tiers, whatever structured ground truth you want the writer to
treat as non-negotiable.
"""
import csv
from typing import Optional

from pydantic import BaseModel

from src.config import TOPIC_FACTS_CSV
from src.utils.logger import setup_logger

logger = setup_logger("topic_facts")


class TopicSpecs(BaseModel):
    name: str
    slug: Optional[str] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    rating: Optional[str] = None
    price_min: Optional[str] = None
    price_max: Optional[str] = None
    tags: Optional[str] = None


class TopicFactsLookup:
    def __init__(self, csv_path=None):
        self.csv_path = csv_path or TOPIC_FACTS_CSV

    def _load_rows(self) -> list[dict]:
        if not self.csv_path.exists():
            return []
        with open(self.csv_path, encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def find_in_topic(self, topic: str) -> Optional[TopicSpecs]:
        """Longest case-insensitive name match inside the topic string."""
        topic_lower = topic.lower()
        best_match, max_len = None, 0

        for row in self._load_rows():
            name = row.get("name", "")
            if name and name.lower() in topic_lower and len(name) > max_len:
                best_match, max_len = row, len(name)

        if best_match:
            logger.debug(f"Facts match: '{best_match['name']}' for topic '{topic}'")
            return TopicSpecs(**best_match)
        logger.debug(f"No facts match for topic '{topic}'")
        return None

    def find_by_slug(self, slug: str) -> Optional[TopicSpecs]:
        for row in self._load_rows():
            if row.get("slug") == slug:
                return TopicSpecs(**row)
        return None
