import json
from typing import Any, Dict, List, Optional, TypedDict

from src.llm_client import LLMClient
from src.production.nodes.fact_validator import ArticlePassport
from src.utils.logger import setup_logger

logger = setup_logger("strategist")


class ArticleSection(TypedDict):
    id: int
    title: str
    user_intent: str
    key_points: List[str]
    rag_query: str


class ArticleOutline(TypedDict):
    main_title: str
    seo_slug: str
    sections: List[ArticleSection]


class StrategistNode:
    """Turns a validated ArticlePassport into a concrete section-by-section outline."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def _clean_json_string(self, text: str) -> str:
        text = text.strip()
        if "```" in text:
            for part in text.split("```"):
                if "{" in part and "}" in part:
                    text = part
                    if text.startswith("json"):
                        text = text[4:]
                    break
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
        return text

    async def run(
        self,
        topic: str,
        topic_specs: Optional[Dict],
        search_results: List[Dict],
        passport: Optional[ArticlePassport] = None,
    ) -> ArticleOutline:
        search_summary = "\n".join(
            f"- {res.get('title', 'No Title')}: {res.get('description', 'No Description')}" for res in search_results
        )
        specs_str = json.dumps(topic_specs, indent=2) if topic_specs else "No specific data found locally."

        strategy_info = ""
        if passport:
            decision = passport.get("decision", {})
            facts = passport.get("facts", {})
            strategy_info = f"""
--- PIVOT STRATEGY INSTRUCTIONS (MUST FOLLOW) ---
MATCH STATUS: {decision.get('match_status')}
STRATEGY: {decision.get('selected_writing_strategy')}
REASON: {decision.get('pivot_reason')}

VERIFIED FACTS (use these, ignore hallucinations):
{json.dumps(facts, indent=2)}

STRATEGY GUIDELINES:
- IF DIRECT_REVIEW: Standard extensive review.
- IF MYTH_BUSTER: Debunk the misconception with a compelling, unique title. Vary your title structure —
  use questions ("Does [Topic] Really Do X?"), benefit-driven angles, or direct statements.
  Never use the same title pattern twice.
- IF CATEGORY_OVERVIEW: Do NOT review a single product. Review the category/segment.
- IF BUYER_GUIDE: Focus on "what to look for" rather than reviewing one specific item.
- Avoid repeating the same facts across multiple sections. Distribute key specs so each section has a unique angle.
- For MYTH_BUSTER, include at least one section with a practical alternative path
  (free trial, warranty options, budget alternatives, where to buy) without naming specific retailers or inventing offers.
-------------------------------------------------
"""

        prompt = f"""
Role: You are a Senior Content Strategist.
Your goal is to create a structured article outline based on the provided topic, facts, and writing strategy.

USER QUERY/TOPIC: "{topic}"

{strategy_info}

SPECS:
{specs_str}

SEARCH SUMMARY:
{search_summary}

---
TITLE GENERATION RULES (critical):
1. The main_title MUST reflect the USER QUERY, not just the subject's name.
2. If two articles are about the same subject but different queries, they MUST have different titles.
3. For MYTH_BUSTER: address the specific misconception in the query, not a generic myth.
---

OUTPUT SCHEMA:
Return a JSON object with keys: 'main_title' (string), 'seo_slug' (string), 'sections' (list of objects).
Each section object must have: 'id' (integer), 'title', 'user_intent', 'key_points', 'rag_query', 'description'.

ADDITIONAL REQUIREMENTS:
- Each section must have distinct key_points. Do not repeat the same metric (rating, price, feature) in more than one section.
- Ensure the outline creates a clear progression: hook -> facts -> features -> alternatives -> conclusion.

EXAMPLE JSON:
{{
  "main_title": "Title Here",
  "seo_slug": "url-slug",
  "sections": [
    {{
      "id": 1,
      "title": "Introduction",
      "user_intent": "Hook the reader",
      "key_points": ["Discuss rating", "Mention price range"],
      "rag_query": "subject basic info"
    }}
  ]
}}

STRICT JSON. No comments. No trailing commas.
"""
        response = await self.llm.async_generate(prompt, temperature=0.4)
        cleaned_response = self._clean_json_string(response)

        try:
            outline = json.loads(cleaned_response)
            if "sections" in outline and isinstance(outline["sections"], list):
                for i, section in enumerate(outline["sections"]):
                    section["id"] = i + 1
            return outline
        except json.JSONDecodeError:
            logger.error(f"Failed to parse outline JSON. Response was: {response[:200]}...")
            return None
