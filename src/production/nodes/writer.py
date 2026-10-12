import json
import re
from typing import Any, Dict, List, Optional

from src.llm_client import LLMClient
from src.production.nodes.fact_validator import ArticlePassport
from src.utils.logger import setup_logger

logger = setup_logger("writer")


class SectionWriterNode:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def retrieve_context(self, retriever: Any, query: str, limit: int = 4) -> str:
        if not retriever:
            return ""
        try:
            chunks = await retriever(query)
            return "\n\n".join(chunks[:limit])
        except Exception as e:
            logger.error(f"Retrieval failed for query '{query}': {e}")
            return ""

    def format_specs(self, topic_specs: Optional[Dict[str, Any]]) -> str:
        if not topic_specs:
            return "No verifiable data specs available locally."
        lines = ["--- LOCAL FACTS (ground truth) ---"]
        for k, v in topic_specs.items():
            if v is not None:
                lines.append(f"{k}: {v}")
        lines.append("-----------------------------------")
        return "\n".join(lines)

    async def write_section(
        self,
        topic: str,
        section: Dict[str, Any],
        specs_text: str,
        context_text: str,
        passport: Optional[ArticlePassport] = None,
    ) -> str:
        section_title = section.get("title", "")
        section_intent = section.get("user_intent", "Inform and engage")
        key_points = section.get("key_points", [])
        points_str = "\n".join(f"- {p}" for p in key_points) if key_points else "Cover the section topic comprehensively."

        strategy_info = ""
        if passport:
            decision = passport.get("decision", {})
            facts = passport.get("facts", {})
            tech_specs = passport.get("technical_specs", {})

            rating_val = tech_specs.get("rating_value", "N/A")
            review_format = tech_specs.get("review_format", "RANKED_LIST")
            price_range = tech_specs.get("price_range", {"min_price": "$0", "max_price": "$0"})
            min_price = price_range.get("min_price", "$0")
            max_price = price_range.get("max_price", "$0")

            strategy_info = f"""
STRICT STRATEGY INSTRUCTIONS:
- Strategy: {decision.get('selected_writing_strategy')}
- Pivot Reason: {decision.get('pivot_reason')}
- Verified Facts: {json.dumps(facts)}

--- CRITICAL: ONE TRUTH POLICY (overrides all other data) ---
1. RATING: You MUST use "{rating_val}" as the definitive rating. Ignore any other rating found in context.
2. PRICE: You MUST use "{min_price}" to "{max_price}". Never use bare numbers like "20".
3. FORMAT:
   - Type: {review_format}
   - IF 'SPEC_COMPARISON': present specs as a clear comparison, not prose paragraphs.
   - IF 'RANKED_LIST': use explicit ranking language ("#1", "runner-up", etc.).
   - IF 'PROS_CONS': structure around clear pros/cons, not a narrative review.
-------------------------------------------------------------
- Do not invent facts not listed in the 'Verified Facts' section.
"""

        prompt = f"""
Role: You are a Conversion-Focused Product Copywriter.
Your goal is to write a compelling section for a product review that drives reader interest while remaining factual.

TOPIC: "{topic}"
SECTION TITLE: "{section_title}"
READER INTENT: {section_intent}

INPUT DATA:
1. HARD SPECS (axioms — never deviate):
{specs_text}

2. STRATEGY POINTS (must cover these):
{points_str}

3. RESEARCH CONTEXT (use for depth/details):
{context_text}

4. STRATEGY COMPLIANCE:
{strategy_info}

--------------------------------------------------
STRICT PROHIBITIONS (violations = failure):
1. NO EXTERNAL LINKS: do not include URLs, domain names, or hyperlinks.
2. NO CITATIONS: do not write "According to [Source]". Integrate facts naturally.
3. NO FAKE RETAILERS: do not invent retailer/brand names or placeholders like [TOP_STORE].
   Instead, use generic terms like "trusted retailers" or "official stores".
4. NO COMPETITOR MENTIONS: do not mention other review sites by name.
--------------------------------------------------

WRITING INSTRUCTIONS:
- Tone: expert, enthusiastic, but grounded in the verified facts.
- Formatting: clean Markdown only. Use **bold** for key metrics, not HTML tags.
- Structure: use ## or ### headers for logical subsections where appropriate.
- Conflict resolution: if research context contradicts hard specs, trust hard specs.
- Flow: do not start with "In this section...". Jump straight into the value.
- Anti-repetition: do not repeat the same metric (rating, price, feature) more than once in this section;
  omit it if it's not essential here. Avoid reusing identical phrasing from other sections.
- Myth-buster pivot: if debunking a claim, state the truth naturally in your own words — vary phrasing
  across articles — then offer a practical alternative (free trial, warranty, budget alternative, where to buy)
  without naming specific retailers or inventing offers.
- Prefer 2-3 concrete, actionable tips over generic filler.

OUTPUT:
Write ONLY the content for the section "{section_title}".
Do NOT prefix content with labels like "Hook:", "Introduction:", etc. Start directly with the content.
"""
        content = await self.llm_client.async_generate(prompt)
        content = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", content)
        content = re.sub(r"https?://\S+", "", content)
        content = re.sub(r"\b\w+\.(com|org|net|io)\b", "", content)
        return content.strip()

    async def run(
        self,
        topic: str,
        outline: Dict,
        topic_specs: Dict,
        retriever: Any,
        passport: Optional[ArticlePassport] = None,
    ) -> Dict[str, str]:
        if not outline or "sections" not in outline:
            logger.warning("No outline provided to writer.")
            return {}

        logger.info(f"Writing {len(outline['sections'])} sections for '{topic}'...")
        specs_text = self.format_specs(topic_specs)
        written_sections = {}

        for section in outline["sections"]:
            s_id = section.get("id")
            s_title = section.get("title")
            query = f"{s_title} {section.get('description', '')}"

            context_text = await self.retrieve_context(retriever, query)
            content = await self.write_section(topic, section, specs_text, context_text, passport)

            written_sections[s_id] = content
            logger.info(f"Completed section {s_id}: {s_title}")

        return written_sections
