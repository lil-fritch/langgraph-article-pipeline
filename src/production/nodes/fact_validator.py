import json
from typing import Dict, List, Literal, Optional, TypedDict

from src.llm_client import LLMClient
from src.utils.logger import setup_logger

logger = setup_logger("fact_validator")


class AnalysisData(TypedDict):
    query_intent: Literal["PRODUCT_SPECIFIC", "GENERIC", "BRAND"]
    detected_subject: Optional[str]
    is_real_subject: bool


class FactsData(TypedDict):
    brand: Optional[str]
    rating: Optional[str]
    category: Optional[str]
    has_notable_feature: bool
    features: List[str]


class DecisionData(TypedDict):
    match_status: Literal[
        "EXACT_MATCH", "FEATURE_MISMATCH", "NON_EXISTENT_SUBJECT", "LOGICAL_CONTRADICTION", "GENERIC_QUERY"
    ]
    selected_writing_strategy: Literal["DIRECT_REVIEW", "MYTH_BUSTER", "CATEGORY_OVERVIEW", "BUYER_GUIDE", "GENERIC_GUIDE"]
    pivot_reason: str


class PriceRange(TypedDict):
    min_price: str
    max_price: str


class TechnicalSpecs(TypedDict):
    review_format: Literal["SPEC_COMPARISON", "PROS_CONS", "RANKED_LIST", "BUYER_GUIDE", "QUICK_VERDICT"]
    rating_value: Optional[str]
    price_range: PriceRange


class ArticlePassport(TypedDict):
    analysis: AnalysisData
    decision: DecisionData
    facts: FactsData
    technical_specs: TechnicalSpecs


_DEFAULT_PASSPORT: ArticlePassport = {
    "analysis": {"query_intent": "GENERIC", "detected_subject": None, "is_real_subject": False},
    "decision": {
        "match_status": "GENERIC_QUERY",
        "selected_writing_strategy": "GENERIC_GUIDE",
        "pivot_reason": "Fallback: missing or unparseable decision data.",
    },
    "facts": {"brand": None, "rating": None, "category": None, "has_notable_feature": False, "features": []},
    "technical_specs": {
        "review_format": "RANKED_LIST",
        "rating_value": "Unknown",
        "price_range": {"min_price": "$0", "max_price": "$0"},
    },
}


class FactValidatorNode:
    """
    Checks the topic against search results before anything gets written, so
    the pipeline doesn't hallucinate a review of a product/feature that
    doesn't exist, or silently contradict itself across sections. Produces
    an ArticlePassport that every downstream node treats as the single
    source of truth.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def run(self, topic: str, search_results: List[Dict]) -> ArticlePassport:
        search_summary = "\n".join(
            f"- Title: {res.get('title', 'No Title')}\n  Snippet: {res.get('description', 'No Description')}"
            for res in search_results
        )

        prompt = f"""
Role: You are an expert Fact-Checker and Content Strategist.
Your goal is to validate the user's query against search results to prevent AI hallucinations.

USER QUERY: "{topic}"

SEARCH SNIPPETS:
{search_summary}

---
TASK:
Analyze if the User Query corresponds to a real, existing product/subject with the requested feature.
Determine the "Truth Status" and select the best writing strategy.

definitions:
1. EXACT_MATCH: Subject exists AND has the requested feature (e.g., "Aurora X1 noise cancelling").
2. FEATURE_MISMATCH: Subject exists, but requested feature is missing.
3. NON_EXISTENT_SUBJECT: Query looks like a specific product name, but no such product exists in snippets.
4. LOGICAL_CONTRADICTION: Query asks for an impossible combo (e.g., "ultra-light heavy-duty jacket").
5. GENERIC_QUERY: Query is about a category/type, not a specific product name.

MAPPING RULES (match_status -> selected_writing_strategy):
- EXACT_MATCH -> DIRECT_REVIEW
- FEATURE_MISMATCH -> MYTH_BUSTER
- NON_EXISTENT_SUBJECT -> CATEGORY_OVERVIEW
- LOGICAL_CONTRADICTION -> BUYER_GUIDE
- GENERIC_QUERY -> GENERIC_GUIDE

OUTPUT SCHEMA (JSON only):
{{
  "analysis": {{
    "query_intent": "PRODUCT_SPECIFIC | GENERIC | BRAND",
    "detected_subject": "String OR null",
    "is_real_subject": true/false
  }},
  "decision": {{
    "match_status": "Status from definitions",
    "selected_writing_strategy": "Strategy from mapping",
    "pivot_reason": "Explain why this strategy was chosen."
  }},
  "facts": {{
    "brand": "String OR null",
    "rating": "String OR null",
    "category": "String OR null",
    "has_notable_feature": true/false,
    "features": ["found feature 1", "found feature 2"]
  }},
  "technical_specs": {{
    "review_format": "SPEC_COMPARISON | PROS_CONS | RANKED_LIST | BUYER_GUIDE | QUICK_VERDICT",
    "rating_value": "4.5/5",
    "price_range": {{ "min_price": "$49", "max_price": "$199" }}
  }}
}}

INSTRUCTIONS:
- Be strict. If the specific product name shows no results, mark NON_EXISTENT_SUBJECT.
- Extract ONLY verified facts from snippets. Do not hallucinate ratings or pricing.
- If NON_EXISTENT_SUBJECT, 'facts' should be empty or general.

NORMALIZATION RULES (critical):
1. Rating fix: if multiple ratings are found, select the one from the most reputable-looking source and put it in 'rating_value'.
2. Price normalizer: convert bare numbers to currency strings, always include a symbol ($/€/£).
"""
        response = await self.llm.async_generate(prompt, temperature=0.0)

        try:
            cleaned = self._extract_json(response)
            passport = json.loads(cleaned)

            for key, default_val in _DEFAULT_PASSPORT.items():
                if key not in passport or not isinstance(passport[key], dict):
                    logger.warning(f"Passport missing '{key}', using default.")
                    passport[key] = default_val

            passport["decision"].setdefault("match_status", "GENERIC_QUERY")
            passport["decision"].setdefault("selected_writing_strategy", "GENERIC_GUIDE")

            logger.info(
                f"Passport generated: {passport['decision']['match_status']} -> "
                f"{passport['decision']['selected_writing_strategy']}"
            )
            return passport
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to parse passport JSON. Error: {e}. Response: {response[:200]}")
            return dict(_DEFAULT_PASSPORT)

    @staticmethod
    def _extract_json(text: str) -> str:
        cleaned = text.strip()
        if "```" in cleaned:
            for part in cleaned.split("```"):
                if "{" in part and "}" in part:
                    cleaned = part
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:]
                    break
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start : end + 1]
        return cleaned
