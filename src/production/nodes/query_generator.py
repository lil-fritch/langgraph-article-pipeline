import datetime
import json
import re

from src.llm_client import LLMClient
from src.utils.logger import setup_logger

logger = setup_logger("query_gen")


class QueryGeneratorNode:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def run(self, topic: str, topic_specs: dict = None) -> list[str]:
        """Generates search queries to gather data for an article on `topic`."""
        current_year = datetime.date.today().year
        specs_context = self._format_specs_for_prompt(topic_specs)

        prompt = f"""
        Role: You are an elite SEO Strategist designed to construct high-precision search queries for Google.

        Task: Generate a list of 3-5 Google search queries to gather comprehensive data for an article on:
        TOPIC: "{topic}"

        {specs_context}
        ---

        Guidelines:
        1. **Focus on Gaps:** Compare the TOPIC with the Context Data. Search ONLY for missing information.
        2. **Intent Diversity:** Cover these angles:
           - *Commercial:* Best options/pricing (include "{current_year}").
           - *Informational:* How it works, common issues, alternatives.
           - *Social Proof:* Real discussions (e.g., use "site:reddit.com").
        3. **Language:** English.

        Output Format:
        Return a valid JSON object with a single key "queries" containing the list of strings.
        STRICT JSON ONLY. No comments (// or #). No intro text.
        Example: {{ "queries": ["query 1", "query 2", "query 3"] }}
        """

        response = await self.llm.async_generate(prompt, temperature=0.5)

        cleaned_response = response.strip()
        json_match = re.search(r"\{.*\}", cleaned_response, re.DOTALL)
        if json_match:
            cleaned_response = json_match.group(0)
        cleaned_response = re.sub(r"^\s*//.*$", "", cleaned_response, flags=re.MULTILINE)
        cleaned_response = re.sub(r"/\*.*?\*/", "", cleaned_response, flags=re.DOTALL)
        cleaned_response = re.sub(r",\s*([\]}])", r"\1", cleaned_response)

        try:
            data = json.loads(cleaned_response)
            if isinstance(data, dict) and "queries" in data:
                return data["queries"]
            elif isinstance(data, list):
                return data
            return list(data.values())[0] if data else [topic]
        except json.JSONDecodeError:
            logger.warning(f"JSON parsing error, falling back to template queries. Raw: {response[:200]}")
            return [
                f"{topic} review",
                f"{topic} pricing {current_year}",
                f"{topic} alternatives",
            ]

    def _format_specs_for_prompt(self, topic_specs: dict) -> str:
        if not topic_specs:
            return "Context Data: None (find everything via search)."

        readable_specs = []
        if topic_specs.get("name"):
            readable_specs.append(f"Name: {topic_specs['name']}")
        if topic_specs.get("brand"):
            readable_specs.append(f"Brand: {topic_specs['brand']}")
        if topic_specs.get("category"):
            readable_specs.append(f"Category: {topic_specs['category']}")
        if topic_specs.get("rating") and str(topic_specs["rating"]).strip():
            readable_specs.append(f"Rating: {topic_specs['rating']}")
        if topic_specs.get("tags"):
            readable_specs.append(f"Tags: {topic_specs['tags']}")

        missing_fields = []
        if not topic_specs.get("price_min"):
            missing_fields.append("price range")
        if not topic_specs.get("rating"):
            missing_fields.append("rating")

        output = "KNOWN FACTS (do not search for these):\n"
        output += "\n".join(f"- {line}" for line in readable_specs)
        if missing_fields:
            output += "\n\nMISSING DATA (priority to search):\n"
            output += f"We explicitly lack: {', '.join(missing_fields)}. Generate queries to find these specifically."
        return output
