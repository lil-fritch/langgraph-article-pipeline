"""
Generate articles for a list of topics using the LangGraph pipeline in
src/production/graph.py: query generation -> web search -> fact validation
-> outline -> RAG-grounded section writing -> compile to Markdown.

Usage:
    python main.py --topics topics.example.txt
    python main.py --topics topics.example.txt --limit 3
    python main.py --topics topics.example.txt --with-cover
"""
import argparse
import asyncio
import sys
import traceback

from src.config import ARTICLES_DIR, OUTPUT_DIR, settings
from src.llm_client import LLMClient
from src.production.graph import GraphBuilder
from src.services.image_generator import generate_article_cover
from src.utils.filename_utils import get_safe_filename
from src.utils.logger import setup_logger
from src.utils.topic_loader import load_topics

logger = setup_logger("main")


async def run(topics_file: str, limit: int, with_cover: bool):
    if not settings.OPENAI_API_KEY:
        logger.critical("OPENAI_API_KEY is not set. Add it to .env (see env.example).")
        sys.exit(1)

    topics = load_topics(topics_file, limit=limit)
    if not topics:
        logger.error(f"No topics loaded from {topics_file}.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTICLES_DIR.mkdir(parents=True, exist_ok=True)

    llm = LLMClient()

    for i, topic in enumerate(topics):
        safe_name = get_safe_filename(topic)
        existing_path = ARTICLES_DIR / f"{safe_name}.md"
        if existing_path.exists():
            logger.info(f"[{i + 1}/{len(topics)}] SKIP (already generated): {topic}")
            continue

        logger.info(f"[{i + 1}/{len(topics)}] Generating: {topic}")

        # Fresh graph per article: simplest way to guarantee clean state —
        # no leftover retriever/outline from a previous topic bleeding through.
        app = GraphBuilder(llm).build()
        initial_state = {
            "topic_data": {"topic": topic},
            "topic_specs": None,
            "specs_missing": True,
            "search_queries": [],
            "search_results": [],
            "outline": None,
            "rag_chunks": [],
        }

        try:
            final_state = await app.ainvoke(initial_state)
            final_path = final_state.get("final_article_path")

            if final_path:
                logger.info(f"[{i + 1}] SUCCESS -> {final_path}")
                if with_cover:
                    cover = await generate_article_cover(
                        llm,
                        topic,
                        final_state.get("topic_specs"),
                        final_state.get("outline"),
                        safe_name,
                        save_to_disk=True,
                    )
                    if cover:
                        logger.info(f"[{i + 1}] Cover saved: {cover}")
            else:
                logger.error(f"[{i + 1}] FAILED — pipeline did not produce a final article.")
        except Exception as e:
            logger.error(f"[{i + 1}] EXCEPTION: {e}")
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Article generation pipeline (LangGraph + OpenAI)")
    parser.add_argument("--topics", required=True, help="Path to a topics file (.txt, one per line, or .csv with a 'topic' column)")
    parser.add_argument("--limit", type=int, default=10, help="Max number of topics to process")
    parser.add_argument("--with-cover", action="store_true", help="Also generate a cover image per article")
    args = parser.parse_args()

    asyncio.run(run(args.topics, args.limit, args.with_cover))


if __name__ == "__main__":
    main()
