import os
import shutil
from typing import Callable, List

import lancedb

from src.config import DB_DIR, settings
from src.llm_client import LLMClient
from src.utils.logger import setup_logger

logger = setup_logger("rag_store")


class EphemeralRAG:
    """
    Temporary per-session LanceDB vector store for RAG. Embeddings come from
    the OpenAI API (same client as the rest of the pipeline) rather than a
    locally downloaded model, so there's nothing to load into memory or keep
    warm between runs — the tradeoff is a network round-trip per chunk batch.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.base_dir = DB_DIR / "lancedb_sessions"
        os.makedirs(self.base_dir, exist_ok=True)
        self.current_session_path = None
        self.db = None
        self.tbl = None

    def init_session(self, session_id: str):
        """Fresh, physically isolated DB directory per topic/session."""
        self.tbl = None
        safe_id = "".join(x for x in session_id if x.isalnum())[:50]
        self.current_session_path = self.base_dir / f"session_{safe_id}"

        if self.current_session_path.exists():
            logger.info(f"Removing existing RAG session: {self.current_session_path}")
            shutil.rmtree(self.current_session_path)

        os.makedirs(self.current_session_path, exist_ok=True)
        self.db = lancedb.connect(str(self.current_session_path))

    async def index_chunks(self, chunks: List[str]):
        if not chunks:
            logger.warning("No chunks to index.")
            return

        if "chunks" in self.db.table_names():
            self.db.drop_table("chunks")

        logger.info(f"Embedding {len(chunks)} chunks via OpenAI...")
        embeddings = await self.llm.async_get_embeddings_batch(chunks)

        data = []
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            if not vector:
                continue
            source = "unknown"
            if "Source: " in chunk:
                try:
                    source = chunk.split("Source: ")[1].split("\n")[0].strip()
                except Exception:
                    pass
            data.append({"vector": vector, "text": chunk, "source": source, "id": i})

        if not data:
            logger.warning("No chunks embedded successfully — skipping index.")
            return

        self.tbl = self.db.create_table("chunks", data)
        logger.info(f"Indexed {len(data)} chunks.")

    def as_retriever(self, limit: int = 5) -> Callable[[str], "asyncio.Future"]:
        """Returns an async callable retriever: `await retriever(query)`."""

        async def retrieve(query: str) -> List[str]:
            if not self.tbl:
                logger.warning("Retriever called but table is None.")
                return []
            try:
                query_vec = await self.llm.async_get_embeddings(query)
                if not query_vec:
                    return []
                results = self.tbl.search(query_vec).limit(limit).to_list()
                return [r["text"] for r in results]
            except Exception as e:
                logger.error(f"Retrieval error: {e}")
                return []

        return retrieve

    def cleanup(self, force: bool = False):
        self.tbl = None
        self.db = None
        if self.current_session_path and self.current_session_path.exists():
            if settings.DEBUG and not force:
                logger.info(f"DEBUG: keeping RAG session at {self.current_session_path}")
            else:
                try:
                    shutil.rmtree(self.current_session_path)
                except Exception as e:
                    logger.error(f"Failed to delete session dir: {e}")
        self.current_session_path = None
