"""
LLM client backed by the OpenAI API — chat completions, embeddings, and
image generation, all through one cloud provider instead of a mix of local
model servers with their own ad-hoc HTTP contracts.

Kept as a thin wrapper (not calling the SDK directly from every node) so the
rest of the pipeline — every node in src/production/nodes — doesn't care
which provider is behind it, and so a failure streak can still raise instead
of silently producing empty articles.
"""
import asyncio

from openai import AsyncOpenAI, OpenAIError

from src.config import settings
from src.services.telegram_bot import telegram_bot
from src.utils.logger import setup_logger

logger = setup_logger("llm_client")

_FAILURE_ALERT_THRESHOLD = 10


class LLMClient:
    def __init__(
        self,
        model: str = None,
        embedding_model: str = None,
        image_model: str = None,
    ):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = model or settings.OPENAI_MODEL
        self.embedding_model = embedding_model or settings.OPENAI_EMBEDDING_MODEL
        self.image_model = image_model or settings.OPENAI_IMAGE_MODEL
        self._consecutive_errors = 0

    def _record_success(self):
        self._consecutive_errors = 0

    async def _record_failure(self, context: str):
        self._consecutive_errors += 1
        logger.warning(f"LLM failure #{self._consecutive_errors}: {context}")
        if self._consecutive_errors >= _FAILURE_ALERT_THRESHOLD:
            msg = f"LLM failed {_FAILURE_ALERT_THRESHOLD} requests in a row. Stopping."
            await telegram_bot.send_message(msg)
            raise RuntimeError(msg)

    # ------------------------------------------------------------------ text

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Sync wrapper — most nodes call async_generate directly instead."""
        return asyncio.run(self.async_generate(prompt, temperature))

    async def async_generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        system: str = "You are a stateless assistant. Focus only on the current request.",
    ) -> str:
        await asyncio.sleep(settings.LLM_DELAY)

        for attempt in range(3):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                )
                self._record_success()
                return response.choices[0].message.content or ""
            except OpenAIError as e:
                await self._record_failure(f"chat.completions attempt {attempt + 1}/3: {e}")
                await asyncio.sleep((attempt + 1) * 2)

        logger.error("LLM generation failed after 3 attempts.")
        return ""

    # ----------------------------------------------------------- embeddings

    async def async_get_embeddings(self, text: str) -> list[float]:
        try:
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=text,
            )
            self._record_success()
            return response.data[0].embedding
        except OpenAIError as e:
            await self._record_failure(f"embeddings: {e}")
            return []

    async def async_get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch form — one request for N chunks instead of N requests."""
        if not texts:
            return []
        try:
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=texts,
            )
            self._record_success()
            return [d.embedding for d in response.data]
        except OpenAIError as e:
            await self._record_failure(f"embeddings batch: {e}")
            return [[] for _ in texts]

    # --------------------------------------------------------------- images

    async def async_generate_image(self, prompt: str) -> str:
        try:
            response = await self.client.images.generate(
                model=self.image_model,
                prompt=prompt,
                n=1,
                size="1024x1024",
            )
            self._record_success()
            return response.data[0].url or ""
        except OpenAIError as e:
            await self._record_failure(f"image generation: {e}")
            return ""

    async def async_download_image(self, image_url: str, dest_path: str) -> bool:
        if not image_url:
            return False
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status != 200:
                        logger.error(f"Image download failed ({response.status}): {image_url}")
                        return False
                    with open(dest_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
            return True
        except Exception as e:
            logger.error(f"Image download error: {e}")
            return False
