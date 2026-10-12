import aiohttp
import asyncio
from functools import partial

import trafilatura

from src.utils.logger import setup_logger

logger = setup_logger("scraper_service")

# Browser-like headers to avoid being blocked
_DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


class ScraperService:
    """
    Service to fetch and extract page content using aiohttp + trafilatura.
    """

    def __init__(self, timeout: int = 30):
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers=_DEFAULT_HEADERS,
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def fetch_content(self, url: str) -> str:
        """
        Fetches the page HTML and extracts the main content as Markdown
        using trafilatura.
        """
        try:
            session = await self._get_session()
            async with session.get(url) as response:
                if response.status == 429:
                    logger.warning(f"Rate limit hit for {url}. Sleeping 2s...")
                    await asyncio.sleep(2)
                    return ""

                if response.status != 200:
                    logger.warning(f"Failed to fetch {url}: Status {response.status}")
                    return ""

                html = await response.text()

            # trafilatura is synchronous — run in a thread pool
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(
                None,
                partial(
                    trafilatura.extract,
                    html,
                    output_format="markdown",
                    include_links=False,
                    include_images=False,
                    include_tables=True,
                ),
            )

            if not text or len(text) < 100:
                logger.warning(f"Content too short for {url} ({len(text) if text else 0} chars). Skipping.")
                return ""

            return text

        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return ""
