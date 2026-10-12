from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenAI (cloud API — no local model server required)
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_IMAGE_MODEL: str = "dall-e-3"
    LLM_DELAY: float = 0.5  # polite delay between requests

    # Web search backend (SearXNG — self-hosted or any public instance)
    SEARCHXNG_HOST: str = ""

    # Telegram notifications (optional — pipeline runs fine without it)
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    DEBUG: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "output"
ARTICLES_DIR = OUTPUT_DIR / "articles"
DB_DIR = DATA_DIR / "db"

TOPIC_FACTS_CSV = DATA_DIR / "topic_facts.example.csv"
