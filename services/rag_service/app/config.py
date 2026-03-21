from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    llm_base_url: str = "https://hackai.centrinvest.ru:6630/v1"
    llm_api_key: str = "hackaton2026"
    llm_model: str = "gpt-oss-20b"
    embedder_model: str = "ai-forever/ru-en-RoSBERTa"
    qdrant_url: str = "http://qdrant:6333"
    vision_enabled: bool = True
    vision_model_id: str = "AvitoTech/avision"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()