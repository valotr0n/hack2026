import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


def _available_cpus() -> int:
    return os.cpu_count() or 1


def _default_torch_threads() -> int:
    cpus = _available_cpus()
    if cpus >= 24:
        return min(28, cpus - 4)
    if cpus >= 8:
        return max(4, cpus - 2)
    return cpus


def _default_torch_interop_threads() -> int:
    cpus = _available_cpus()
    if cpus >= 24:
        return 4
    if cpus >= 8:
        return 2
    return 1


def _default_embedding_batch_size() -> int:
    cpus = _available_cpus()
    if cpus >= 24:
        return 128
    if cpus >= 12:
        return 64
    return 32


class Settings(BaseSettings):
    llm_base_url: str = "https://hackai.centrinvest.ru:6630/v1"
    llm_api_key: str = "hackaton2026"
    llm_model: str = "gpt-oss-20b"
    embedder_model: str = "ai-forever/ru-en-RoSBERTa"
    qdrant_url: str = "http://qdrant:6333"
    torch_num_threads: int = _default_torch_threads()
    torch_num_interop_threads: int = _default_torch_interop_threads()
    embedding_batch_size: int = _default_embedding_batch_size()
    vision_enabled: bool = True
    vision_model_id: str = "AvitoTech/avision"
    vision_preload: bool = True
    vision_max_new_tokens: int = 64
    vision_max_image_side: int = 1024
    vision_max_images: int = 5       # макс. картинок на документ
    vision_min_image_side: int = 100  # пропускать картинки меньше N пикселей (логотипы, линейки)
    llm_proxy: str | None = None     # например socks5://ss-proxy:1080
    ollama_base_url: str = "http://ollama:11434/v1"
    ollama_model: str = "qwen2.5:7b"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
