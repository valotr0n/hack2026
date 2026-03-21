import os


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None and value != "" else default


def _env_int(name: str, default: int) -> int:
    return int(_env_str(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env_str(name, str(default)))


class Settings:
    llm_base_url: str = _env_str("LLM_BASE_URL", "https://hackai.centrinvest.ru:6630/v1")
    llm_api_key: str = _env_str("LLM_API_KEY", "hackaton2026")
    llm_model: str = _env_str("LLM_MODEL", "gpt-oss-20b")
    llm_open_timeout_seconds: float = _env_float("LLM_OPEN_TIMEOUT_SECONDS", 180.0)
    llm_open_max_connections: int = _env_int("LLM_OPEN_MAX_CONNECTIONS", 200)
    llm_open_max_keepalive_connections: int = _env_int("LLM_OPEN_MAX_KEEPALIVE_CONNECTIONS", 50)

    stt_base_url: str = _env_str("STT_BASE_URL", "https://hackai.centrinvest.ru:6640/v1")
    stt_api_key: str = _env_str("STT_API_KEY", "hackaton2026")
    # api | local | auto (попробовать API, при ошибке — локальный)
    stt_mode: str = _env_str("STT_MODE", "auto")
    whisper_model_size: str = _env_str("WHISPER_MODEL_SIZE", "small")

    # Закрытый контур — ollama
    ollama_base_url: str = _env_str("OLLAMA_BASE_URL", "http://ollama:11434/v1")
    ollama_model: str = _env_str("OLLAMA_MODEL", "qwen2.5:7b")
    llm_closed_timeout_seconds: float = _env_float("LLM_CLOSED_TIMEOUT_SECONDS", 300.0)
    llm_closed_max_connections: int = _env_int("LLM_CLOSED_MAX_CONNECTIONS", 64)
    llm_closed_max_keepalive_connections: int = _env_int("LLM_CLOSED_MAX_KEEPALIVE_CONNECTIONS", 16)

    tts_voice_alex: str = _env_str("TTS_VOICE_ALEX", "ru-RU-DmitryNeural")
    tts_voice_maria: str = _env_str("TTS_VOICE_MARIA", "ru-RU-SvetlanaNeural")

    audio_dir: str = _env_str("AUDIO_DIR", "/app/audio")
    timeline_max_tokens: int = _env_int("TIMELINE_MAX_TOKENS", 700)
    contract_extract_max_tokens: int = _env_int("CONTRACT_EXTRACT_MAX_TOKENS", 1200)
    contract_verify_max_tokens: int = _env_int("CONTRACT_VERIFY_MAX_TOKENS", 1200)
    questions_max_tokens: int = _env_int("QUESTIONS_MAX_TOKENS", 1000)
    compare_max_tokens: int = _env_int("COMPARE_MAX_TOKENS", 1400)


settings = Settings()
