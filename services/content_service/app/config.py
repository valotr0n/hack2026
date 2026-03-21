import os


class Settings:
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://hackai.centrinvest.ru:6630/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", "hackaton2026")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-oss-20b")

    stt_base_url: str = os.getenv("STT_BASE_URL", "https://hackai.centrinvest.ru:6640/v1")
    stt_api_key: str = os.getenv("STT_API_KEY", "hackaton2026")
    # api | local | auto (попробовать API, при ошибке — локальный)
    stt_mode: str = os.getenv("STT_MODE", "auto")
    whisper_model_size: str = os.getenv("WHISPER_MODEL_SIZE", "small")

    # Закрытый контур — ollama
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434/v1")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

    tts_voice_alex: str = os.getenv("TTS_VOICE_ALEX", "ru-RU-DmitryNeural")
    tts_voice_maria: str = os.getenv("TTS_VOICE_MARIA", "ru-RU-SvetlanaNeural")

    audio_dir: str = os.getenv("AUDIO_DIR", "/app/audio")


settings = Settings()
