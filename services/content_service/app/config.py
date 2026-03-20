import os


class Settings:
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://hackai.centrinvest.ru:6630/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", "hackaton2026")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-oss-20b")

    tts_voice_alex: str = os.getenv("TTS_VOICE_ALEX", "ru-RU-DmitryNeural")
    tts_voice_maria: str = os.getenv("TTS_VOICE_MARIA", "ru-RU-SvetlanaNeural")

    audio_dir: str = os.getenv("AUDIO_DIR", "/app/audio")


settings = Settings()
