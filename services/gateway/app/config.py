from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None and value != "" else default


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    rag_service_url: str
    content_service_url: str
    cors_origins: list[str]
    request_timeout_seconds: float
    contract_max_text_length: int
    timeline_max_text_length: int
    questions_max_text_length: int
    compare_max_text_length: int
    presentation_max_text_length: int
    gateway_max_connections: int
    gateway_max_keepalive_connections: int
    http_port: int
    https_port: int
    tls_cert_file: str
    tls_key_file: str
    auth_secret_key: str
    db_path: str


settings = Settings(
    rag_service_url=_env("RAG_SERVICE_URL", "http://rag_service:8001"),
    content_service_url=_env("CONTENT_SERVICE_URL", "http://content_service:8002"),
    cors_origins=_split_csv(_env("CORS_ORIGINS", "*")),
    request_timeout_seconds=float(_env("REQUEST_TIMEOUT_SECONDS", "300")),
    contract_max_text_length=int(_env("CONTRACT_MAX_TEXT_LENGTH", "16000")),
    timeline_max_text_length=int(_env("TIMELINE_MAX_TEXT_LENGTH", "12000")),
    questions_max_text_length=int(_env("QUESTIONS_MAX_TEXT_LENGTH", "14000")),
    compare_max_text_length=int(_env("COMPARE_MAX_TEXT_LENGTH", "12000")),
    presentation_max_text_length=int(_env("PRESENTATION_MAX_TEXT_LENGTH", "14000")),
    gateway_max_connections=int(_env("GATEWAY_MAX_CONNECTIONS", "200")),
    gateway_max_keepalive_connections=int(_env("GATEWAY_MAX_KEEPALIVE_CONNECTIONS", "50")),
    http_port=int(_env("GATEWAY_HTTP_PORT", "8000")),
    https_port=int(_env("GATEWAY_HTTPS_PORT", "443")),
    tls_cert_file=_env("TLS_CERT_FILE", "/certs/server.crt"),
    tls_key_file=_env("TLS_KEY_FILE", "/certs/server.key"),
    auth_secret_key=_env("AUTH_SECRET_KEY", "change-me-in-production"),
    db_path=_env("DB_PATH", "/app/data/platform.db"),
)
