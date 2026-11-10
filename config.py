"""Настройки бота: чтение .env и значения по умолчанию."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    """Все настройки приложения в одном месте."""

    bot_token: str
    api_key: str
    base_url: str
    model: str
    default_temperature: float
    default_max_tokens: int
    history_token_limit: int
    thinking_mode: str
    log_level: str
    request_timeout: float
    retry_attempts: int

    def validate(self) -> None:
        """Проверить, что обязательные секреты заданы."""
        if not self.bot_token:
            raise ValueError("В .env не задан BOT_TOKEN (токен от @BotFather)")
        if not self.api_key:
            raise ValueError("В .env не задан DEEPSEEK_API_KEY (ключ platform.deepseek.com)")


settings = Settings(
    bot_token=os.getenv("BOT_TOKEN", "").strip(),
    api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip(),
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash").strip(),
    default_temperature=_get_float("DEFAULT_TEMPERATURE", 0.7),
    default_max_tokens=_get_int("DEFAULT_MAX_TOKENS", 512),
    history_token_limit=_get_int("HISTORY_TOKEN_LIMIT", 6000),
    thinking_mode=os.getenv("THINKING_MODE", "disabled").strip().lower(),
    log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
    request_timeout=_get_float("REQUEST_TIMEOUT", 60.0),
    retry_attempts=_get_int("RETRY_ATTEMPTS", 2),
)