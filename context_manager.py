"""Контекст диалога в памяти и параметры генерации на пользователя."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Ты полезный ассистент в Telegram. Отвечай по-русски, кратко и по делу. "
    "Если нужен пример — дай короткий."
)

CHARS_PER_TOKEN = 4
MESSAGE_OVERHEAD = 4


def estimate_tokens(text: str) -> int:
    """Приблизительная оценка токенов: 1 токен ≈ 4 символа текста."""
    return len(text) // CHARS_PER_TOKEN + MESSAGE_OVERHEAD


def estimate_messages_tokens(messages: list[dict[str, str]]) -> int:
    """Суммарная оценка токенов для списка сообщений."""
    return sum(estimate_tokens(message.get("content", "")) for message in messages)


@dataclass
class GenerationParams:
    """Параметры генерации конкретного пользователя."""

    temperature: float
    max_tokens: int


class ContextManager:
    """Хранит истории и параметры пользователей в оперативной памяти."""

    def __init__(self, token_limit: int, default_temperature: float, default_max_tokens: int) -> None:
        self.token_limit = token_limit
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self._history: dict[int, list[dict[str, str]]] = {}
        self._params: dict[int, GenerationParams] = {}
        self._usage: dict[int, dict[str, int]] = {}

    # --- параметры генерации ---

    def params(self, user_id: int) -> GenerationParams:
        if user_id not in self._params:
            self._params[user_id] = GenerationParams(self.default_temperature, self.default_max_tokens)
        return self._params[user_id]

    def set_temperature(self, user_id: int, value: float) -> GenerationParams:
        params = self.params(user_id)
        params.temperature = value
        return params

    def set_max_tokens(self, user_id: int, value: int) -> GenerationParams:
        params = self.params(user_id)
        params.max_tokens = value
        return params

    # --- история ---

    def history(self, user_id: int) -> list[dict[str, str]]:
        return self._history.setdefault(user_id, [])

    def build_messages(self, user_id: int, text: str) -> list[dict[str, str]]:
        """Собрать запрос: системная инструкция + история + новое сообщение."""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.history(user_id),
            {"role": "user", "content": text},
        ]

    def append_exchange(self, user_id: int, user_text: str, assistant_text: str) -> None:
        """Дописать пару «вопрос — ответ» и при необходимости обрезать историю."""
        history = self.history(user_id)
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": assistant_text})
        self.trim(user_id)

    def trim(self, user_id: int) -> int:
        """Удалять самые старые пары, пока история не влезет в лимит."""
        history = self.history(user_id)
        removed = 0
        while len(history) > 2 and self.used_tokens(user_id) > self.token_limit:
            history.pop(0)
            history.pop(0)
            removed += 2
        if removed:
            logger.info(
                "Контекст user_id=%s обрезан: удалено %s сообщений, осталось %s (~%s токенов)",
                user_id, removed, len(history), self.used_tokens(user_id),
            )
        return removed

    def used_tokens(self, user_id: int) -> int:
        return estimate_messages_tokens(self.history(user_id)) + estimate_tokens(SYSTEM_PROMPT)

    def clear(self, user_id: int) -> int:
        """Очистить историю пользователя. Возвращает число удалённых сообщений."""
        removed = len(self.history(user_id))
        self._history[user_id] = []
        self._usage.pop(user_id, None)
        logger.info("Контекст user_id=%s очищен (было %s сообщений)", user_id, removed)
        return removed

    def stats(self, user_id: int) -> dict[str, int]:
        return {"messages": len(self.history(user_id)), "tokens": self.used_tokens(user_id)}

    # --- расход токенов ---

    def set_usage(self, user_id: int, usage: dict[str, int]) -> None:
        self._usage[user_id] = dict(usage)

    def last_usage(self, user_id: int) -> dict[str, int] | None:
        return self._usage.get(user_id)