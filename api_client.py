"""Клиент DeepSeek API (формат OpenAI Chat Completions)."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Ошибка обращения к LLM. Текст user_message можно показывать пользователю."""

    user_message = "Не удалось получить ответ. Попробуйте позже."

    def __init__(self, details: str = "") -> None:
        super().__init__(details or self.user_message)
        self.details = details


class AuthError(LLMError):
    user_message = "Сервис недоступен: проверьте ключ API."


class BalanceError(LLMError):
    user_message = "Закончился баланс API. Пополните счёт в кабинете провайдера."


class RateLimitError(LLMError):
    user_message = "Слишком много запросов. Попробуйте через минуту."


class ServiceError(LLMError):
    user_message = "Сервис временно недоступен. Попробуйте позже."


class BadRequestError(LLMError):
    user_message = "Некорректные параметры запроса."


@dataclass
class LLMResponse:
    """Ответ модели: текст и статистика расхода токенов."""

    text: str
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    finish_reason: str = ""


class DeepSeekClient:
    """Асинхронный клиент к /chat/completions с повторами на 429 и 5xx."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        thinking_mode: str = "disabled",
        timeout: float = 60.0,
        retry_attempts: int = 2,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._thinking_mode = thinking_mode.strip().lower()
        self._timeout = timeout
        self._retry_attempts = max(0, retry_attempts)

    @property
    def model(self) -> str:
        return self._model

    def _payload(self, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> dict:
        payload: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        # В режиме размышлений temperature не действует, поэтому по умолчанию режим выключен
        if self._thinking_mode in ("enabled", "disabled"):
            payload["thinking"] = {"type": self._thinking_mode}
        return payload

    async def chat(self, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> LLMResponse:
        payload = self._payload(messages, temperature, max_tokens)
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        last_error: LLMError = ServiceError("запрос не выполнен")
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for attempt in range(self._retry_attempts + 1):
                try:
                    response = await client.post(url, json=payload, headers=headers)
                except httpx.TimeoutException as exc:
                    last_error = ServiceError(f"таймаут запроса: {exc}")
                except httpx.HTTPError as exc:
                    last_error = ServiceError(f"сетевая ошибка: {exc}")
                else:
                    if response.status_code == 200:
                        return self._parse(response)
                    last_error = self._error_from(response)

                if not isinstance(last_error, (RateLimitError, ServiceError)):
                    raise last_error

                if attempt < self._retry_attempts:
                    pause = 1.5 ** attempt
                    logger.warning(
                        "Повтор запроса через %.1f с (попытка %s из %s): %s",
                        pause, attempt + 1, self._retry_attempts + 1, last_error.details,
                    )
                    await asyncio.sleep(pause)

        raise last_error

    @staticmethod
    def _error_from(response: httpx.Response) -> LLMError:
        body = response.text[:500]
        status = response.status_code
        if status == 401:
            return AuthError(f"401: {body}")
        if status == 402:
            return BalanceError(f"402: {body}")
        if status == 429:
            return RateLimitError(f"429: {body}")
        if status == 400:
            return BadRequestError(f"400: {body}")
        return ServiceError(f"{status}: {body}")

    @staticmethod
    def _parse(response: httpx.Response) -> LLMResponse:
        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = (message.get("content") or "").strip()

        usage_raw = data.get("usage") or {}
        usage: dict[str, int] = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if isinstance(usage_raw.get(key), int):
                usage[key] = usage_raw[key]
        details = usage_raw.get("prompt_tokens_details") or {}
        for key in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "cached_tokens"):
            value = usage_raw.get(key, details.get(key))
            if isinstance(value, int):
                usage[key] = value

        result = LLMResponse(
            text=text,
            usage=usage,
            model=data.get("model", ""),
            finish_reason=choice.get("finish_reason", ""),
        )
        logger.info(
            "Ответ модели: model=%s finish=%s tokens=%s",
            result.model, result.finish_reason, result.usage,
        )
        return result