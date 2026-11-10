"""Telegram-бот с контекстом диалога на DeepSeek API.

Запуск: python bot.py
Команды: /start, /help, /reset, /settings, /temperature, /tokens, /usage
"""

from __future__ import annotations

import asyncio
import logging
from logging.handlers import RotatingFileHandler

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandObject
from aiogram.types import Message

from api_client import DeepSeekClient, LLMError
from config import settings
from context_manager import ContextManager, estimate_messages_tokens

logger = logging.getLogger("bot")

dp = Dispatcher()

context = ContextManager(
    token_limit=settings.history_token_limit,
    default_temperature=settings.default_temperature,
    default_max_tokens=settings.default_max_tokens,
)

client = DeepSeekClient(
    api_key=settings.api_key,
    base_url=settings.base_url,
    model=settings.model,
    thinking_mode=settings.thinking_mode,
    timeout=settings.request_timeout,
    retry_attempts=settings.retry_attempts,
)

CLEAR_PHRASES = {"очистить контекст", "очисти контекст", "сброс контекста", "забудь всё", "забудь все"}

TELEGRAM_LIMIT = 4000


def setup_logging() -> None:
    console = logging.StreamHandler()
    file = RotatingFileHandler("bot.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[console, file],
    )


def cut(text: str) -> str:
    """Обрезать длинный ответ под лимит сообщения Telegram."""
    if len(text) <= TELEGRAM_LIMIT:
        return text
    return text[:TELEGRAM_LIMIT] + "\n\n…ответ обрезан, увеличьте max_tokens"


@dp.message(Command("start"))
async def cmd_start(message: Message) -> None:
    await message.answer(
        f"Привет! Я бот с памятью диалога на модели {settings.model}.\n\n"
        "Просто напиши сообщение — я отвечу и запомню контекст беседы.\n\n"
        "Команды:\n"
        "/reset — очистить контекст\n"
        "/settings — текущие параметры\n"
        "/temperature 0.2 — креативность (0..2)\n"
        "/tokens 800 — предел длины ответа (64..8192)\n"
        "/usage — токены последнего ответа\n"
        "/help — помощь"
    )


@dp.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "Как пользоваться:\n"
        "1. Пиши сообщение — получишь ответ модели.\n"
        f"2. Бот помнит последние сообщения (до {settings.history_token_limit} токенов).\n"
        "3. /reset — забыть всё.\n\n"
        "Параметры генерации:\n"
        f"/temperature <0..2> — креативность (по умолчанию {settings.default_temperature}).\n"
        f"/tokens <64..8192> — длина ответа (по умолчанию {settings.default_max_tokens}).\n"
        "/usage — сколько токенов истрачено на последний ответ."
    )


@dp.message(Command("reset"))
async def cmd_reset(message: Message) -> None:
    user_id = message.from_user.id
    removed = context.clear(user_id)
    await message.answer(f"Контекст очищен. Удалено сообщений: {removed}.")


@dp.message(Command("settings"))
async def cmd_settings(message: Message) -> None:
    user_id = message.from_user.id
    params = context.params(user_id)
    stats = context.stats(user_id)
    await message.answer(
        f"Модель: {settings.model}\n"
        f"Режим мышления: {settings.thinking_mode}\n"
        f"temperature: {params.temperature}\n"
        f"max_tokens: {params.max_tokens}\n"
        f"История: {stats['messages']} сообщений, ~{stats['tokens']} из {settings.history_token_limit} токенов"
    )


@dp.message(Command("temperature"))
async def cmd_temperature(message: Message, command: CommandObject) -> None:
    user_id = message.from_user.id
    if not command.args:
        await message.answer(f"Текущая temperature: {context.params(user_id).temperature}")
        return

    try:
        value = float(command.args.replace(",", "."))
    except ValueError:
        await message.answer("Нужно число от 0 до 2, например: /temperature 0.3")
        return

    if not 0.0 <= value <= 2.0:
        await message.answer("temperature должна быть в диапазоне 0..2")
        return

    context.set_temperature(user_id, value)
    logger.info("user_id=%s установил temperature=%s", user_id, value)
    await message.answer(f"temperature = {value}")


@dp.message(Command("tokens"))
async def cmd_tokens(message: Message, command: CommandObject) -> None:
    user_id = message.from_user.id
    if not command.args:
        await message.answer(f"Текущий max_tokens: {context.params(user_id).max_tokens}")
        return

    try:
        value = int(command.args)
    except ValueError:
        await message.answer("Нужно целое число от 64 до 8192, например: /tokens 800")
        return

    if not 64 <= value <= 8192:
        await message.answer("max_tokens должен быть в диапазоне 64..8192")
        return

    context.set_max_tokens(user_id, value)
    logger.info("user_id=%s установил max_tokens=%s", user_id, value)
    await message.answer(f"max_tokens = {value}")


@dp.message(Command("usage"))
async def cmd_usage(message: Message) -> None:
    user_id = message.from_user.id
    usage = context.last_usage(user_id)
    if not usage:
        await message.answer("Пока нет данных: задайте первый вопрос.")
        return

    titles = {
        "prompt_tokens": "вход (история + вопрос)",
        "completion_tokens": "выход (ответ)",
        "total_tokens": "всего",
        "prompt_cache_hit_tokens": "вход из кеша",
        "prompt_cache_miss_tokens": "вход без кеша",
    }
    lines = ["Последний ответ:"]
    for key, title in titles.items():
        if key in usage:
            lines.append(f"{title}: {usage[key]}")
    await message.answer("\n".join(lines))


@dp.message(F.text)
async def handle_text(message: Message) -> None:
    user_id = message.from_user.id
    text = (message.text or "").strip()
    if not text:
        return

    if text.lower() in CLEAR_PHRASES:
        await cmd_reset(message)
        return

    if text.startswith("/"):
        await message.answer("Неизвестная команда. Список команд: /help")
        return

    params = context.params(user_id)
    messages = context.build_messages(user_id, text)

    logger.info(
        "Запрос user_id=%s model=%s temperature=%s max_tokens=%s сообщений=%s ~%s токенов",
        user_id,
        settings.model,
        params.temperature,
        params.max_tokens,
        len(messages),
        estimate_messages_tokens(messages),
    )

    await message.bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)

    try:
        response = await client.chat(
            messages,
            temperature=params.temperature,
            max_tokens=params.max_tokens,
        )
    except LLMError as exc:
        logger.error("Ошибка LLM user_id=%s: %s", user_id, exc.details or exc)
        await message.answer(exc.user_message)
        return

    context.append_exchange(user_id, text, response.text)
    context.set_usage(user_id, response.usage)
    logger.info(
        "Ответ user_id=%s finish=%s usage=%s",
        user_id, response.finish_reason, response.usage,
    )

    if not response.text:
        await message.answer("Модель вернула пустой ответ. Попробуйте переформулировать.")
        return

    await message.answer(cut(response.text))


async def main() -> None:
    setup_logging()
    logger.info(
        "Запуск бота: model=%s base_url=%s thinking=%s",
        settings.model, settings.base_url, settings.thinking_mode,
    )
    bot = Bot(token=settings.bot_token, default=DefaultBotProperties())
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    try:
        settings.validate()
        asyncio.run(main())
    except ValueError as exc:
        print(f"Ошибка настроек: {exc}")
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен")