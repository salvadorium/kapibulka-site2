"""Telegram joke agent using Yandex AI Studio and PostgreSQL."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import psycopg
import requests
import telebot
from dotenv import load_dotenv
from telebot.types import InlineKeyboardButton, InlineKeyboardMarkup


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("kapibulka")


class YandexAPIError(RuntimeError):
    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(f"Yandex AI Studio returned HTTP {status_code}")


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value or value.startswith("replace_with_"):
        raise RuntimeError(f"Required environment variable {name} is missing or still a placeholder")
    return value


def yandex_request_config(base_url: str) -> tuple[str, bool]:
    """Return endpoint and whether the configured URL uses OpenAI-compatible /v1."""
    base = base_url.strip().rstrip("/")
    if base.endswith("/v1/chat/completions"):
        return base, True
    if base.endswith("/v1"):
        return base + "/chat/completions", True
    suffix = "/foundationModels/v1/completion"
    if base.endswith(suffix):
        return base, False
    return base + suffix, False


def database_connection() -> psycopg.Connection[Any]:
    return psycopg.connect(required_env("DATABASE_URL"), connect_timeout=10)


def initialize_database() -> None:
    with database_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS jokes (
                id BIGSERIAL PRIMARY KEY,
                telegram_user_id BIGINT NOT NULL,
                joke_text TEXT NOT NULL,
                category TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        # Additive migrations keep databases created by earlier bot versions usable.
        connection.execute("ALTER TABLE jokes ADD COLUMN IF NOT EXISTS topic TEXT NOT NULL DEFAULT ''")
        connection.execute(
            "ALTER TABLE jokes ADD COLUMN IF NOT EXISTS parent_joke_id BIGINT "
            "REFERENCES jokes(id) ON DELETE SET NULL"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id BIGSERIAL PRIMARY KEY,
                joke_id BIGINT NOT NULL REFERENCES jokes(id) ON DELETE CASCADE,
                telegram_user_id BIGINT NOT NULL,
                rating SMALLINT CHECK (rating IS NULL OR rating BETWEEN 1 AND 5),
                comment TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (joke_id, telegram_user_id)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS jokes_user_created_idx "
            "ON jokes (telegram_user_id, created_at DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS feedback_user_created_idx "
            "ON feedback (telegram_user_id, created_at DESC)"
        )


def recent_feedback(telegram_user_id: int) -> list[tuple[int | None, str | None]]:
    with database_connection() as connection:
        rows = connection.execute(
            """
            SELECT rating, comment
            FROM feedback
            WHERE telegram_user_id = %s
            ORDER BY created_at DESC
            LIMIT 12
            """,
            (telegram_user_id,),
        ).fetchall()
    return [(row[0], row[1]) for row in rows]


def latest_joke(telegram_user_id: int) -> tuple[int, str, str, str] | None:
    with database_connection() as connection:
        row = connection.execute(
            """
            SELECT id, joke_text, category, topic FROM jokes
            WHERE telegram_user_id = %s
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (telegram_user_id,),
        ).fetchone()
    return (int(row[0]), str(row[1]), str(row[2]), str(row[3] or "")) if row else None


def parse_generated_joke(generated_text: str) -> tuple[str, str]:
    candidate = generated_text.strip()
    if not candidate.startswith("{"):
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if match:
            candidate = match.group(0)
    try:
        parsed = json.loads(candidate)
        joke_text = str(parsed.get("joke", "")).strip()
        category = str(parsed.get("category", "Другое")).strip()[:80] or "Другое"
        if joke_text:
            return joke_text, category
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    logger.warning("Yandex response was not in the expected JSON shape; using its text as the joke")
    return generated_text.strip(), "Другое"


def generate_joke(
    telegram_user_id: int,
    topic: str = "",
    previous_joke: str = "",
) -> tuple[str, str]:
    api_key = required_env("YANDEX_API_KEY")
    folder_id = required_env("YANDEX_FOLDER_ID")
    model_uri = required_env("YANDEX_MODEL_URI")
    base_url = required_env("YANDEX_BASE_URL")
    endpoint, openai_compatible = yandex_request_config(base_url)

    feedback = recent_feedback(telegram_user_id)
    feedback_context = "\n".join(
        f"Оценка: {rating if rating is not None else 'нет'}; комментарий: {comment or 'нет'}"
        for rating, comment in feedback
    ) or "Пока отзывов нет."
    system_text = (
        "Ты придумываешь короткие доброжелательные анекдоты на русском языке. "
        "Учитывай отзывы этого пользователя: избегай того, что ему не понравилось, "
        "и развивай то, что получило высокую оценку. Верни только JSON с полями joke и category.\n"
        "Отзывы пользователя:\n" + feedback_context
    )
    user_text = "Придумай один короткий оригинальный анекдот."
    if topic:
        user_text += f" Тема: {topic[:300]}"
    if previous_joke:
        user_text += (
            " Это новый вариант по той же теме с учётом комментария пользователя. "
            "Не повторяй предыдущий анекдот: " + previous_joke[:1000]
        )

    if openai_compatible:
        payload = {
            "model": model_uri,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            "temperature": 0.8,
            "max_tokens": 400,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        folder_header = "OpenAI-Project"
    else:
        payload = {
            "modelUri": model_uri,
            "completionOptions": {"stream": False, "temperature": 0.8, "maxTokens": "400"},
            "messages": [
                {"role": "system", "text": system_text},
                {"role": "user", "text": user_text},
            ],
            "jsonObject": True,
        }
        folder_header = "x-folder-id"

    try:
        response = requests.post(
            endpoint,
            headers={
                "Authorization": f"Api-Key {api_key}",
                folder_header: folder_id,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
    except requests.RequestException as exc:
        logger.error("Could not reach Yandex AI Studio (%s)", type(exc).__name__)
        raise RuntimeError("Yandex AI Studio is unreachable or timed out") from exc

    if not response.ok:
        request_id = response.headers.get("x-request-id", "not provided")
        logger.error("Yandex AI Studio HTTP %s; request_id=%s", response.status_code, request_id)
        raise YandexAPIError(response.status_code)

    data = response.json()
    if openai_compatible:
        choices = data.get("choices", [])
        generated_text = choices[0].get("message", {}).get("content", "") if choices else ""
    else:
        alternatives = data.get("result", {}).get("alternatives", [])
        generated_text = alternatives[0].get("message", {}).get("text", "") if alternatives else ""

    if isinstance(generated_text, list):
        generated_text = "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in generated_text
        )
    generated_text = str(generated_text).strip()
    if not generated_text:
        logger.error("Yandex AI Studio returned a successful response without generated text")
        raise RuntimeError("Yandex AI Studio returned no generated text")
    return parse_generated_joke(generated_text)


def save_joke(
    telegram_user_id: int,
    joke_text: str,
    category: str,
    topic: str,
    parent_joke_id: int | None = None,
) -> int:
    with database_connection() as connection:
        row = connection.execute(
            """
            INSERT INTO jokes (telegram_user_id, joke_text, category, topic, parent_joke_id)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (telegram_user_id, joke_text, category, topic[:300], parent_joke_id),
        ).fetchone()
    if row is None:
        raise RuntimeError("Database did not return a saved joke ID")
    return int(row[0])


def save_feedback(
    telegram_user_id: int,
    joke_id: int,
    rating: int | None = None,
    comment: str | None = None,
) -> None:
    with database_connection() as connection:
        connection.execute(
            """
            INSERT INTO feedback (joke_id, telegram_user_id, rating, comment)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (joke_id, telegram_user_id) DO UPDATE SET
                rating = COALESCE(EXCLUDED.rating, feedback.rating),
                comment = COALESCE(EXCLUDED.comment, feedback.comment),
                created_at = NOW()
            """,
            (joke_id, telegram_user_id, rating, comment),
        )


def rating_keyboard(joke_id: int) -> InlineKeyboardMarkup:
    keyboard = InlineKeyboardMarkup(row_width=5)
    keyboard.add(
        *[
            InlineKeyboardButton(str(score), callback_data=f"rate:{joke_id}:{score}")
            for score in range(1, 6)
        ]
    )
    return keyboard


def user_id_from_message(message: telebot.types.Message) -> int:
    if message.from_user is None:
        raise RuntimeError("Telegram message has no sender")
    return message.from_user.id


def main() -> None:
    load_dotenv()
    token = required_env("TELEGRAM_BOT_TOKEN")
    required_env("YANDEX_API_KEY")
    required_env("YANDEX_FOLDER_ID")
    required_env("YANDEX_MODEL_URI")
    required_env("YANDEX_BASE_URL")
    required_env("DATABASE_URL")

    initialize_database()
    bot = telebot.TeleBot(token, threaded=True)
    webhook = bot.get_webhook_info()
    if webhook.url:
        raise RuntimeError("Telegram webhook is configured. Remove it before starting long polling.")

    def send_generated(
        message: telebot.types.Message,
        topic: str,
        previous: tuple[int, str, str, str] | None = None,
    ) -> None:
        user_id = user_id_from_message(message)
        bot.send_chat_action(message.chat.id, "typing")
        try:
            joke_text, category = generate_joke(
                user_id,
                topic,
                previous_joke=previous[1] if previous else "",
            )
            joke_id = save_joke(
                user_id,
                joke_text,
                category,
                topic,
                parent_joke_id=previous[0] if previous else None,
            )
            header = "🎭 Новый вариант:" if previous else "🎭 Готово:"
            bot.send_message(
                message.chat.id,
                f"{header}\n\n— {joke_text}\n\n"
                f"Сохранено в PostgreSQL. Напиши комментарий, чтобы переделать, "
                f"или /new новая тема.\nКатегория: {category}",
                reply_markup=rating_keyboard(joke_id),
            )
        except YandexAPIError as exc:
            logger.exception("Joke generation failed at Yandex AI Studio")
            if exc.status_code in (401, 403):
                text = "Yandex AI Studio отклонил доступ. Проверьте API-ключ и права сервисного аккаунта."
            elif exc.status_code == 429:
                text = "У Yandex AI Studio закончилась квота или превышен лимит. Проверьте квоты и попробуйте позже."
            elif exc.status_code == 400:
                text = "Yandex AI Studio отклонил запрос. Проверьте URI модели, endpoint и формат запроса в Railway Logs."
            else:
                text = f"Yandex AI Studio вернул ошибку HTTP {exc.status_code}. Проверьте Railway Logs."
            bot.reply_to(message, text)
        except Exception:
            logger.exception("Joke generation or database save failed")
            bot.reply_to(message, "Не получилось подготовить анекдот. Проверьте Railway Logs и попробуйте позже.")

    @bot.message_handler(commands=["start", "help"])
    def show_help(message: telebot.types.Message) -> None:
        bot.reply_to(
            message,
            "Привет! Я придумываю короткие анекдоты на русском.\n\n"
            "1. Пришли тему: «офис», «космос», «кот-программист».\n"
            "2. После шутки пришли комментарий: «смешнее», «страннее» — сделаю новый вариант.\n"
            "3. Для новой темы используй /new тема.\n\n"
            "Каждый готовый анекдот сохраняется в PostgreSQL. /last покажет последний вариант.\n"
            "/joke тема — тоже создаёт анекдот; /rate 1-5 — оценка; /help — справка.",
        )

    @bot.message_handler(commands=["new"])
    def new_topic(message: telebot.types.Message) -> None:
        topic = (message.text or "").partition(" ")[2].strip()
        if not topic:
            bot.reply_to(message, "Укажите тему: /new космос")
            return
        send_generated(message, topic)

    @bot.message_handler(commands=["last"])
    def show_last(message: telebot.types.Message) -> None:
        last = latest_joke(user_id_from_message(message))
        if last is None:
            bot.reply_to(message, "Пока нет сохранённых анекдотов. Пришли тему, например: Яндекс.")
            return
        joke_id, joke_text, category, topic = last
        bot.send_message(
            message.chat.id,
            f"Последний сохранённый анекдот:\n\n— {joke_text}\n\n"
            f"Тема: {topic or 'без темы'}\nКатегория: {category}",
            reply_markup=rating_keyboard(joke_id),
        )

    @bot.message_handler(commands=["joke"])
    def slash_joke(message: telebot.types.Message) -> None:
        topic = (message.text or "").partition(" ")[2].strip()
        send_generated(message, topic)

    @bot.message_handler(commands=["rate"])
    def rate_latest_joke(message: telebot.types.Message) -> None:
        user_id = user_id_from_message(message)
        parts = (message.text or "").split(maxsplit=1)
        if len(parts) != 2 or not parts[1].isdigit() or not 1 <= int(parts[1]) <= 5:
            bot.reply_to(message, "Используйте формат: /rate 1-5")
            return
        last = latest_joke(user_id)
        if last is None:
            bot.reply_to(message, "Сначала пришлите тему для анекдота.")
            return
        save_feedback(user_id, last[0], rating=int(parts[1]))
        bot.reply_to(message, "Спасибо за оценку! Учту её в следующих анекдотах.")

    def revise_with_comment(message: telebot.types.Message, comment: str) -> None:
        user_id = user_id_from_message(message)
        previous = latest_joke(user_id)
        if previous is None:
            send_generated(message, comment)
            return
        save_feedback(user_id, previous[0], comment=comment[:2000])
        send_generated(message, previous[3], previous=previous)

    @bot.message_handler(commands=["feedback"])
    def slash_feedback(message: telebot.types.Message) -> None:
        comment = (message.text or "").partition(" ")[2].strip()
        if not comment:
            bot.reply_to(message, "Напишите комментарий после команды: /feedback смешнее")
            return
        revise_with_comment(message, comment)

    @bot.callback_query_handler(func=lambda call: bool(call.data and call.data.startswith("rate:")))
    def rate_from_button(call: telebot.types.CallbackQuery) -> None:
        try:
            _, joke_id_text, rating_text = (call.data or "").split(":", 2)
            joke_id = int(joke_id_text)
            rating = int(rating_text)
            user_id = call.from_user.id
            if rating < 1 or rating > 5:
                raise ValueError("Invalid rating")
            with database_connection() as connection:
                owned_joke = connection.execute(
                    "SELECT 1 FROM jokes WHERE id = %s AND telegram_user_id = %s",
                    (joke_id, user_id),
                ).fetchone()
            if not owned_joke:
                bot.answer_callback_query(call.id, "Этот анекдот не найден.", show_alert=True)
                return
            save_feedback(user_id, joke_id, rating=rating)
            bot.answer_callback_query(call.id, "Спасибо за оценку!")
            if call.message:
                bot.send_message(call.message.chat.id, "Спасибо! Учту оценку в следующих анекдотах.")
        except Exception:
            logger.exception("Could not save rating from button")
            bot.answer_callback_query(call.id, "Не удалось сохранить оценку. Проверьте Railway Logs.")

    @bot.message_handler(func=lambda message: bool(message.text and not message.text.startswith("/")))
    def plain_text(message: telebot.types.Message) -> None:
        text = (message.text or "").strip()
        if not text:
            return
        revise_with_comment(message, text)

    logger.info("Starting Telegram long polling")
    bot.infinity_polling(
        timeout=30,
        long_polling_timeout=30,
        skip_pending=False,
        allowed_updates=["message", "callback_query"],
    )


if __name__ == "__main__":
    load_dotenv()
    main()
