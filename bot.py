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


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value or value.startswith("replace_with_"):
        raise RuntimeError(f"Required environment variable {name} is missing or still a placeholder")
    return value


def yandex_completion_url(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    suffix = "/foundationModels/v1/completion"
    if base.endswith(suffix):
        return base
    return base + suffix


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


def generate_joke(telegram_user_id: int, topic: str = "") -> tuple[str, str]:
    api_key = required_env("YANDEX_API_KEY")
    folder_id = required_env("YANDEX_FOLDER_ID")
    model_uri = required_env("YANDEX_MODEL_URI")
    base_url = required_env("YANDEX_BASE_URL")

    feedback = recent_feedback(telegram_user_id)
    feedback_lines = [
        f"Оценка: {rating if rating is not None else 'нет'}; комментарий: {comment or 'нет'}"
        for rating, comment in feedback
    ]
    feedback_context = "\n".join(feedback_lines) or "Пока отзывов нет."
    user_request = "Придумай один короткий оригинальный анекдот на русском языке."
    if topic:
        user_request += f" Тема: {topic[:300]}"
    user_request += (
        " Верни только JSON-объект с полями joke (текст анекдота) и category "
        "(короткая категория юмора)."
    )

    payload = {
        "modelUri": model_uri,
        "completionOptions": {"stream": False, "temperature": 0.8, "maxTokens": "400"},
        "messages": [
            {
                "role": "system",
                "text": (
                    "Ты пишешь короткие доброжелательные анекдоты на русском. "
                    "Учитывай прошлые отзывы этого пользователя: избегай того, что ему не понравилось, "
                    "и развивай то, что получило высокую оценку. Не повторяй старые шутки. "
                    "Отзывы пользователя:\n" + feedback_context
                ),
            },
            {"role": "user", "text": user_request},
        ],
        "jsonObject": True,
    }
    response = requests.post(
        yandex_completion_url(base_url),
        headers={
            "Authorization": f"Api-Key {api_key}",
            "x-folder-id": folder_id,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    if not response.ok:
        logger.error("Yandex AI Studio returned HTTP %s", response.status_code)
        raise RuntimeError("Yandex AI Studio could not generate a joke")

    data = response.json()
    alternatives = data.get("result", {}).get("alternatives", [])
    if not alternatives:
        raise RuntimeError("Yandex AI Studio returned no generated text")
    generated_text = alternatives[0].get("message", {}).get("text", "").strip()
    if not generated_text:
        raise RuntimeError("Yandex AI Studio returned an empty joke")

    try:
        candidate = generated_text
        if not candidate.startswith("{"):
            match = re.search(r"\{.*\}", candidate, re.DOTALL)
            if match:
                candidate = match.group(0)
        parsed = json.loads(candidate)
        joke_text = str(parsed.get("joke", "")).strip()
        category = str(parsed.get("category", "Другое")).strip()[:80] or "Другое"
        if not joke_text:
            raise ValueError("Missing joke field")
        return joke_text, category
    except (json.JSONDecodeError, TypeError, ValueError):
        logger.warning("Model returned text outside the expected JSON shape")
        return generated_text, "Другое"


def save_joke(telegram_user_id: int, joke_text: str, category: str) -> int:
    with database_connection() as connection:
        row = connection.execute(
            """
            INSERT INTO jokes (telegram_user_id, joke_text, category)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (telegram_user_id, joke_text, category),
        ).fetchone()
    if row is None:
        raise RuntimeError("Database did not return a saved joke ID")
    return int(row[0])


def latest_joke_id(telegram_user_id: int) -> int | None:
    with database_connection() as connection:
        row = connection.execute(
            """
            SELECT id FROM jokes
            WHERE telegram_user_id = %s
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (telegram_user_id,),
        ).fetchone()
    return int(row[0]) if row else None


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
    # Validate the model settings at startup so configuration errors appear in Railway logs.
    required_env("YANDEX_API_KEY")
    required_env("YANDEX_FOLDER_ID")
    required_env("YANDEX_MODEL_URI")
    required_env("YANDEX_BASE_URL")
    required_env("DATABASE_URL")

    initialize_database()
    bot = telebot.TeleBot(token, threaded=True)
    webhook = bot.get_webhook_info()
    if webhook.url:
        raise RuntimeError(
            "Telegram webhook is configured. Remove it before starting long polling."
        )

    @bot.message_handler(commands=["start", "help"])
    def show_help(message: telebot.types.Message) -> None:
        bot.reply_to(
            message,
            "Привет! Я придумываю анекдоты и учитываю ваши отзывы.\n\n"
            "/joke [тема] — получить анекдот\n"
            "/rate 1-5 — оценить последний анекдот\n"
            "/feedback текст — оставить комментарий к последнему анекдоту\n"
            "/help — показать эту справку",
        )

    @bot.message_handler(commands=["joke"])
    def send_joke(message: telebot.types.Message) -> None:
        user_id = user_id_from_message(message)
        topic = (message.text or "").partition(" ")[2].strip()
        bot.send_chat_action(message.chat.id, "typing")
        try:
            joke_text, category = generate_joke(user_id, topic)
            joke_id = save_joke(user_id, joke_text, category)
            bot.send_message(
                message.chat.id,
                f"{joke_text}\n\nКатегория: {category}\n\nОцените шутку от 1 до 5:",
                reply_markup=rating_keyboard(joke_id),
            )
        except Exception:
            logger.exception("Could not generate or save a joke")
            bot.reply_to(message, "Не получилось подготовить анекдот. Попробуйте чуть позже.")

    @bot.message_handler(commands=["rate"])
    def rate_latest_joke(message: telebot.types.Message) -> None:
        user_id = user_id_from_message(message)
        parts = (message.text or "").split(maxsplit=1)
        if len(parts) != 2 or not parts[1].isdigit() or not 1 <= int(parts[1]) <= 5:
            bot.reply_to(message, "Используйте формат: /rate 1-5")
            return
        joke_id = latest_joke_id(user_id)
        if joke_id is None:
            bot.reply_to(message, "Сначала запросите анекдот командой /joke.")
            return
        save_feedback(user_id, joke_id, rating=int(parts[1]))
        bot.reply_to(message, "Спасибо за оценку! Учту её в следующих анекдотах.")

    @bot.message_handler(commands=["feedback"])
    def comment_on_latest_joke(message: telebot.types.Message) -> None:
        user_id = user_id_from_message(message)
        comment = (message.text or "").partition(" ")[2].strip()
        if not comment:
            bot.reply_to(message, "Напишите комментарий после команды: /feedback ваш отзыв")
            return
        joke_id = latest_joke_id(user_id)
        if joke_id is None:
            bot.reply_to(message, "Сначала запросите анекдот командой /joke.")
            return
        save_feedback(user_id, joke_id, comment=comment[:2000])
        bot.reply_to(message, "Спасибо за отзыв! Учту его в следующих анекдотах.")

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
            bot.answer_callback_query(call.id, "Не удалось сохранить оценку. Попробуйте ещё раз.")

    logger.info("Starting Telegram long polling")
    bot.infinity_polling(
        timeout=30,
        long_polling_timeout=30,
        skip_pending=False,
        allowed_updates=["message", "callback_query"],
    )


if __name__ == "__main__":
    main()
