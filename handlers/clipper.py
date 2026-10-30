import asyncio
import re
from pathlib import Path

from aiogram import Bot, F, Router, types
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import FSInputFile, Message

from config import logger, TEMP_DIR
from services.ai_analyzer import AIAnalyzerError, AIQuotaExceededError
from services.downloader import (
    DownloaderError,
    FFmpegNotFoundError,
    VideoTooLongError,
)
from services.ffmpeg_editor import VideoEditorError
from services.transcriber import TranscriberError
from services.video_processor import execute_clipping_pipeline

router = Router(name="clipper_router")

# Regex to detect YouTube & web URLs
URL_REGEX = re.compile(
    r"(https?://(?:www\.|m\.)?(?:youtube\.com|youtu\.be|instagram\.com|tiktok\.com|twitter\.com|x\.com)/[^\s]+|"
    r"https?://[^\s]+)"
)


@router.message(CommandStart())
async def cmd_start(message: Message):
    """Handles the /start command."""
    welcome_text = (
        "👋 **Привет! Я AI Video Clipper (OpusClip Clone).**\n\n"
        "Я превращаю длинные видео в готовые вирусные вертикальные Shorts, Reels и TikTok (формат 9:16) со стильными субтитрами!\n\n"
        "⚡ **Технологический стек:**\n"
        "• ⚡ **Groq Cloud API** (Whisper Large V3) — сверхбыстрая расшифровка\n"
        "• 🧠 **Gemini 2.5 Flash** — точный поиск вирусных хуков\n"
        "• ✂️ **FFmpeg** — умное кадрирование 9:16 и стилизованные субтитры\n\n"
        "🚀 **Как использовать?**\n"
        "Отправьте мне **ссылку на YouTube видео** или **видеофайл MP4**!"
    )
    await message.answer(welcome_text, parse_mode=ParseMode.MARKDOWN)


@router.message(Command("help"))
async def cmd_help(message: Message):
    """Handles the /help command."""
    help_text = (
        "📖 **Инструкция по использованию:**\n\n"
        "1. **Ссылка на YouTube:** Отправьте ссылку в чат (например: `https://youtu.be/...`).\n"
        "2. **Прямой файл:** Прикрепите MP4 видео (до 50 МБ через Telegram).\n"
        "3. **Длительность:** Видео до 30 минут обрабатываются максимально эффективно.\n\n"
        "✨ Бот автоматически нарежет ТОП-3 лучших момента, кадрирует их в 9:16 и наложит динамические субтитры."
    )
    await message.answer(help_text, parse_mode=ParseMode.MARKDOWN)


async def handle_video_job(
    message: Message,
    bot: Bot,
    source_type: str,
    source_data: str | Path
):
    """Processes video clipping job with real-time status updates and exception handling."""
    status_msg = await message.answer("⏳ Инициализация обработки видео...")
    pipeline_ctx = None

    async def update_status(text: str):
        try:
            await status_msg.edit_text(text)
        except Exception:
            pass

    try:
        clips, pipeline_ctx = await execute_clipping_pipeline(
            source_type=source_type,
            source_data=source_data,
            status_callback=update_status
        )

        # Send all clips to user
        for idx, clip in enumerate(clips, start=1):
            caption = (
                f"🎬 **Клип #{idx}: {clip.title}**\n\n"
                f"🔥 **Оценка вирусности:** {clip.virality_score}/10\n"
                f"⏱ **Таймкод:** {clip.start_time:.1f}с — {clip.end_time:.1f}с (Длительность: {clip.duration:.1f}с)\n"
            )
            if clip.reason:
                caption += f"💡 **Почему выбран:** {clip.reason}\n"

            video_input = FSInputFile(str(clip.clip_path))
            await message.answer_video(
                video=video_input,
                caption=caption,
                parse_mode=ParseMode.MARKDOWN,
                supports_streaming=True,
                width=1080,
                height=1920
            )

        try:
            await status_msg.delete()
        except Exception:
            pass

    except VideoTooLongError as vte:
        logger.warning("Video exceeds duration: %s", vte)
        await status_msg.edit_text(f"⏱ **Ошибка:** {str(vte)}")
    except FFmpegNotFoundError as ffe:
        logger.error("FFmpeg not found: %s", ffe)
        await status_msg.edit_text(
            "❌ **Ошибка системы:** FFmpeg не найден на сервере.\n"
            "Пожалуйста, убедитесь, что FFmpeg установлен и добавлен в PATH."
        )
    except AIQuotaExceededError as qe:
        logger.warning("Quota exceeded: %s", qe)
        await status_msg.edit_text(f"🛑 **Лимит запросов:** {str(qe)}")
    except (DownloaderError, TranscriberError, AIAnalyzerError, VideoEditorError, ValueError) as known_err:
        logger.error("Known pipeline error: %s", known_err)
        await status_msg.edit_text(f"⚠️ **Ошибка обработки:** {str(known_err)}")
    except Exception as exc:
        logger.error("Unhandled pipeline error: %s", exc, exc_info=True)
        await status_msg.edit_text(
            "💥 Произошла непредвиденная ошибка при обработке видео.\n"
            "Пожалуйста, проверьте ссылку или формат файла и попробуйте снова."
        )
    finally:
        if pipeline_ctx:
            pipeline_ctx.cleanup()


@router.message(F.video)
async def handle_video_upload(message: Message, bot: Bot):
    """Handles direct video upload in chat."""
    video = message.video
    if not video:
        return

    if video.file_size and video.file_size > 50 * 1024 * 1024:
        await message.answer(
            "⚠️ Видеофайл превышает 50 МБ (лимит Telegram Bot API).\n"
            "Пожалуйста, загрузите видео на YouTube и отправьте ссылку!"
        )
        return

    status_msg = await message.answer("📥 Загружаю видео из Telegram...")
    try:
        tg_file = await bot.get_file(video.file_id)
        if not tg_file.file_path:
            await status_msg.edit_text("❌ Не удалось получить путь к файлу из Telegram.")
            return

        temp_video_path = TEMP_DIR / f"tg_upload_{video.file_id[:10]}.mp4"
        await bot.download_file(tg_file.file_path, destination=temp_video_path)
        await status_msg.delete()

        await handle_video_job(
            message=message,
            bot=bot,
            source_type="local_path",
            source_data=temp_video_path
        )
    except Exception as exc:
        logger.error("Error downloading telegram video: %s", exc)
        await status_msg.edit_text("❌ Ошибка при загрузке видеофайла из Telegram.")


@router.message(F.document)
async def handle_document_upload(message: Message, bot: Bot):
    """Handles video files sent as documents."""
    doc = message.document
    if not doc:
        return

    mime = doc.mime_type or ""
    filename = (doc.file_name or "").lower()
    if not (mime.startswith("video/") or filename.endswith((".mp4", ".mov", ".mkv", ".webm"))):
        return

    if doc.file_size and doc.file_size > 50 * 1024 * 1024:
        await message.answer(
            "⚠️ Документ превышает 50 МБ.\n"
            "Рекомендуется загрузить видео на YouTube и прислать ссылку."
        )
        return

    status_msg = await message.answer("📥 Загружаю видео из Telegram...")
    try:
        tg_file = await bot.get_file(doc.file_id)
        if not tg_file.file_path:
            await status_msg.edit_text("❌ Не удалось получить путь к файлу из Telegram.")
            return

        temp_video_path = TEMP_DIR / f"tg_upload_{doc.file_id[:10]}.mp4"
        await bot.download_file(tg_file.file_path, destination=temp_video_path)
        await status_msg.delete()

        await handle_video_job(
            message=message,
            bot=bot,
            source_type="local_path",
            source_data=temp_video_path
        )
    except Exception as exc:
        logger.error("Error downloading telegram document: %s", exc)
        await status_msg.edit_text("❌ Ошибка при загрузке документа из Telegram.")


@router.message(F.text)
async def handle_text_url(message: Message, bot: Bot):
    """Handles text message containing video URL."""
    text = (message.text or "").strip()
    match = URL_REGEX.search(text)
    if not match:
        await message.answer(
            "ℹ️ Пожалуйста, отправьте ссылку на YouTube видео или прикрепите видеофайл MP4."
        )
        return

    url = match.group(0).strip()
    await handle_video_job(
        message=message,
        bot=bot,
        source_type="url",
        source_data=url
    )
