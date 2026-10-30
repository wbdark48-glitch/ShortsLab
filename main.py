import asyncio
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from config import BOT_TOKEN, logger, validate_config
from handlers.clipper import router as clipper_router
from services.downloader import FFmpegNotFoundError, check_ffmpeg_installed


async def main() -> None:
    """Application entry point."""
    logger.info("Initializing AI Video Clipper (OpusClip Clone) Bot...")

    # Validate required API keys and environment variables
    try:
        validate_config()
        logger.info("Configuration validated: BOT_TOKEN, GEMINI_API_KEY, GROQ_API_KEY found.")
    except Exception as exc:
        logger.critical("Configuration validation failed: %s", exc)
        return

    # Check FFmpeg dependency
    try:
        check_ffmpeg_installed()
        logger.info("FFmpeg binary detected in system PATH.")
    except FFmpegNotFoundError as exc:
        logger.error("FFmpeg check warning: %s", exc)

    # Initialize Bot and Dispatcher
    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )

    dp = Dispatcher()
    dp.include_router(clipper_router)

    logger.info("Bot starting polling updates...")
    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        await bot.session.close()
        logger.info("Bot session closed successfully.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped by user.")
