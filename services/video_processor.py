import asyncio
import shutil
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, List, Optional

from config import logger, TEMP_DIR
from services.ai_analyzer import AIAnalyzer, ClipProposal
from services.downloader import (
    DownloadedMedia,
    check_ffmpeg_installed,
    download_youtube_video,
    process_local_video,
)
from services.ffmpeg_editor import ProcessedClip, process_all_clips
from services.transcriber import transcribe_audio


class PipelineContext:
    """Manages the lifecycle and artifacts of an OpusClip processing session."""
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or uuid.uuid4().hex[:8]
        self.session_dir = TEMP_DIR / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.downloaded: Optional[DownloadedMedia] = None
        self.clips: List[ProcessedClip] = []

    def cleanup(self) -> None:
        """Cleans up all temporary files created for this session."""
        try:
            if self.session_dir.exists():
                shutil.rmtree(self.session_dir, ignore_errors=True)
                logger.info("Session %s temporary files removed.", self.session_id)
        except Exception as exc:
            logger.warning("Cleanup error for session %s: %s", self.session_id, exc)


async def execute_clipping_pipeline(
    source_type: str,
    source_data: Any,
    status_callback: Optional[Callable[[str], Awaitable[None]]] = None
) -> tuple[List[ProcessedClip], PipelineContext]:
    """
    Executes the monolithic AI Video Clipper pipeline:
    1. Downloads media (YouTube / Local file)
    2. Transcribes audio via Groq Cloud API
    3. Analyzes transcript with Gemini 2.5 Flash
    4. Slices & crops to vertical 9:16 with styled ASS subtitles via FFmpeg
    Returns:
        (processed_clips, context) - context should be cleaned up after clips are sent.
    """
    check_ffmpeg_installed()
    ctx = PipelineContext()

    async def notify(status_text: str):
        if status_callback:
            try:
                await status_callback(status_text)
            except Exception as e:
                logger.debug("Status notification callback failed: %s", e)

    try:
        # Step 1: Download video & extract audio
        await notify("📥 Скачивание видео с YouTube...")
        if source_type == "url":
            ctx.downloaded = await asyncio.to_thread(
                download_youtube_video, str(source_data), ctx.session_id
            )
        elif source_type == "local_path":
            ctx.downloaded = await asyncio.to_thread(
                process_local_video, Path(source_data), "Telegram Video", ctx.session_id
            )
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

        # Step 2: Instant speech transcription with Groq Whisper API
        await notify("⚡ Мгновенная расшифровка речи (Groq API)...")
        segments = await asyncio.to_thread(transcribe_audio, ctx.downloaded.audio_path)
        if not segments:
            raise ValueError("Не удалось распознать речь в видео (аудиодорожка пуста).")

        # Step 3: AI Analysis via Gemini 2.5 Flash
        await notify("🧠 Gemini ищет вирусные моменты...")
        analyzer = AIAnalyzer()
        proposals = await asyncio.to_thread(
            analyzer.analyze_transcript, segments, ctx.downloaded.duration
        )
        if not proposals:
            raise ValueError("AI не нашел подходящих фрагментов для нарезки.")

        # Step 4: Slicing 9:16 + burn subtitles via FFmpeg
        await notify("✂️ Нарезка видео 9:16 и наложение субтитров (FFmpeg)...")
        ctx.clips = await asyncio.to_thread(
            process_all_clips,
            ctx.downloaded.video_path,
            proposals,
            segments,
            ctx.session_dir
        )

        await notify("🚀 Отправка готовых Shorts!")
        return ctx.clips, ctx

    except Exception:
        # On error, clean up immediately
        ctx.cleanup()
        raise
