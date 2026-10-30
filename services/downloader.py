import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yt_dlp

from config import logger, TEMP_DIR, MAX_VIDEO_DURATION_SECONDS


class DownloaderError(Exception):
    """Base exception for downloader failures."""
    pass


class VideoTooLongError(DownloaderError):
    """Raised when video exceeds maximum permitted duration."""
    pass


class FFmpegNotFoundError(DownloaderError):
    """Raised when FFmpeg binary is missing from PATH."""
    pass


@dataclass
class DownloadedMedia:
    video_path: Path
    audio_path: Path
    duration: float
    title: str
    session_id: str


def check_ffmpeg_installed() -> None:
    """Verifies that ffmpeg executable is available in PATH."""
    if not shutil.which("ffmpeg"):
        raise FFmpegNotFoundError(
            "FFmpeg is not installed or not found in system PATH. "
            "Please install FFmpeg to process videos."
        )


def extract_audio(video_path: Path, output_audio_path: Path) -> Path:
    """
    Extracts lightweight 16kHz mono MP3 audio from video file for Groq Whisper.
    MP3 32kbps mono ensures 1 hour of audio is only ~14 MB, well below the 25MB Groq API limit.
    """
    check_ffmpeg_installed()
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "libmp3lame",
        "-ar", "16000",
        "-ac", "1",
        "-b:a", "32k",
        str(output_audio_path)
    ]

    logger.info("Extracting compressed audio from %s to %s", video_path.name, output_audio_path.name)
    process = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if process.returncode != 0:
        logger.error("FFmpeg audio extraction failed: %s", process.stderr)
        raise DownloaderError(f"Failed to extract audio from video: {process.stderr[-300:]}")

    if not output_audio_path.exists() or output_audio_path.stat().st_size == 0:
        raise DownloaderError("Extracted audio file is empty or does not exist.")

    return output_audio_path


def get_media_duration_ffprobe(file_path: Path) -> float:
    """Uses ffprobe to obtain media duration in seconds."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0

    cmd = [
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path)
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return float(res.stdout.strip())
    except Exception as exc:
        logger.warning("Could not probe duration via ffprobe: %s", exc)
        return 0.0


def download_youtube_video(url: str, session_id: Optional[str] = None) -> DownloadedMedia:
    """
    Downloads YouTube (or supported web) video via yt-dlp, validates duration,
    and extracts lightweight audio.
    """
    check_ffmpeg_installed()
    session_id = session_id or uuid.uuid4().hex[:8]
    session_dir = TEMP_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    video_template = str(session_dir / "source_video.%(ext)s")

    ydl_opts = {
        "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best[height<=1080]/best",
        "outtmpl": video_template,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    logger.info("Fetching video info for: %s", url)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                raise DownloaderError("Failed to extract video information from URL.")

            duration = float(info.get("duration") or 0.0)
            title = info.get("title", "YouTube Video")

            if duration > MAX_VIDEO_DURATION_SECONDS:
                max_min = MAX_VIDEO_DURATION_SECONDS // 60
                curr_min = int(duration // 60)
                raise VideoTooLongError(
                    f"Видео слишком длинное ({curr_min} мин.). Максимальная длительность: {max_min} мин."
                )

            logger.info("Downloading video '%s' (duration: %.1fs)...", title, duration)
            ydl.download([url])

    except VideoTooLongError:
        raise
    except yt_dlp.utils.DownloadError as exc:
        logger.error("yt-dlp download error: %s", exc)
        raise DownloaderError(f"Ошибка скачивания видео: {str(exc)}")
    except Exception as exc:
        logger.error("Unexpected download error: %s", exc)
        raise DownloaderError(f"Не удалось загрузить видео: {str(exc)}")

    # Locate downloaded file
    video_path = session_dir / "source_video.mp4"
    if not video_path.exists():
        matches = list(session_dir.glob("source_video.*"))
        if matches:
            video_path = matches[0]
        else:
            raise DownloaderError("Downloaded video file was not found on disk.")

    if duration <= 0:
        duration = get_media_duration_ffprobe(video_path)
        if duration > MAX_VIDEO_DURATION_SECONDS:
            raise VideoTooLongError(
                f"Видео слишком длинное ({int(duration // 60)} мин.). Максимально: {MAX_VIDEO_DURATION_SECONDS // 60} мин."
            )

    audio_path = session_dir / "audio.mp3"
    extract_audio(video_path, audio_path)

    return DownloadedMedia(
        video_path=video_path,
        audio_path=audio_path,
        duration=duration,
        title=title,
        session_id=session_id
    )


def process_local_video(local_video_path: Path, title: str = "Telegram Video", session_id: Optional[str] = None) -> DownloadedMedia:
    """
    Validates a locally received video (e.g. from Telegram message),
    checks duration, and extracts audio.
    """
    check_ffmpeg_installed()
    session_id = session_id or uuid.uuid4().hex[:8]
    session_dir = TEMP_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    target_video_path = session_dir / "source_video.mp4"
    if local_video_path.resolve() != target_video_path.resolve():
        shutil.copyfile(local_video_path, target_video_path)

    duration = get_media_duration_ffprobe(target_video_path)
    if duration > MAX_VIDEO_DURATION_SECONDS:
        max_min = MAX_VIDEO_DURATION_SECONDS // 60
        curr_min = int(duration // 60)
        raise VideoTooLongError(
            f"Видео слишком длинное ({curr_min} мин.). Максимальная длительность: {max_min} мин."
        )

    audio_path = session_dir / "audio.mp3"
    extract_audio(target_video_path, audio_path)

    return DownloadedMedia(
        video_path=target_video_path,
        audio_path=audio_path,
        duration=duration,
        title=title,
        session_id=session_id
    )
