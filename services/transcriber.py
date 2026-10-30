import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from groq import Groq

from config import GROQ_API_KEY, logger

# Groq API has a strict 25 MB file limit. We set a safe threshold of 24 MB.
MAX_GROQ_FILE_SIZE_BYTES = 24 * 1024 * 1024
CHUNK_DURATION_SECONDS = 1200  # 20 minutes per audio chunk


class TranscriberError(Exception):
    """Exception raised for errors in transcription process."""
    pass


def get_audio_duration(audio_path: Path) -> float:
    """Uses ffprobe to obtain audio duration in seconds."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0

    cmd = [
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path)
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return float(res.stdout.strip())
    except Exception as exc:
        logger.warning("Could not probe audio duration: %s", exc)
        return 0.0


def compress_audio(input_path: Path, output_path: Path) -> Path:
    """
    Compresses any audio file to 16kHz mono MP3 at 32kbps for optimal Groq Whisper ingestion.
    Produces ~14.4 MB per hour of audio.
    """
    if not shutil.which("ffmpeg"):
        raise TranscriberError("FFmpeg executable not found in system PATH.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-i", str(input_path),
        "-vn",
        "-acodec", "libmp3lame",
        "-ar", "16000",
        "-ac", "1",
        "-b:a", "32k",
        str(output_path)
    ]

    logger.info("Compressing audio %s -> %s (16kHz mono 32kbps)...", input_path.name, output_path.name)
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        logger.error("Audio compression failed: %s", res.stderr)
        raise TranscriberError(f"Ошибка сжатия аудио перед отправкой в Groq: {res.stderr[-300:]}")

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise TranscriberError("Сжатый аудиофайл пуст или не был создан.")

    return output_path


def split_audio_into_chunks(audio_path: Path, total_duration: float, chunk_dir: Path) -> List[tuple[Path, float]]:
    """
    Splits long audio (>24MB) into 20-minute chunks for sequential transcription.
    Returns list of tuples: (chunk_file_path, start_offset_seconds).
    """
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunks: List[tuple[Path, float]] = []

    if total_duration <= 0:
        total_duration = get_audio_duration(audio_path)

    num_chunks = math.ceil(total_duration / CHUNK_DURATION_SECONDS)
    logger.info(
        "Splitting audio (%.1fs) into %d chunks of up to %ds...",
        total_duration, num_chunks, CHUNK_DURATION_SECONDS
    )

    for i in range(num_chunks):
        start_sec = i * CHUNK_DURATION_SECONDS
        chunk_file = chunk_dir / f"chunk_{i:03d}.mp3"
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-ss", f"{start_sec:.2f}",
            "-i", str(audio_path),
            "-t", f"{CHUNK_DURATION_SECONDS}",
            "-vn",
            "-acodec", "libmp3lame",
            "-ar", "16000",
            "-ac", "1",
            "-b:a", "32k",
            str(chunk_file)
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0 or not chunk_file.exists():
            logger.error("Failed creating audio chunk %d: %s", i, res.stderr)
            raise TranscriberError(f"Ошибка разделения длинного аудио на части: {res.stderr[-200:]}")

        chunks.append((chunk_file, start_sec))

    return chunks


def _transcribe_single_file(client: Groq, file_path: Path) -> List[Dict[str, Any]]:
    """
    Sends a single audio file (<= 24MB) to Groq Whisper API and parses the returned segments.
    """
    logger.info("Sending %s (%.2f MB) to Groq API (whisper-large-v3)...", file_path.name, file_path.stat().st_size / (1024 * 1024))
    with open(file_path, "rb") as file_obj:
        transcription = client.audio.transcriptions.create(
            file=(file_path.name, file_obj.read()),
            model="whisper-large-v3",
            response_format="verbose_json"
        )

    # Handle Pydantic model response or dictionary
    raw_segments = getattr(transcription, "segments", None)
    if raw_segments is None and isinstance(transcription, dict):
        raw_segments = transcription.get("segments", [])

    if not raw_segments:
        # Fallback to single text block if segments missing
        full_text = getattr(transcription, "text", "") or (transcription.get("text", "") if isinstance(transcription, dict) else "")
        full_text = full_text.strip()
        if full_text:
            return [{"start": 0.0, "end": 30.0, "text": full_text}]
        return []

    clean_segments: List[Dict[str, Any]] = []
    for seg in raw_segments:
        if hasattr(seg, "start"):
            start_val = getattr(seg, "start")
        elif isinstance(seg, dict):
            start_val = seg.get("start", 0.0)
        else:
            start_val = 0.0

        if hasattr(seg, "end"):
            end_val = getattr(seg, "end")
        elif isinstance(seg, dict):
            end_val = seg.get("end", 0.0)
        else:
            end_val = 0.0

        if hasattr(seg, "text"):
            text_val = getattr(seg, "text")
        elif isinstance(seg, dict):
            text_val = seg.get("text", "")
        else:
            text_val = ""

        text_clean = str(text_val).strip()
        if not text_clean:
            continue

        clean_segments.append({
            "start": round(float(start_val), 2),
            "end": round(float(end_val), 2),
            "text": text_clean
        })

    return clean_segments


def transcribe_audio(audio_path: Path) -> List[Dict[str, Any]]:
    """
    Transcribes audio using Groq Cloud API (whisper-large-v3) with full 413 protection:
    1. Ensures audio is compressed to 16kHz mono 32kbps MP3 (< 15MB/hour).
    2. Checks file size against the 24 MB Groq limit.
    3. If > 24 MB, automatically chunks into 20-minute fragments, transcribes sequentially,
       and merges timestamps with exact offset adjustment.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if not GROQ_API_KEY:
        raise TranscriberError("GROQ_API_KEY is not configured in .env")

    client = Groq(api_key=GROQ_API_KEY)
    file_size = audio_path.stat().st_size
    temp_files_to_cleanup: List[Path] = []

    target_audio = audio_path

    # Step 1: Ensure audio is properly compressed if it's large or not an MP3
    if file_size > MAX_GROQ_FILE_SIZE_BYTES or audio_path.suffix.lower() != ".mp3":
        compressed_path = audio_path.with_name(f"{audio_path.stem}_groq_opt.mp3")
        try:
            target_audio = compress_audio(audio_path, compressed_path)
            temp_files_to_cleanup.append(compressed_path)
            file_size = target_audio.stat().st_size
        except Exception as comp_err:
            logger.warning("Compression error, trying original audio: %s", comp_err)
            target_audio = audio_path
            file_size = target_audio.stat().st_size

    try:
        # Step 2: If file is still > 24 MB, split into 20-minute chunks
        if file_size > MAX_GROQ_FILE_SIZE_BYTES:
            logger.warning(
                "Audio file size (%.2f MB) exceeds Groq limit (24 MB). Splitting into 20-minute chunks...",
                file_size / (1024 * 1024)
            )
            total_duration = get_audio_duration(target_audio)
            chunk_dir = target_audio.parent / "audio_chunks"
            chunk_items = split_audio_into_chunks(target_audio, total_duration, chunk_dir)

            all_segments: List[Dict[str, Any]] = []
            for chunk_file, offset_sec in chunk_items:
                temp_files_to_cleanup.append(chunk_file)
                logger.info("Transcribing chunk at offset %.1fs...", offset_sec)
                chunk_segments = _transcribe_single_file(client, chunk_file)
                for seg in chunk_segments:
                    all_segments.append({
                        "start": round(seg["start"] + offset_sec, 2),
                        "end": round(seg["end"] + offset_sec, 2),
                        "text": seg["text"]
                    })

            if chunk_dir.exists():
                shutil.rmtree(chunk_dir, ignore_errors=True)

            logger.info("Total transcribed segments from all chunks: %d", len(all_segments))
            return all_segments

        # Step 3: Single file transcription (<= 24 MB)
        segments = _transcribe_single_file(client, target_audio)
        logger.info("Groq transcription completed: %d segments generated.", len(segments))
        return segments

    except Exception as exc:
        err_msg = str(exc)
        logger.error("Groq transcription failed: %s", err_msg, exc_info=True)
        if "413" in err_msg or "too large" in err_msg.lower():
            raise TranscriberError(
                "Файл аудио слишком большой для Groq API (413). Попробуйте отправить видео меньшей длины."
            )
        raise TranscriberError(f"Ошибка расшифровки аудио через Groq API: {err_msg}")

    finally:
        # Clean up any intermediate compression artifacts
        for p in temp_files_to_cleanup:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass


def format_segments_for_llm(segments: List[Dict[str, Any]]) -> str:
    """
    Formats raw transcript segments into timestamped readable string for Gemini.
    Example: [00:15 -> 00:45] (sec: 15.0 - 45.0) Text content
    """
    formatted_lines = []
    for seg in segments:
        start_sec = float(seg.get("start", 0.0))
        end_sec = float(seg.get("end", 0.0))
        text = seg.get("text", "").strip()

        start_m, start_s = divmod(int(start_sec), 60)
        end_m, end_s = divmod(int(end_sec), 60)
        timestamp_str = f"[{start_m:02d}:{start_s:02d} -> {end_m:02d}:{end_s:02d}]"
        formatted_lines.append(f"{timestamp_str} (sec: {start_sec:.1f} - {end_sec:.1f}) {text}")

    return "\n".join(formatted_lines)
