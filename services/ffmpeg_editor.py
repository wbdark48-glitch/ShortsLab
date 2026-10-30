import functools
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import logger
from services.ai_analyzer import ClipProposal


class VideoEditorError(Exception):
    """Base exception for video editing failures."""
    pass


@dataclass
class ProcessedClip:
    clip_path: Path
    title: str
    virality_score: int
    reason: str
    duration: float
    start_time: float
    end_time: float


@functools.lru_cache(maxsize=1)
def get_optimal_encoder_settings() -> List[str]:
    """
    Auto-detects available hardware acceleration encoder (NVIDIA NVENC, Intel QSV)
    with seamless fallback to CPU libx264 with ultrafast preset.
    """
    # 1. Test NVIDIA NVENC
    try:
        res = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64", "-vframes", "1", "-c:v", "h264_nvenc", "-f", "null", "-"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2
        )
        if res.returncode == 0:
            logger.info("Hardware acceleration enabled: NVIDIA NVENC (h264_nvenc)")
            return ["-c:v", "h264_nvenc", "-preset", "p1", "-cq", "24"]
    except Exception:
        pass

    # 2. Test Intel QuickSync (QSV)
    try:
        res = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64", "-vframes", "1", "-c:v", "h264_qsv", "-f", "null", "-"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2
        )
        if res.returncode == 0:
            logger.info("Hardware acceleration enabled: Intel QSV (h264_qsv)")
            return ["-c:v", "h264_qsv", "-preset", "veryfast", "-global_quality", "24"]
    except Exception:
        pass

    # 3. CPU fallback: libx264 ultrafast with full multithreading
    logger.info("Using optimized CPU encoder: libx264 (preset: ultrafast, threads: 0, crf: 24)")
    return ["-c:v", "libx264", "-preset", "ultrafast", "-threads", "0", "-crf", "24"]


def format_ass_timestamp(seconds: float) -> str:
    """Converts seconds (e.g. 75.34) to ASS format: H:MM:SS.cs"""
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis >= 100:
        centis = 99
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def split_segment_into_dynamic_chunks(
    rel_start: float,
    rel_end: float,
    text: str,
    max_words: int = 4
) -> List[Tuple[float, float, str]]:
    """
    Splits transcript segments strictly into short 2-4 word chunks
    with proportional timestamps for TikTok/Reels/Shorts dynamic pacing.
    """
    clean = text.replace("{", "").replace("}", "").replace("\n", " ").strip()
    words = clean.split()
    if not words:
        return []

    total_words = len(words)
    total_duration = max(0.3, rel_end - rel_start)

    # Group words into 2 to 4 words per chunk (target 2-3 words for rapid pacing)
    grouped_words = []
    i = 0
    while i < total_words:
        rem = total_words - i
        if rem <= max_words:
            grouped_words.append(words[i:])
            break
        elif rem == 5:
            # 3 + 2
            grouped_words.append(words[i:i + 3])
            grouped_words.append(words[i + 3:i + 5])
            break
        elif rem == 6:
            # 3 + 3
            grouped_words.append(words[i:i + 3])
            grouped_words.append(words[i + 3:i + 6])
            break
        else:
            grouped_words.append(words[i:i + 3])
            i += 3

    chunks = []
    current_time = rel_start
    for group in grouped_words:
        chunk_word_count = len(group)
        chunk_dur = total_duration * (chunk_word_count / total_words)
        chunk_start = current_time
        chunk_end = min(rel_end, current_time + chunk_dur)
        chunk_text = " ".join(group).upper()
        chunks.append((chunk_start, chunk_end, chunk_text))
        current_time = chunk_end

    return chunks


def create_ass_subtitles(
    clip_start: float,
    clip_end: float,
    segments: List[Dict[str, Any]],
    clip_title: str,
    output_ass_path: Path
) -> Path:
    """
    Creates an Advanced SubStation Alpha (.ass) subtitle file formatted
    in professional TikTok/Reels/Shorts style:
    - Font: Impact / Arial Black (Bold)
    - Font size: FontSize=24 (large, mobile-optimized format on 720p virtual canvas)
    - Spoken text: Bright yellow &H0000FFFF, dense black outline Outline=3.5, shadow Shadow=1.5
    - Alignment: Strictly bottom-center (Alignment=2, MarginV=180) to avoid UI overlay
    - Hook title: Elegant semi-transparent banner top-center (Alignment=8, MarginV=120)
                  with BackColour=&H80000000&, BorderStyle=3, Outline=12 for first 3.5s
    """
    output_ass_path.parent.mkdir(parents=True, exist_ok=True)
    clip_duration = clip_end - clip_start

    # Filter segments that overlap with this clip's time window
    relevant_chunks: List[Tuple[float, float, str]] = []
    for seg in segments:
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", 0.0))
        seg_text = str(seg.get("text", "")).strip()

        if seg_end > clip_start and seg_start < clip_end:
            rel_start = max(0.0, seg_start - clip_start)
            rel_end = min(clip_duration, seg_end - clip_start)
            if rel_end - rel_start > 0.15 and seg_text:
                # Break down into dynamic 2-4 word chunks
                chunks = split_segment_into_dynamic_chunks(rel_start, rel_end, seg_text, max_words=4)
                relevant_chunks.extend(chunks)

    safe_title = clip_title.replace("{", "").replace("}", "").replace("\n", " ").strip().upper()

    # ASS Script structure (PlayResX: 405, PlayResY: 720 provides exact 9:16 vertical coordinate space)
    ass_content = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 405",
        "PlayResY: 720",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        # Style for Hook Title: Elegant semi-transparent box background (BorderStyle=3, Outline=12, MarginV=120)
        "Style: HookTitle,Arial Black,20,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,3,12,0,8,25,25,120,1",
        # Style for Spoken Subtitles: Bold yellow font, 3.5px black outline, 1.5px shadow, lower third (MarginV=180)
        "Style: SpokenText,Impact,24,&H0000FFFF,&H000000FF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,3.5,1.5,2,20,20,180,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
    ]

    # Show Hook Title in top center during the first 3.5 seconds
    title_duration = min(3.5, clip_duration)
    ass_content.append(
        f"Dialogue: 1,{format_ass_timestamp(0.0)},{format_ass_timestamp(title_duration)},HookTitle,,0,0,0,,{{\\b1}}{safe_title}{{\\b0}}"
    )

    # Add dynamic rapid subtitles (2-4 words each)
    for chunk_start, chunk_end, chunk_text in relevant_chunks:
        if chunk_text:
            start_str = format_ass_timestamp(chunk_start)
            end_str = format_ass_timestamp(chunk_end)
            ass_content.append(
                f"Dialogue: 0,{start_str},{end_str},SpokenText,,0,0,0,,{chunk_text}"
            )

    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ass_content) + "\n")

    return output_ass_path


def escape_ffmpeg_filter_path(path: Path) -> str:
    """
    Escapes filesystem path for FFmpeg filter expressions.
    Handles Windows colons and backslashes properly.
    """
    posix_path = path.resolve().as_posix()
    escaped = posix_path.replace(":", r"\:")
    return escaped


def render_clip(
    source_video: Path,
    proposal: ClipProposal,
    segments: List[Dict[str, Any]],
    output_clip_path: Path
) -> ProcessedClip:
    """
    Renders vertical 9:16 clip in a single-pass filtergraph:
    - Crops & scales (crop=ih*9/16:ih,scale=1080:1920)
    - Burns in hook title & subtitles in one pass
    - Utilizes hardware acceleration (h264_nvenc/h264_qsv) or ultrafast libx264 with threads 0
    """
    if not shutil.which("ffmpeg"):
        raise VideoEditorError("FFmpeg executable not found in system PATH.")

    output_clip_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(5.0, proposal.end_time - proposal.start_time)

    # Generate ASS subtitle file
    ass_path = output_clip_path.with_suffix(".ass")
    create_ass_subtitles(
        clip_start=proposal.start_time,
        clip_end=proposal.end_time,
        segments=segments,
        clip_title=proposal.title,
        output_ass_path=ass_path
    )

    escaped_ass = escape_ffmpeg_filter_path(ass_path)

    # Single-pass filtergraph combining crop, scale, hook title banner and subtitles
    crop_scale = "crop=ih*9/16:ih,scale=1080:1920"
    vf_with_subs = f"{crop_scale},subtitles='{escaped_ass}'"
    vf_plain = crop_scale

    encoder_args = get_optimal_encoder_settings()
    cpu_encoder_args = ["-c:v", "libx264", "-preset", "ultrafast", "-threads", "0", "-crf", "24"]

    base_input_args = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-ss", f"{proposal.start_time:.2f}",
        "-i", str(source_video),
        "-t", f"{duration:.2f}",
    ]
    common_output_args = [
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(output_clip_path)
    ]

    cmd = base_input_args + ["-vf", vf_with_subs] + encoder_args + common_output_args

    logger.info(
        "Rendering clip '%s' (%.1fs -> %.1fs, dur: %.1fs) via %s...",
        proposal.title, proposal.start_time, proposal.end_time, duration, encoder_args[1]
    )

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Fallback 1: If hardware encoder failed, retry immediately with CPU ultrafast
    if result.returncode != 0 and encoder_args[1] != "libx264":
        logger.warning(
            "Hardware encoder (%s) failed with code %d. Falling back to libx264 ultrafast. Details: %s",
            encoder_args[1], result.returncode, result.stderr[-300:]
        )
        cmd_cpu = base_input_args + ["-vf", vf_with_subs] + cpu_encoder_args + common_output_args
        result = subprocess.run(cmd_cpu, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Fallback 2: If subtitles filter failed, retry crop-only
    if result.returncode != 0:
        logger.warning(
            "FFmpeg failed with subtitles filter (code %d). Retrying with crop-only. Details: %s",
            result.returncode, result.stderr[-300:]
        )
        cmd_fallback = base_input_args + ["-vf", vf_plain] + cpu_encoder_args + common_output_args
        fallback_result = subprocess.run(cmd_fallback, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if fallback_result.returncode != 0:
            logger.error("FFmpeg fallback rendering failed: %s", fallback_result.stderr)
            raise VideoEditorError(f"Ошибка нарезки видео FFmpeg: {fallback_result.stderr[-300:]}")

    if not output_clip_path.exists() or output_clip_path.stat().st_size == 0:
        raise VideoEditorError(f"Rendered clip file is missing or empty: {output_clip_path}")

    # Remove temporary ASS file
    try:
        if ass_path.exists():
            ass_path.unlink()
    except Exception as exc:
        logger.warning("Could not delete temp subtitle file %s: %s", ass_path, exc)

    return ProcessedClip(
        clip_path=output_clip_path,
        title=proposal.title,
        virality_score=proposal.virality_score,
        reason=proposal.reason,
        duration=duration,
        start_time=proposal.start_time,
        end_time=proposal.end_time
    )


def process_all_clips(
    source_video: Path,
    proposals: List[ClipProposal],
    segments: List[Dict[str, Any]],
    session_dir: Path
) -> List[ProcessedClip]:
    """
    Renders all proposed viral clips sequentially and returns list of ProcessedClip objects.
    """
    processed_clips: List[ProcessedClip] = []
    for idx, proposal in enumerate(proposals, start=1):
        output_file = session_dir / f"output_clip_{idx}.mp4"
        try:
            clip = render_clip(
                source_video=source_video,
                proposal=proposal,
                segments=segments,
                output_clip_path=output_file
            )
            processed_clips.append(clip)
        except Exception as exc:
            logger.error("Failed to render clip %d ('%s'): %s", idx, proposal.title, exc)

    if not processed_clips:
        raise VideoEditorError("Не удалось создать ни один видеоклип.")

    return processed_clips
