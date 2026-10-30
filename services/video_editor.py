"""
Re-export ffmpeg_editor methods for backwards compatibility.
"""
from services.ffmpeg_editor import (
    ProcessedClip,
    VideoEditorError,
    create_ass_subtitles,
    format_ass_timestamp,
    process_all_clips,
    render_clip,
)

__all__ = [
    "ProcessedClip",
    "VideoEditorError",
    "create_ass_subtitles",
    "format_ass_timestamp",
    "process_all_clips",
    "render_clip",
]
