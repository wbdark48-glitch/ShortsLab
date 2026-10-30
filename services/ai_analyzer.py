import json
import re
import warnings
from typing import Any, Dict, List
from pydantic import BaseModel, Field, field_validator

warnings.filterwarnings("ignore", category=FutureWarning)
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError

from config import GEMINI_API_KEY, logger
from services.transcriber import format_segments_for_llm


class ClipProposal(BaseModel):
    start_time: float = Field(..., description="Start timestamp in seconds")
    end_time: float = Field(..., description="End timestamp in seconds")
    title: str = Field(..., description="Short catchy hook title for vertical clip")
    virality_score: int = Field(default=8, description="Score from 1 to 10 predicting virality")
    reason: str = Field(default="", description="Reason why this fragment was selected")

    @field_validator("end_time")
    @classmethod
    def validate_duration(cls, end_val: float, info):
        start_val = info.data.get("start_time")
        if start_val is not None and end_val <= start_val:
            raise ValueError(f"end_time ({end_val}) must be greater than start_time ({start_val})")
        return end_val


class AIAnalyzerError(Exception):
    """Base exception for Gemini analysis errors."""
    pass


class AIQuotaExceededError(AIAnalyzerError):
    """Raised when Gemini API quota or rate limit is reached."""
    pass


def extract_json_from_text(raw_text: str) -> str:
    """
    Cleans raw LLM response removing Markdown formatting (```json ... ```)
    and extracts the JSON array block using regex fallback if needed.
    """
    cleaned = raw_text.strip()

    # Remove Markdown code blocks if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    # Try standard parse
    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        pass

    # Regex search for the outermost JSON array [...]
    match = re.search(r"\[\s*\{.*\}\s*\]", cleaned, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # Regex search for any JSON objects and wrap them into a list
    objects = re.findall(r"\{[^{}]*\}", cleaned, re.DOTALL)
    if objects:
        valid_items = []
        for obj_str in objects:
            try:
                valid_items.append(json.loads(obj_str))
            except json.JSONDecodeError:
                continue
        if valid_items:
            return json.dumps(valid_items)

    raise AIAnalyzerError(f"Failed to extract valid JSON from Gemini response: {raw_text[:200]}")


class AIAnalyzer:
    def __init__(self, api_key: str = GEMINI_API_KEY):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")
        genai.configure(api_key=api_key)
        self.model_name = "gemini-3.8-flash"
        self._init_model()

    def _init_model(self):
        try:
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "temperature": 0.4,
                    "top_p": 0.9,
                    "response_mime_type": "application/json"
                }
            )
        except Exception as exc:
            logger.warning("Failed initializing %s (%s), falling back to gemini-1.5-flash", self.model_name, exc)
            self.model_name = "gemini-1.5-flash"
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "temperature": 0.4,
                    "top_p": 0.9,
                    "response_mime_type": "application/json"
                }
            )

    def analyze_transcript(
        self,
        segments: List[Dict[str, Any]],
        total_duration: float
    ) -> List[ClipProposal]:
        """
        Sends timestamped transcript to Gemini 2.5 Flash to identify top 3 viral moments
        with durations between 15 and 75 seconds.
        """
        if not segments:
            raise AIAnalyzerError("Transcript is empty, cannot identify clips.")

        formatted_transcript = format_segments_for_llm(segments)

        prompt = (
            "You are an elite short-form video producer (specializing in YouTube Shorts, TikTok, and Instagram Reels). "
            "Analyze the following transcript with timestamps. "
            "Identify the TOP 3 most engaging, funny, controversial, insightful, or viral moments. "
            "CRITICAL RULES:\n"
            "1. Each clip duration MUST be between 15 and 75 seconds (ideally 30 to 60 seconds).\n"
            f"2. The total video duration is {total_duration:.1f} seconds. Never exceed this duration.\n"
            "3. Select fragments that start with a strong hook and have a coherent thought.\n\n"
            "Return STRICT JSON matching this schema:\n"
            "[\n"
            "  {\n"
            '    "start_time": 12.5,\n'
            '    "end_time": 45.0,\n'
            '    "title": "Short Hook Title",\n'
            '    "virality_score": 9,\n'
            '    "reason": "Brief reason"\n'
            "  }\n"
            "]\n\n"
            f"TRANSCRIPT WITH TIMESTAMPS:\n{formatted_transcript}"
        )

        logger.info("Calling Gemini (%s) to select viral moments...", self.model_name)
        try:
            try:
                response = self.model.generate_content(prompt)
            except Exception as model_err:
                # If gemini-2.5-flash fails specifically due to model name availability, fallback gracefully
                if "not found" in str(model_err).lower() or "unsupported" in str(model_err).lower():
                    logger.warning("Gemini 2.5 flash unavailable, switching to 1.5 flash: %s", model_err)
                    self.model = genai.GenerativeModel(
                        model_name="gemini-1.5-flash",
                        generation_config={
                            "temperature": 0.4,
                            "top_p": 0.9,
                            "response_mime_type": "application/json"
                        }
                    )
                    response = self.model.generate_content(prompt)
                else:
                    raise model_err

            if not response or not response.text:
                raise AIAnalyzerError("Empty response received from Gemini API.")

            raw_response = response.text.strip()
            logger.info("Gemini raw response received (%d chars).", len(raw_response))

            clean_json_str = extract_json_from_text(raw_response)
            parsed_data = json.loads(clean_json_str)

            if not isinstance(parsed_data, list):
                if isinstance(parsed_data, dict):
                    # If model returned wrapped dict like {"clips": [...]}
                    for val in parsed_data.values():
                        if isinstance(val, list):
                            parsed_data = val
                            break
                    else:
                        parsed_data = [parsed_data]
                else:
                    raise AIAnalyzerError("Expected a JSON array of clip proposals.")

            clips: List[ClipProposal] = []
            for item in parsed_data:
                try:
                    clip = ClipProposal(**item)
                    # Clamp start time
                    if clip.start_time < 0:
                        clip.start_time = 0.0

                    # Clamp end time to total duration
                    if total_duration > 0 and clip.end_time > total_duration:
                        clip.end_time = total_duration

                    # Validate clip duration bounds (15s <= duration <= 75s)
                    clip_dur = clip.end_time - clip.start_time
                    if clip_dur < 15.0:
                        clip.end_time = min(total_duration, clip.start_time + 30.0)
                    elif clip_dur > 75.0:
                        clip.end_time = clip.start_time + 60.0

                    clips.append(clip)
                except Exception as pydantic_err:
                    logger.warning("Skipping invalid clip proposal %s: %s", item, pydantic_err)

            if not clips:
                raise AIAnalyzerError("No valid clip proposals could be parsed from Gemini response.")

            # Sort by virality score descending and take up to top 3
            clips.sort(key=lambda x: x.virality_score, reverse=True)
            top_clips = clips[:3]
            logger.info("Identified %d viral clips successfully.", len(top_clips))
            return top_clips

        except GoogleAPIError as exc:
            err_str = str(exc)
            logger.error("Gemini API error: %s", err_str)
            if "429" in err_str or "quota" in err_str.lower() or "resource exhausted" in err_str.lower():
                raise AIQuotaExceededError(
                    "Превышен лимит запросов к Gemini API. Пожалуйста, подождите минуту и повторите попытку."
                )
            raise AIAnalyzerError(f"Ошибка Gemini API: {err_str}")
        except Exception as exc:
            logger.error("Unexpected error in AI analyzer: %s", exc)
            if isinstance(exc, (AIAnalyzerError, AIQuotaExceededError)):
                raise
            raise AIAnalyzerError(f"Не удалось проанализировать видео через AI: {str(exc)}")
