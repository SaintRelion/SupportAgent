"""
Analyzer pipeline:
  CloudFront URL → download bytes locally → Deepgram transcription → LLM feedback
"""

import asyncio
import logging
import httpx
from deepgram import DeepgramClient
from langchain_core.messages import HumanMessage, SystemMessage

from shared.config import DEEPGRAM_API_KEY
from shared.llm import get_llm

log = logging.getLogger("pipeline")

DOWNLOAD_TIMEOUT = 300      # 5 min for large wav files
TRANSCRIBE_RETRIES = 3      # retry transcription up to 3 times
RETRY_DELAY = 5             # seconds between retries


FEEDBACK_PROMPT = """You are an expert coaching call reviewer for a sales/coaching team.

Analyze the following call transcript and provide structured feedback.

Return your feedback in this exact format:

## Overall Score
X/10

## Strengths
- (specific strength with timestamp or quote if possible)
- ...

## Areas for Improvement
- (specific weakness with actionable suggestion)
- ...

## Key Moments
- (notable moments, good or bad, worth discussing)

## Summary
(2-3 sentence overall takeaway)

Transcript:
{transcript}
"""


async def download_audio(url: str) -> bytes:
    """Download audio from CloudFront with extended timeout."""
    log.info("[DOWNLOAD] Starting download: %s", url)
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(DOWNLOAD_TIMEOUT),
        follow_redirects=True,
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        size_kb = len(response.content) / 1024
        log.info("[DOWNLOAD] Done — %.1f KB", size_kb)
        return response.content


def _transcribe_once(audio_bytes: bytes) -> str:
    """Single transcription attempt."""
    deepgram = DeepgramClient(api_key=DEEPGRAM_API_KEY)
    response = deepgram.listen.v1.media.transcribe_file(
        request=audio_bytes,
        model="nova-3",
        language="en",
        smart_format=True,
        diarize=True,
        punctuate=True,
        utterances=True,
    )
    utterances = response.results.utterances
    if utterances:
        lines = []
        for u in utterances:
            mins, secs = divmod(int(u.start), 60)
            lines.append(f"[{mins:02d}:{secs:02d}] Speaker {u.speaker}: {u.transcript}")
        return "\n".join(lines)
    return response.results.channels[0].alternatives[0].transcript


async def transcribe_bytes(audio_bytes: bytes) -> str:
    """
    Transcribe with retry on timeout or transient errors.
    Retries up to TRANSCRIBE_RETRIES times with RETRY_DELAY seconds between.
    """
    last_error = None
    for attempt in range(1, TRANSCRIBE_RETRIES + 1):
        try:
            log.info("[TRANSCRIBE] Attempt %d/%d (%.1f KB)", attempt, TRANSCRIBE_RETRIES, len(audio_bytes) / 1024)
            transcript = await asyncio.to_thread(_transcribe_once, audio_bytes)
            log.info("[TRANSCRIBE] Success on attempt %d", attempt)
            return transcript
        except Exception as e:
            last_error = e
            log.warning("[TRANSCRIBE] Attempt %d failed: %s", attempt, e)
            if attempt < TRANSCRIBE_RETRIES:
                log.info("[TRANSCRIBE] Retrying in %ds...", RETRY_DELAY)
                await asyncio.sleep(RETRY_DELAY)

    raise RuntimeError(f"Transcription failed after {TRANSCRIBE_RETRIES} attempts: {last_error}")


async def generate_feedback(transcript: str) -> str:
    """Run transcript through LLM and return structured feedback."""
    llm = get_llm(temperature=0.4)
    response = llm.invoke([
        SystemMessage(content="You are a coaching call quality reviewer."),
        HumanMessage(content=FEEDBACK_PROMPT.format(transcript=transcript)),
    ])
    return response.content


async def analyze_recording(url: str) -> dict:
    """
    Full pipeline: CloudFront URL → download → transcribe (with retry) → feedback.

    Returns:
        {
            "transcript": str,
            "feedback": str,
            "error": str | None,
        }
    """
    try:
        # Run in thread to avoid blocking Discord's heartbeat during long downloads
        audio_bytes = await asyncio.to_thread(_download_sync, url)
        transcript = await transcribe_bytes(audio_bytes)
        feedback = await generate_feedback(transcript)
        return {"transcript": transcript, "feedback": feedback, "error": None}
    except Exception as e:
        log.error("[PIPELINE] Failed: %s", e)
        return {"transcript": "", "feedback": "", "error": str(e)}


def _download_sync(url: str) -> bytes:
    """Synchronous download — used inside asyncio.to_thread to avoid blocking heartbeat."""
    import httpx as _httpx
    log.info("[DOWNLOAD] Sync download starting: %s", url)
    with _httpx.Client(timeout=_httpx.Timeout(DOWNLOAD_TIMEOUT), follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        size_kb = len(response.content) / 1024
        log.info("[DOWNLOAD] Done — %.1f KB", size_kb)
        return response.content