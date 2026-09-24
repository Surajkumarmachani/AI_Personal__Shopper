"""
voice.py
========
Voice I/O:

  Speech-to-text : OpenAI Whisper API (whisper-1 model).
                   Audio is sent to OpenAI's API endpoint. Cost: ~$0.006/min.
  Text-to-speech : ElevenLabs (primary, production-grade) with automatic
                   fallback to gTTS if ElevenLabs is unavailable — no API key
                   configured, quota exhausted, network error, etc.

ENV VARS (loaded from .env by config.py):
  OPENAI_API_KEY        required — Whisper transcription
  ELEVENLABS_API_KEY    optional — enables ElevenLabs TTS; without it we use gTTS
  ELEVENLABS_VOICE_ID   optional — defaults to "21m00Tcm4TlvDq8ikWAM" (Rachel)
  ELEVENLABS_MODEL_ID   optional — defaults to "eleven_flash_v2_5" (low latency)
"""

import io
import os
import logging
from openai import OpenAI
from gtts import gTTS

log = logging.getLogger(__name__)

# OpenAI client — reads OPENAI_API_KEY from env automatically.
_client = OpenAI()

# ─────────────────────────────────────────────
# ElevenLabs TTS (primary) — created once at import time
# ─────────────────────────────────────────────
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
# eleven_flash_v2_5 is ElevenLabs' low-latency model (~75ms model time) —
# noticeably faster than eleven_multilingual_v2 for conversational replies,
# still multilingual. Override via env if you prefer higher-fidelity voices.
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")

_elevenlabs = None
if os.getenv("ELEVENLABS_API_KEY"):
    try:
        from elevenlabs.client import ElevenLabs

        _elevenlabs = ElevenLabs()  # reads ELEVENLABS_API_KEY from env
        log.info(
            f"ElevenLabs TTS ready (voice={ELEVENLABS_VOICE_ID}, "
            f"model={ELEVENLABS_MODEL_ID}). gTTS available as fallback."
        )
    except Exception as e:
        log.warning(f"ElevenLabs init failed ({e}) — falling back to gTTS.")
else:
    log.warning("ELEVENLABS_API_KEY not set — using gTTS for TTS (dev fallback).")


def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribe a recorded audio blob (webm/mp4/ogg/wav) to text using the
    OpenAI Whisper API. The browser's MediaRecorder output works directly —
    OpenAI accepts webm, mp4, wav, m4a, mp3, etc.
    """
    audio_file = io.BytesIO(audio_bytes)
    # OpenAI SDK requires a filename with an extension so it can infer the format.
    audio_file.name = "recording.webm"

    transcript = _client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
    )
    return transcript.text.strip()


def _speak_elevenlabs(text: str) -> bytes:
    """ElevenLabs TTS → MP3 bytes. Raises on any API failure (caller handles)."""
    assert _elevenlabs is not None
    stream = _elevenlabs.text_to_speech.convert(
        voice_id=ELEVENLABS_VOICE_ID,
        model_id=ELEVENLABS_MODEL_ID,   # multilingual — detects language from text
        text=text,
        output_format="mp3_44100_128",
    )
    # The SDK returns an iterator of MP3 chunks — join into one blob.
    audio = b"".join(stream)
    if not audio:
        raise RuntimeError("ElevenLabs returned empty audio")
    return audio


def _speak_gtts(text: str, lang: str = "en") -> bytes:
    """gTTS TTS → MP3 bytes (free, unofficial endpoint — fallback only)."""
    buf = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(buf)
    buf.seek(0)
    return buf.read()


def synthesize_speech(text: str, lang: str = "en") -> bytes:
    """
    Turn text into spoken MP3 audio bytes.

    Primary:  ElevenLabs (when ELEVENLABS_API_KEY is configured).
    Fallback: gTTS — used when ElevenLabs is not configured or its call fails,
              so /speak keeps working even during quota/network issues.
    """
    if _elevenlabs is not None:
        try:
            return _speak_elevenlabs(text)
        except Exception as e:
            log.warning(f"ElevenLabs TTS failed ({e}) — falling back to gTTS.")
    return _speak_gtts(text, lang)
