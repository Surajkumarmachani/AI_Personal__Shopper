"""
main.py
=======
FastAPI backend for the emotion-aware shopping assistant.

Endpoints:
  GET  /health      → simple liveness check
  POST /chat        → send a message, get the assistant's reply + product cards
  POST /emotion     → send a webcam frame (base64), get the detected emotion
  POST /transcribe  → send a recorded audio blob, get the transcribed text
                      (+ voice-tone emotion when the webcam is unavailable)
  POST /speak       → send text, get spoken audio (MP3) back

Auth: every endpoint except /health requires an `X-API-Key` header
(see api_keys.py for creating/revoking keys and rate limits).

Privacy: user text is scrubbed of PII (Presidio) BEFORE being logged.
The real, unscrubbed message still goes to the agent — only logs are scrubbed.

Run it:
  uvicorn main:app --reload --port 8000
"""

import base64
import logging
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException, UploadFile, File, Form, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging BEFORE importing our own modules, so their import-time
# log messages (like Presidio's readiness check) actually get printed.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from agent import run_agent
from emotion import detect_emotion, warm_up
from voice import transcribe_audio, synthesize_speech
from voice_emotion import detect_emotion_from_audio
from privacy import scrub_for_logging
from api_keys import Client, require_api_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up DeepFace once, at startup, so the first /emotion call isn't slow.
    warm_up()
    yield


app = FastAPI(title="Emotion Shop Backend", lifespan=lifespan)

# Allow ANY frontend origin to call this API (safe for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# /chat
# ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str           # unique per browser session — drives memory
    message: str              # the shopper's text
    emotion: str = "neutral"  # from webcam/voice/text; default neutral
    speak: bool = False       # true → also synthesize the reply as speech and
                              # return it in the same response (audio_b64), so
                              # the frontend doesn't need a second /speak call


class Product(BaseModel):
    title: str
    description: str = ""
    category: str = ""
    brand: str = ""
    price: float | None = None
    currency: str = "USD"
    image_url: str = ""
    product_url: str = ""
    avg_rating: float | None = None
    rating_count: int | None = None
    score: float = 0.0


class ChatResponse(BaseModel):
    response: str
    emotion_used: str
    products: list[Product]
    audio_b64: str = ""       # base64 MP3 of the reply (only when speak=true)
    audio_mime: str = "audio/mpeg"


# ─────────────────────────────────────────────
# /emotion
# ─────────────────────────────────────────────
class EmotionRequest(BaseModel):
    session_id: str = "anon"
    image: str                # base64 JPEG/PNG (data-URL prefix is fine)


class EmotionResponse(BaseModel):
    emotion: str              # normalized to tone-map keys (e.g. "surprised")
    emoji: str = "😐"
    confidence: float         # 0-1
    face_detected: bool
    webcam_available: bool
    error: str = ""


# ─────────────────────────────────────────────
# /transcribe  and  /speak
# ─────────────────────────────────────────────
class TranscribeResponse(BaseModel):
    text: str
    # Voice-tone emotion — only populated when webcam_active was "false"
    emotion: str | None = None
    emotion_confidence: float = 0.0
    emotion_source: str = "none"   # "voice" when librosa analysis ran


class SpeakRequest(BaseModel):
    text: str
    lang: str = "en"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/emotion", response_model=EmotionResponse, dependencies=[Depends(require_api_key)])
async def emotion(req: EmotionRequest):
    # detect_emotion never raises and DeepFace inference is CPU-bound, so run it
    # in a threadpool to keep the event loop free for chat requests.
    result = await run_in_threadpool(detect_emotion, req.image)
    return EmotionResponse(
        emotion=result.get("emotion", "neutral"),
        emoji=result.get("emoji", "😐"),
        confidence=result.get("confidence", 0.0),
        face_detected=result.get("face_detected", False),
        webcam_available=result.get("webcam_available", False),
        error=result.get("error", ""),
    )


@app.post("/transcribe", response_model=TranscribeResponse, dependencies=[Depends(require_api_key)])
async def transcribe(
    audio: UploadFile = File(...),
    webcam_active: str = Form("true"),
):
    audio_bytes = await audio.read()

    # Whisper inference is CPU-bound → threadpool so it doesn't block the loop.
    text = await run_in_threadpool(transcribe_audio, audio_bytes)
    log.info(f"transcript: {scrub_for_logging(text)}")

    # Fallback tier 2: webcam unavailable → estimate emotion from voice tone.
    emotion_val, confidence, source = None, 0.0, "none"
    if webcam_active.strip().lower() != "true" and audio_bytes:
        emotion_val, confidence = await run_in_threadpool(
            detect_emotion_from_audio, audio_bytes
        )
        source = "voice"

    return TranscribeResponse(
        text=text,
        emotion=emotion_val,
        emotion_confidence=confidence,
        emotion_source=source,
    )


@app.post("/speak", dependencies=[Depends(require_api_key)])
async def speak(req: SpeakRequest):
    audio = await run_in_threadpool(synthesize_speech, req.text, req.lang)
    return Response(content=audio, media_type="audio/mpeg")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, client: Client = Depends(require_api_key)):
    # Namespace the session by client so two clients sending the same
    # session_id never share conversation memory or product results.
    session_id = f"{client.id}:{req.session_id}"

    # PII-scrubbed copy for the logs; the REAL message goes to the agent.
    log.info(f"[{session_id}] ({client.name}) user: {scrub_for_logging(req.message)}")
    try:
        reply, products = await run_agent(
            message=req.message,
            emotion=req.emotion,
            session_id=session_id,
        )

        # Voice-with-text: synthesize the reply in the SAME response so the
        # frontend can play audio the moment the text renders — no second
        # round trip to /speak. TTS failure never breaks the chat reply.
        audio_b64 = ""
        if req.speak and reply.strip():
            try:
                audio = await run_in_threadpool(synthesize_speech, reply)
                audio_b64 = base64.b64encode(audio).decode("ascii")
            except Exception:
                log.exception("TTS during /chat failed — returning text only")

        return ChatResponse(
            response=reply,
            emotion_used=(req.emotion or "neutral").lower(),
            # run_agent returns plain dicts; build Product models explicitly
            # (same as Pydantic's implicit coercion, but typed cleanly).
            products=[Product(**p) for p in products],
            audio_b64=audio_b64,
        )
    except Exception as e:
        log.exception("chat endpoint failed")
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")