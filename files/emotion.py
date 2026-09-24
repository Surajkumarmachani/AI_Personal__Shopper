"""
emotion.py
==========
Wraps DeepFace facial-emotion detection for the /emotion endpoint.

Design points:
- enforce_detection=False → NEVER raises when no face is visible; returns a
  best-effort result instead of crashing (we map the no-face case to 'neutral').
- DeepFace's labels (angry/disgust/fear/happy/sad/surprise/neutral) are
  normalized to the tone-map keys used in config.py.
- warm_up() runs one dummy analysis at startup so the first REAL request
  doesn't pay the one-time model download + load cost.

Everything is wrapped so this module never crashes the /emotion endpoint —
on any failure it returns neutral + webcam_available=False, keeping the
fallback chain (webcam → voice → text) clean.
"""

import base64
import logging
import numpy as np
import cv2
from deepface import DeepFace
from typing import Any

from config import emotion_to_emoji

log = logging.getLogger(__name__)

# DeepFace emotion labels → keys used in config.EMOTION_TONE_MAP
DEEPFACE_TO_TONE = {
    "angry":    "angry",
    "disgust":  "disgusted",
    "fear":     "fearful",
    "happy":    "happy",
    "sad":      "sad",
    "surprise": "surprised",
    "neutral":  "neutral",
}

# opencv is bundled with the opencv-python package, needs no GPU, and is fast
# enough for a 1-frame-every-2-seconds webcam feed.
DETECTOR_BACKEND = "opencv"


def _decode_base64_image(image_b64: str) -> np.ndarray:
    """Decode a base64 (optionally data-URL) string into a BGR numpy image."""
    if "," in image_b64:                       # strip 'data:image/jpeg;base64,' prefix
        image_b64 = image_b64.split(",", 1)[1]
    img_bytes = base64.b64decode(image_b64)
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR — what DeepFace expects
    if img is None:
        raise ValueError("could not decode image")
    return img


def _pick_face(result: Any) -> dict:
    """DeepFace may return a list of faces or a single face dict."""
    if isinstance(result, list):
        return result[0] if result else {}
    if isinstance(result, dict):
        return result
    return {}


def _analyze(img: np.ndarray) -> dict:
    """Run DeepFace, normalize the result to {emotion, confidence, face_detected}."""
    h_img, w_img = img.shape[:2]

    result = DeepFace.analyze(
        img,
        actions=["emotion"],
        enforce_detection=False,     # ← never crash on a missing face
        detector_backend=DETECTOR_BACKEND,
        silent=True,
    )
    # DeepFace returns a LIST (one dict per detected face). Older versions
    # returned a bare dict — handle both.
    face = _pick_face(result)

    dominant = str(face.get("dominant_emotion", "neutral")).lower()
    scores = face.get("emotion", {}) or {}
    # DeepFace emotion scores are 0-100 percentages → normalize to 0-1
    raw_conf = float(scores.get(dominant, 0.0)) / 100.0
    emotion = DEEPFACE_TO_TONE.get(dominant, "neutral")

    # Decide whether a real face was actually found. When DeepFace finds nothing
    # (enforce_detection=False), it returns the WHOLE frame as the "face" region
    # and face_confidence ~0. In that case we don't trust the emotion — it was
    # guessed from background pixels — so we return neutral.
    face_conf = float(face.get("face_confidence", 0.0) or 0.0)
    region = face.get("region", {}) or {}
    rw = float(region.get("w", w_img) or w_img)
    rh = float(region.get("h", h_img) or h_img)
    covers_full_frame = rw >= w_img * 0.98 and rh >= h_img * 0.98
    face_detected = (face_conf > 0.0) or (not covers_full_frame) or (emotion != "neutral")

    if not face_detected:
        log.info(f"No face detected. face_conf={face_conf}, rw={rw}, rh={rh}, w_img={w_img}, h_img={h_img}")
        return {
            "emotion": "neutral",
            "emoji": emotion_to_emoji("neutral"),
            "confidence": 0.0,
            "face_detected": False,
        }

    log.info(f"Face detected. Emotion: {emotion}, confidence: {raw_conf}")
    return {
        "emotion": emotion,
        "emoji": emotion_to_emoji(emotion),
        "confidence": round(raw_conf, 3),
        "face_detected": True,
    }


def detect_emotion(image_b64: str) -> dict:
    """
    Public entry point. ALWAYS returns a dict; never raises.
    On any failure: neutral + webcam_available=False (keeps the fallback clean).
    """
    try:
        img = _decode_base64_image(image_b64)
        out = _analyze(img)
        out["webcam_available"] = True
        out["error"] = ""
        return out
    except Exception as e:
        log.warning(f"emotion detection failed: {e}")
        return {
            "emotion": "neutral",
            "emoji": emotion_to_emoji("neutral"),
            "confidence": 0.0,
            "face_detected": False,
            "webcam_available": False,
            "error": str(e),
        }


def warm_up() -> None:
    """
    Trigger the one-time model download + load at startup so the first real
    request is fast. On first ever run this downloads the emotion model weights
    (~a few MB from GitHub). Failure here is non-fatal — the first request just
    pays the cost instead.
    """
    try:
        log.info("Warming up DeepFace (first run downloads model weights)…")
        dummy = np.zeros((48, 48, 3), dtype=np.uint8)
        DeepFace.analyze(
            dummy,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend=DETECTOR_BACKEND,
            silent=True,
        )
        log.info("DeepFace ready.")
    except Exception as e:
        log.warning(f"DeepFace warm-up skipped (first request will be slower): {e}")