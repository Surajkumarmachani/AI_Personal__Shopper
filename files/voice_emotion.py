"""
voice_emotion.py
================
Tier 2 of the emotion fallback chain: estimate emotion from HOW something was
said (pitch, loudness, tempo, brightness of voice) using librosa — fully
open-source, runs locally, no API.

Only used when the webcam is NOT available (tier 1). The /transcribe endpoint
calls detect_emotion_from_audio() with the same audio blob it transcribes.

Decoding note: browsers record webm/opus (or mp4), which librosa.load can't
read without a system ffmpeg. Instead we decode with PyAV — already installed
as a faster-whisper dependency — straight into a numpy array, then run librosa
feature extractors on the raw samples.

Accuracy expectation: this is a rule-based estimate (~70% at best, honest
number). It detects happy / angry / sad / calm / neutral. It exists as a
FALLBACK, not a replacement for facial detection.
"""

import io
import logging
import numpy as np
import av
import librosa

log = logging.getLogger(__name__)

TARGET_SR = 16000


def _decode_audio(audio_bytes: bytes, target_sr: int = TARGET_SR):
    """Decode any browser-recorded container (webm/mp4/ogg/wav) → mono float32."""
    container = av.open(io.BytesIO(audio_bytes))
    resampler = av.AudioResampler(format="s16", layout="mono", rate=target_sr)
    chunks = []
    for frame in container.decode(audio=0):  # type: ignore[attr-defined]  # PyAV stubs miss .decode()
        for rf in resampler.resample(frame):
            chunks.append(rf.to_ndarray().reshape(-1))
    for rf in resampler.resample(None):  # flush remaining samples
        chunks.append(rf.to_ndarray().reshape(-1))
    container.close()
    if not chunks:
        return np.array([], dtype=np.float32), target_sr
    y = np.concatenate(chunks).astype(np.float32) / 32768.0
    return y, target_sr


def detect_emotion_from_audio(audio_bytes: bytes) -> tuple[str, float]:
    """
    Analyse voice tone → (emotion, confidence 0-1).
    Emotions: happy | angry | sad | calm | neutral (all exist in the tone map).
    Never raises — returns ("neutral", 0.0) on any failure or too-short audio.
    """
    try:
        y, sr = _decode_audio(audio_bytes)
        if len(y) < sr * 0.5:  # under half a second — nothing to analyse
            return "neutral", 0.0

        # Cap analysis at the first ~10s so a long recording stays fast
        y = y[: sr * 10]

        # ── Acoustic features ─────────────────────────────
        # 1) Pitch (fundamental frequency of the voice)
        f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=400, sr=sr)
        pitch = (
            float(np.nanmean(f0[voiced]))
            if voiced is not None and voiced.any()
            else 0.0
        )

        # 2) Energy (how loud/intense)
        energy = float(np.mean(librosa.feature.rms(y=y)))

        # 3) Tempo (speaking pace proxy; 0.0 when nothing rhythmic detected)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.atleast_1d(tempo)[0])

        # 4) Spectral centroid (bright/sharp voice vs dull/flat)
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

        # 5) Zero-crossing rate (rough/tense vs smooth voice)
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

        emotion, confidence = _classify(pitch, energy, tempo, centroid, zcr)
        log.info(
            f"voice emotion: {emotion} ({confidence:.0%}) | "
            f"pitch={pitch:.0f}Hz energy={energy:.4f} tempo={tempo:.0f} "
            f"centroid={centroid:.0f} zcr={zcr:.3f}"
        )
        return emotion, confidence

    except Exception as e:
        log.warning(f"voice emotion detection failed: {e}")
        return "neutral", 0.0


def _classify(
    pitch: float, energy: float, tempo: float, centroid: float, zcr: float
) -> tuple[str, float]:
    """
    Rule-based mapping from acoustic features → emotion scores.
    Thresholds are sensible defaults for adult speech at 16kHz; tune freely
    once you've seen real readings in the logs.
    """
    scores = {"happy": 0.0, "angry": 0.0, "sad": 0.0, "calm": 0.0, "neutral": 0.3}

    # Pitch — high pitch leans happy/angry, low leans sad/calm
    if pitch > 200:
        scores["happy"] += 0.3
        scores["angry"] += 0.2
    elif 0 < pitch < 120:
        scores["sad"] += 0.3
        scores["calm"] += 0.2
    else:
        scores["neutral"] += 0.2

    # Energy — loud leans angry/happy, quiet leans sad/calm
    if energy > 0.06:
        scores["angry"] += 0.3
        scores["happy"] += 0.2
    elif energy < 0.015:
        scores["sad"] += 0.3
        scores["calm"] += 0.2
    else:
        scores["neutral"] += 0.1

    # Tempo — fast speech leans happy/angry, slow leans sad/calm
    if tempo > 140:
        scores["happy"] += 0.2
        scores["angry"] += 0.1
    elif 0 < tempo < 80:
        scores["sad"] += 0.2
        scores["calm"] += 0.1

    # Voice brightness — sharp leans angry/happy, dull leans sad
    if centroid > 3000:
        scores["angry"] += 0.15
        scores["happy"] += 0.1
    elif centroid < 1500:
        scores["sad"] += 0.15

    # Voice roughness — tense leans angry, smooth leans calm
    if zcr > 0.12:
        scores["angry"] += 0.1
    elif zcr < 0.04:
        scores["calm"] += 0.1

    # key=lambda (not scores.get) so the checker sees a float, never float|None
    emotion = max(scores, key=lambda k: scores[k])
    total = sum(scores.values())
    confidence = round(scores[emotion] / total, 2) if total > 0 else 0.0
    return emotion, confidence