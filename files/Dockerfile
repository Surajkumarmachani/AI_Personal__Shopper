# Backend image for Google Cloud Run (also runs anywhere Docker does).
# Build + deploy:  see "Deploy to Google Cloud" in the repo README.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libglib/libgl are needed by OpenCV (via DeepFace); ffmpeg by librosa/av for
# decoding browser-recorded webm audio.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt \
    && python -m spacy download en_core_web_lg

COPY . .

# Download the DeepFace emotion weights at build time so startup doesn't
# depend on GitHub being reachable.
RUN python -c "from emotion import warm_up; warm_up()"

# Cloud Run sends traffic to $PORT (8080 by default).
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
