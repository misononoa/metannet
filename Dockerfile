FROM python:3.12-slim

# sounddevice の PortAudio は PulseAudio API を持たない (ALSA/OSS/JACK のみ) ため、
# ALSA の pulse プラグイン経由でホストの PipeWire/PulseAudio ソケットへ繋ぐ。
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libportaudio2 \
        libasound2-plugins \
        libpulse0 \
    && rm -rf /var/lib/apt/lists/*

# ALSA の既定デバイスを pulse に向ける
RUN printf 'pcm.!default { type pulse }\nctl.!default { type pulse }\n' > /etc/asound.conf

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# 依存だけ先に解決してレイヤキャッシュを効かせる
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

# 本体
COPY metannet ./metannet
COPY main.py ./
RUN uv sync --frozen

CMD ["uv", "run", "main.py"]
