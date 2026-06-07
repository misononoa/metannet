# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Streaming voice changer: mic → Whisper STT (Japanese) → VOICEVOX TTS, so a streamer's real voice is replaced by a synthesized one. See README.md for end-user usage.

## Commands

```bash
uv sync                              # set up the venv (CUDA torch, ~several GB)
uv run main.py                       # full pipeline (needs VOICEVOX on :50021)
uv run main.py --transcribe-only     # STT only, no VOICEVOX needed
uv run main.py --list-devices        # enumerate audio devices

podman-compose up                    # run app + VOICEVOX engine together
podman-compose build app             # rebuild after dependency/Dockerfile changes
```

There is no test framework. Logic in `segmenter.py` is verified by stubbing
`SileroSegmenter._speech_prob` with a deterministic probability sequence and
feeding `process()` synthetic windows (no real audio/model needed).

## Architecture

A 4-stage pipeline wired by `queue.Queue`s in `metannet/app.py` (`Pipeline`):

1. **Capture** — `sd.InputStream` callback pushes mono float32 blocks into `_raw_q`.
2. **Segment** (`segmenter.py`) — `SileroSegmenter` cuts utterances → `_utter_q`.
3. **Transcribe** (`transcriber.py`) — Whisper on GPU → text → `_text_q`.
4. **TTS** (`voicevox.py`) — synthesize + `sd.play`.

Each stage is its own thread; shutdown propagates a `_SENTINEL` object down the
queues so every stage drains and joins cleanly. `config.py` holds the single
`Config` dataclass; the CLI in `app.main` maps args onto it.

### segmenter.py — the core, and the part the original author got stuck on

silero-vad at 16kHz requires **exactly 512-sample windows**, so `process()`
re-buffers arbitrary input block sizes into 512-sample windows (`_carry`). A
hysteresis state machine decides utterance boundaries from per-window speech
probability: a ring buffer keeps pre-speech padding, trailing silence beyond
`min_silence_ms` (or `max_speech_ms`) ends an utterance, and the min-length
filter uses the count of **voiced** windows (not padded length) so short noise
blips are rejected. The silero model is stateful — windows are fed in order and
states are not reset mid-stream. The old code's bug was passing float32 arrays
to webrtcvad (which needs 16-bit PCM bytes); webrtcvad was replaced by silero.

### Hallucination filtering

Japanese Whisper emits phantom phrases (e.g. 「ご視聴ありがとうございました」) on
near-silence. `transcriber._is_hallucination` drops segments by
`no_speech_prob`/`avg_logprob`; `app._is_meaningful` drops punctuation-only text.

## Constraints / gotchas

- **Python is pinned to `>=3.12,<3.13`** — 3.14 lacks wheels for the ML/audio stack. Don't bump it.
- Whisper and silero both assume **16kHz mono float32 in [-1, 1]**; sounddevice provides this directly, so audio is never written to disk or re-decoded.
- Container audio (Dockerfile/compose.yaml): the Linux sounddevice wheel bundles no PortAudio, so the image needs `libportaudio2`; PortAudio has no Pulse backend, so `libasound2-plugins`+`libpulse0`+`/etc/asound.conf` route ALSA→PipeWire via the mounted pulse socket.
- podman needs **fully-qualified image names** (`docker.io/...`); Fedora SELinux requires `security_opt: label=disable` for socket/GPU access; GPU is passed via CDI (`devices: nvidia.com/gpu=all`).
- compose passes `--voicevox-url http://voicevox:50021` as a CLI argument (using the Docker service name); the app waits up to 60s for the engine on startup.
