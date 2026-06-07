import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SAMPLE_RATE = 16000  # Whisper / silero-vad はどちらも 16kHz mono を前提とする
DEFAULT_TOML = Path("metannet.toml")


@dataclass
class Config:
    # --- 音声入力 ---
    input_device: int | str | None = None  # None なら既定の入力デバイス
    output_device: int | str | None = None  # None なら既定の出力デバイス

    # --- VAD (発話区間検出) ---
    vad_threshold: float = 0.5  # この確率以上を「発話」とみなす
    min_speech_ms: int = 250  # これより短い区間は誤検出として捨てる
    min_silence_ms: int = 600  # この長さ無音が続いたら発話の区切りとする
    speech_pad_ms: int = 200  # 発話前後に残す余白
    max_speech_ms: int = 15000  # 区切りが来なくても強制的に確定させる上限

    # --- Whisper ---
    model_name: str = "small"
    language: str = "ja"
    device: str | None = None  # None なら CUDA があれば CUDA

    # --- VOICEVOX ---
    voicevox_url: str = "http://127.0.0.1:50021"
    speaker: int = 1  # 話者ID (1 = 四国めたん ノーマル)
    speed_scale: float = 1.0
    pitch_scale: float = 0.0
    timeout: float = 30.0

    # 認識テキストはあるが読み上げはしたくない場合に True
    transcribe_only: bool = False

    @classmethod
    def from_toml(cls, path: Path = DEFAULT_TOML) -> "Config":
        if not path.exists():
            return cls()
        with path.open("rb") as f:
            raw = tomllib.load(f)

        audio = raw.get("audio", {})
        vad = raw.get("vad", {})
        whisper = raw.get("whisper", {})
        vv = raw.get("voicevox", {})
        app = raw.get("app", {})

        kw: dict[str, Any] = {}

        for key in ("input_device", "output_device"):
            if key in audio:
                kw[key] = audio[key]

        for toml_key, cfg_key in (
            ("threshold", "vad_threshold"),
            ("min_speech_ms", "min_speech_ms"),
            ("min_silence_ms", "min_silence_ms"),
            ("speech_pad_ms", "speech_pad_ms"),
            ("max_speech_ms", "max_speech_ms"),
        ):
            if toml_key in vad:
                kw[cfg_key] = vad[toml_key]

        if "model" in whisper:
            kw["model_name"] = whisper["model"]
        for key in ("language", "device"):
            if key in whisper:
                kw[key] = whisper[key]

        if "url" in vv:
            kw["voicevox_url"] = vv["url"]
        for key in ("speaker", "speed_scale", "pitch_scale", "timeout"):
            if key in vv:
                kw[key] = vv[key]

        if "transcribe_only" in app:
            kw["transcribe_only"] = app["transcribe_only"]

        return cls(**kw)
