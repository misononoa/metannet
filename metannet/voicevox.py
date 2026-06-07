import io
import wave

import numpy as np
import requests


class VoicevoxError(RuntimeError):
    pass


class VoicevoxClient:
    """VOICEVOX ENGINE (既定で http://127.0.0.1:50021) を叩く薄いクライアント。"""

    def __init__(
        self,
        base_url: str,
        speaker: int,
        speed_scale: float,
        pitch_scale: float,
        timeout: float,
    ):
        self.base_url = base_url.rstrip("/")
        self.speaker = speaker
        self.speed_scale = speed_scale
        self.pitch_scale = pitch_scale
        self.timeout = timeout

    def version(self) -> str | None:
        try:
            r = requests.get(f"{self.base_url}/version", timeout=3)
            r.raise_for_status()
            return r.text.strip().strip('"')
        except requests.RequestException:
            return None

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """テキストを音声合成し、(float32 波形, サンプリングレート) を返す。"""
        # 1) 読み・アクセント等のクエリを生成
        q = requests.post(
            f"{self.base_url}/audio_query",
            params={"text": text, "speaker": self.speaker},
            timeout=self.timeout,
        )
        if q.status_code != 200:
            raise VoicevoxError(f"audio_query failed: {q.status_code} {q.text}")
        query = q.json()
        query["speedScale"] = self.speed_scale
        query["pitchScale"] = self.pitch_scale

        # 2) クエリから wav を合成
        s = requests.post(
            f"{self.base_url}/synthesis",
            params={"speaker": self.speaker},
            headers={"Content-Type": "application/json"},
            json=query,
            timeout=self.timeout,
        )
        if s.status_code != 200:
            raise VoicevoxError(f"synthesis failed: {s.status_code} {s.text}")

        return _decode_wav(s.content)


def _decode_wav(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels)
    return audio, sample_rate
