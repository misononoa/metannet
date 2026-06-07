import numpy as np
import torch
import whisper


class WhisperTranscriber:
    def __init__(self, model_name: str, language: str, device: str | None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = whisper.load_model(model_name, device=self.device)
        self.language = language
        self.fp16 = self.device == "cuda"

    def transcribe(self, audio: np.ndarray) -> str:
        # Whisper は 16kHz / float32 / [-1, 1] 正規化済みの ndarray をそのまま受ける
        result = self.model.transcribe(
            audio,
            language=self.language,
            fp16=self.fp16,
            condition_on_previous_text=False,  # 直前テキストへの引きずられ(幻聴)を抑える
        )
        # セグメント単位で信頼度の低いものを捨て、無音区間の幻聴を抑える
        kept = [
            seg["text"]
            for seg in result.get("segments", [])
            if not _is_hallucination(seg)
        ]
        return "".join(kept).strip()


def _is_hallucination(seg: dict) -> bool:
    no_speech = seg.get("no_speech_prob", 0.0)
    avg_logprob = seg.get("avg_logprob", 0.0)
    # 「無音らしさが高い」かつ「自信がない」セグメントは幻聴とみなす
    return no_speech > 0.6 and avg_logprob < -0.5
