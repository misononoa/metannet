"""silero-vad を使った発話区間切り出し。

このツールで一番難しかった「低レベル音声処理」の中心。要点は2つ:

1. silero-vad のモデルは 16kHz では「ちょうど 512 サンプル」の窓を 1 回ずつ
   渡す必要がある。マイクのブロックサイズはまちまちなので、内部で 512
   サンプル単位に詰め直す (carry バッファ)。
2. モデルは連続ストリームを前提とした内部状態を持つので、窓を順番に渡し続け、
   状態をリセットしない。各窓ごとの発話確率からヒステリシス付きの状態機械で
   発話の開始・終了を判定する。
"""

from collections import deque

import numpy as np
import torch
from silero_vad import load_silero_vad

from .config import SAMPLE_RATE

WINDOW = 512  # 16kHz における silero-vad の必須窓サイズ (= 32ms)
WINDOW_MS = 1000 * WINDOW / SAMPLE_RATE


def _ms_to_windows(ms: int) -> int:
    return max(1, round(ms / WINDOW_MS))


class SileroSegmenter:
    def __init__(
        self,
        threshold: float,
        min_speech_ms: int,
        min_silence_ms: int,
        speech_pad_ms: int,
        max_speech_ms: int,
    ):
        self.model = load_silero_vad()
        self.model.reset_states()

        self.threshold = threshold
        # 終了判定はチャタリング防止のため少し低い閾値を使う
        self.neg_threshold = max(0.15, threshold - 0.15)

        self.min_speech_windows = _ms_to_windows(min_speech_ms)
        self.min_silence_windows = _ms_to_windows(min_silence_ms)
        self.pad_windows = _ms_to_windows(speech_pad_ms)
        self.max_windows = _ms_to_windows(max_speech_ms)

        # 発話前の余白を確保するためのリングバッファ
        self._pre: deque[np.ndarray] = deque(maxlen=self.pad_windows)
        self._carry = np.empty(0, dtype=np.float32)
        self._reset_utterance()

    def _reset_utterance(self) -> None:
        self._triggered = False
        self._speech: list[np.ndarray] = []
        self._silence_windows = 0
        self._voiced_windows = 0  # 実際に「発話」と判定された窓数 (パディング除く)

    def _speech_prob(self, window: np.ndarray) -> float:
        with torch.no_grad():
            tensor = torch.from_numpy(window)
            return self.model(tensor, SAMPLE_RATE).item()

    def process(self, samples: np.ndarray) -> list[np.ndarray]:
        """任意長の float32 サンプルを受け取り、確定した発話の配列を返す。"""
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        self._carry = np.concatenate((self._carry, samples))

        utterances: list[np.ndarray] = []
        while self._carry.shape[0] >= WINDOW:
            window = self._carry[:WINDOW]
            self._carry = self._carry[WINDOW:]
            done = self._feed_window(window)
            if done is not None:
                utterances.append(done)
        return utterances

    def _feed_window(self, window: np.ndarray) -> np.ndarray | None:
        prob = self._speech_prob(window)

        if not self._triggered:
            self._pre.append(window)
            if prob >= self.threshold:
                # 発話開始: 直前の余白ごと取り込む
                self._triggered = True
                self._speech = list(self._pre)
                self._pre.clear()
                self._silence_windows = 0
            return None

        # 発話中
        self._speech.append(window)
        if prob < self.neg_threshold:
            self._silence_windows += 1
        else:
            self._silence_windows = 0
            self._voiced_windows += 1

        if self._silence_windows >= self.min_silence_windows:
            # 無音で区切れた場合は、末尾の無音を余白ぶんだけ残して切り詰める
            return self._finish(trim_trailing=self._silence_windows)
        if len(self._speech) >= self.max_windows:
            return self._finish(trim_trailing=0)
        return None

    def _finish(self, trim_trailing: int) -> np.ndarray | None:
        speech = self._speech
        voiced = self._voiced_windows
        self._reset_utterance()

        if voiced < self.min_speech_windows:
            return None
        keep = len(speech) - max(0, trim_trailing - self.pad_windows)
        return np.concatenate(speech[:keep])

    def flush(self) -> np.ndarray | None:
        """ストリーム終了時に、進行中の発話があれば確定して返す。"""
        if self._triggered and self._voiced_windows >= self.min_speech_windows:
            audio = np.concatenate(self._speech)
            self._reset_utterance()
            return audio
        self._reset_utterance()
        return None
