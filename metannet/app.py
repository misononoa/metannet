import argparse
import queue
import signal
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd

from .config import SAMPLE_RATE, Config
from .segmenter import SileroSegmenter
from .transcriber import WhisperTranscriber
from .voicevox import VoicevoxClient, VoicevoxError

# マイクのブロックサイズ。silero の窓(512)の倍数にしておくと詰め直しが素直。
BLOCK_SIZE = 1024
_SENTINEL = object()


class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        self._stop = threading.Event()

        self._raw_q: queue.Queue[Any] = queue.Queue()
        self._utter_q: queue.Queue[Any] = queue.Queue()
        self._text_q: queue.Queue[Any] = queue.Queue()

        self.segmenter = SileroSegmenter(
            threshold=config.vad_threshold,
            min_speech_ms=config.min_speech_ms,
            min_silence_ms=config.min_silence_ms,
            speech_pad_ms=config.speech_pad_ms,
            max_speech_ms=config.max_speech_ms,
        )
        self.transcriber = WhisperTranscriber(
            model_name=config.model_name,
            language=config.language,
            device=config.device,
        )
        self.voicevox: VoicevoxClient | None = None
        if not config.transcribe_only:
            self.voicevox = VoicevoxClient(
                base_url=config.voicevox_url,
                speaker=config.speaker,
                speed_scale=config.speed_scale,
                pitch_scale=config.pitch_scale,
                timeout=config.timeout,
            )

    # --- スレッド本体 -----------------------------------------------------
    def _segment_loop(self) -> None:
        while True:
            item = self._raw_q.get()
            if item is _SENTINEL:
                break
            for utterance in self.segmenter.process(item):
                self._utter_q.put(utterance)
        tail = self.segmenter.flush()
        if tail is not None:
            self._utter_q.put(tail)
        self._utter_q.put(_SENTINEL)

    def _transcribe_loop(self) -> None:
        while True:
            item = self._utter_q.get()
            if item is _SENTINEL:
                break
            text = self.transcriber.transcribe(item)
            if not _is_meaningful(text):
                continue
            print(f"🗣  {text}", flush=True)
            self._text_q.put(text)
        self._text_q.put(_SENTINEL)

    def _tts_loop(self) -> None:
        while True:
            item = self._text_q.get()
            if item is _SENTINEL:
                break
            if self.voicevox is None:
                continue
            try:
                audio, sr = self.voicevox.synthesize(item)
            except (VoicevoxError, OSError) as e:
                print(f"⚠ VOICEVOX 合成に失敗: {e}", file=sys.stderr, flush=True)
                continue
            try:
                sd.play(audio, sr, device=self.config.output_device)
                sd.wait()
            except sd.PortAudioError as e:
                print(f"⚠ 再生に失敗: {e}", file=sys.stderr, flush=True)

    # --- 起動 / 停止 -----------------------------------------------------
    def run(self) -> None:
        if self.voicevox is not None:
            ver = self._wait_for_voicevox()
            if ver is None:
                print(
                    f"⚠ VOICEVOX ENGINE に接続できません ({self.config.voicevox_url})。\n"
                    "  ENGINE を起動するか、--transcribe-only で認識のみ実行してください。",
                    file=sys.stderr,
                )
                return
            print(f"VOICEVOX ENGINE {ver} / speaker={self.config.speaker}")

        threads = [
            threading.Thread(target=self._segment_loop, daemon=True),
            threading.Thread(target=self._transcribe_loop, daemon=True),
            threading.Thread(target=self._tts_loop, daemon=True),
        ]
        for t in threads:
            t.start()

        def callback(indata, frames, time_info, status):
            if status:
                print(f"⚠ input status: {status}", file=sys.stderr)
            # mono に落としてコピー(callback の buffer は使い回されるため)
            self._raw_q.put(indata[:, 0].copy())

        # Ctrl-C(SIGINT)でも、podman stop 等の SIGTERM でも同じ経路で
        # 停止フラグを立て、各段を流し切ってから抜ける。
        def _handle_signal(signum, frame):
            self._stop.set()

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            dtype="float32",
            device=self.config.input_device,
            callback=callback,
        ):
            print("🎤 待機中… 話してください (Ctrl-C で終了)", flush=True)
            while not self._stop.is_set():
                sd.sleep(200)
            print("\n終了します…", flush=True)

        # 入力停止 → 各段を順に流し切る
        self._raw_q.put(_SENTINEL)
        for t in threads:
            t.join()

    def _wait_for_voicevox(self, attempts: int = 60, interval: float = 1.0) -> str | None:
        # コンテナ同時起動時、ENGINE の起動完了まで少し待つ
        assert self.voicevox is not None
        for i in range(attempts):
            ver = self.voicevox.version()
            if ver is not None:
                return ver
            if i == 0:
                print(
                    f"VOICEVOX ENGINE の起動を待っています ({self.config.voicevox_url}) …",
                    flush=True,
                )
            if self._stop.is_set():
                break
            time.sleep(interval)
        return None


def _is_meaningful(text: str) -> bool:
    stripped = text.strip(" 　。、.,!?！？…・「」\"'\n\t")
    return len(stripped) > 0


def _apply_overrides(base: Config, ns: argparse.Namespace) -> Config:
    kw: dict[str, Any] = {}
    if ns.input_device is not None:
        kw["input_device"] = ns.input_device
    if ns.output_device is not None:
        kw["output_device"] = ns.output_device
    if ns.model is not None:
        kw["model_name"] = ns.model
    if ns.language is not None:
        kw["language"] = ns.language
    if ns.device is not None:
        kw["device"] = ns.device
    if ns.voicevox_url is not None:
        kw["voicevox_url"] = ns.voicevox_url
    if ns.speaker is not None:
        kw["speaker"] = ns.speaker
    if ns.speed is not None:
        kw["speed_scale"] = ns.speed
    if ns.pitch is not None:
        kw["pitch_scale"] = ns.pitch
    if ns.vad_threshold is not None:
        kw["vad_threshold"] = ns.vad_threshold
    if ns.min_silence_ms is not None:
        kw["min_silence_ms"] = ns.min_silence_ms
    if ns.max_speech_ms is not None:
        kw["max_speech_ms"] = ns.max_speech_ms
    if ns.transcribe_only:
        kw["transcribe_only"] = True
    return replace(base, **kw)


def main() -> None:
    parser = argparse.ArgumentParser(prog="metannet")
    parser.add_argument("--config", default="metannet.toml", metavar="FILE", help="設定ファイルのパス")
    parser.add_argument("--list-devices", action="store_true", help="音声デバイス一覧を表示")
    parser.add_argument("--input-device", default=None, help="入力デバイス (番号 or 名前)")
    parser.add_argument("--output-device", default=None, help="出力デバイス (番号 or 名前)")
    parser.add_argument("--model", default=None, help="Whisper モデル名")
    parser.add_argument("--language", default=None)
    parser.add_argument("--device", default=None, help="cuda / cpu (既定は自動)")
    parser.add_argument("--voicevox-url", default=None)
    parser.add_argument("--speaker", type=int, default=None, help="VOICEVOX 話者ID")
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument("--pitch", type=float, default=None)
    parser.add_argument("--vad-threshold", type=float, default=None)
    parser.add_argument("--min-silence-ms", type=int, default=None)
    parser.add_argument("--max-speech-ms", type=int, default=None)
    parser.add_argument(
        "--transcribe-only", action="store_true", default=None, help="読み上げをせず認識結果のみ表示"
    )
    ns = parser.parse_args()

    if ns.list_devices:
        print(sd.query_devices())
        return

    # 数字で渡されたデバイス指定は int に変換
    for attr in ("input_device", "output_device"):
        val = getattr(ns, attr)
        if isinstance(val, str) and val.isdigit():
            setattr(ns, attr, int(val))

    base = Config.from_toml(Path(ns.config))
    Pipeline(_apply_overrides(base, ns)).run()


if __name__ == "__main__":
    main()
