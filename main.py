from queue import Empty, Queue
from threading import Event, Thread
from typing import Any

import whisper
from numpy import concatenate, float32, ndarray, squeeze
from sounddevice import CallbackFlags, InputStream
from webrtcvad import Vad
from whisper.model import Whisper

VAD_MODE = 2
BLOCK_SIZE = 1024
SAMPLE_RATE = 16000
CHANNELS = 1


class TranscribeThread(Thread):
    def __init__(self, model: Whisper, vad: Vad, language: str, required_samples: int):
        super().__init__(daemon=True)
        self._model = model
        self._vad = vad
        self._language = language
        self._required_samples = required_samples
        self._frame_duration = 30
        self._frame_size = int(SAMPLE_RATE * 2 * self._frame_duration / 1000)

        self._queue: Queue[ndarray] = Queue()
        self._running = Event()

        self._buffer: list[ndarray] = list()
        self._total_samples = 0
        self._last_silence_cnt = 0

    def push(self, audio: ndarray) -> None:
        self._queue.put(audio)

    def run(self) -> None:
        self._running.set()
        while self._running.is_set():
            try:
                chunk = self._queue.get(timeout=0.1)

                for i in range(0, len(chunk) - self._frame_size + 1, self._frame_size):
                    frame = chunk[i : i + self._frame_size]
                    print("frame_size: ", len(frame))
                    if len(frame) == self._frame_size:
                        if self._vad.is_speech(frame, SAMPLE_RATE):
                            self._last_silence_cnt += 1
                            break

                self._buffer.append(chunk)
                self._total_samples += chunk.shape[0]

                if (
                    self._last_silence_cnt > 5
                    or self._total_samples >= self._required_samples
                ):
                    audio_data = concatenate(self._buffer, axis=0)
                    is_speech = False
                    for i in range(
                        0, len(audio_data) - self._frame_size + 1, self._frame_size
                    ):
                        frame = audio_data[i : i + self._frame_size]
                        print("frame_size: ", len(frame))
                        if len(frame) == self._frame_size:
                            if self._vad.is_speech(frame, SAMPLE_RATE):
                                is_speech = True
                                break
                    if is_speech:
                        print(
                            "last_silence: ",
                            self._last_silence_cnt,
                            ", data_size: ",
                            self._total_samples,
                        )
                        result = self._model.transcribe(
                            audio=audio_data, language=self._language, fp16=False
                        )
                        print("result: ", result)

                    self._buffer.clear()
                    self._total_samples = 0
                    self._last_silence_cnt = 0
            except Empty:
                continue

    def stop(self) -> None:
        self._running.clear()
        self.join()
        self._queue.queue.clear()


model = whisper.load_model(name="small")
vad = Vad(mode=VAD_MODE)


def main():
    transcribe_thread = TranscribeThread(
        model=model,
        vad=vad,
        language="ja",
        required_samples=SAMPLE_RATE * 2,
    )

    def cb(
        in_data: ndarray,
        frame_count: int,
        time_info: dict[str, Any],
        status: CallbackFlags,
    ) -> None:
        nonlocal transcribe_thread
        if status:
            print("callback status: ", status)
        transcribe_thread.push(in_data[:, 0].copy())

    with InputStream(
        callback=cb,
        blocksize=BLOCK_SIZE,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        dtype=float32,
    ):
        transcribe_thread.start()
        input("press any key to stop.")


if __name__ == "__main__":
    main()
