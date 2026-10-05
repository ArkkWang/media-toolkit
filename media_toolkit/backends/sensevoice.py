"""SenseVoice INT8 CPU 转录；每个文件独立 VAD，复用识别模型。

输入为连续的 16 kHz 单声道 float32 音频块，不接受带重叠的分片。
第三方模型与运行库来源见 THIRD_PARTY_NOTICES.md。
"""
from pathlib import Path
from collections.abc import Iterable

import numpy as np

SAMPLE_RATE = 16000


class SenseVoiceBackend:
    def __init__(self, model_dir: str | Path, *, num_threads: int):
        if type(num_threads) is not int or num_threads < 1:
            raise ValueError("num_threads 必须是正整数")
        directory = Path(model_dir).expanduser().resolve()
        for name in ("model.int8.onnx", "tokens.txt", "silero_vad.onnx"):
            if not (directory / name).is_file():
                raise FileNotFoundError(f"SenseVoice 模型文件不存在：{directory / name}")
        import sherpa_onnx
        self._sherpa = sherpa_onnx
        self._vad_config = sherpa_onnx.VadModelConfig()
        self._vad_config.silero_vad.model = str(directory / "silero_vad.onnx")
        self._vad_config.silero_vad.min_silence_duration = 0.5
        self._vad_config.silero_vad.min_speech_duration = 0.25
        # 这是 VAD 的切分目标，不是精确上限；由停顿位置决定实际段长。
        self._vad_config.silero_vad.max_speech_duration = 25.0
        self._vad_config.sample_rate = SAMPLE_RATE
        self._vad_config.num_threads = 1
        self._vad_config.provider = "cpu"
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(directory / "model.int8.onnx"),
            tokens=str(directory / "tokens.txt"),
            num_threads=num_threads, provider="cpu", language="auto", use_itn=True,
        )

    def transcribe_chunks(self, chunks: Iterable[np.ndarray]) -> str:
        """完整消费一个文件才返回文本；失败不返回部分结果。由 Transcriber 串行调用。"""
        if self._recognizer is None:
            raise RuntimeError("SenseVoiceBackend 已关闭")
        # VAD 包含位置、缓存和神经网络状态，不能跨文件保留。
        vad = self._sherpa.VoiceActivityDetector(self._vad_config, buffer_size_in_seconds=60)
        window = self._vad_config.silero_vad.window_size
        pending = np.empty(0, dtype=np.float32)
        pieces = []

        def drain():
            while not vad.empty():
                samples = np.asarray(vad.front.samples, dtype=np.float32).copy()
                vad.pop()
                stream = self._recognizer.create_stream()
                stream.accept_waveform(SAMPLE_RATE, samples)
                self._recognizer.decode_stream(stream)
                text = stream.result.text.strip()
                if text:
                    pieces.append(text)

        for audio in chunks:
            if not isinstance(audio, np.ndarray) or audio.dtype != np.float32:
                raise TypeError("音频必须为 float32 NumPy 数组")
            if audio.ndim != 1 or not np.isfinite(audio).all():
                raise ValueError("音频必须为有限值的单声道采样")
            pending = np.concatenate((pending, audio))
            consumed = 0
            while consumed + window <= len(pending):
                vad.accept_waveform(pending[consumed:consumed + window])
                consumed += window
                drain()
            pending = pending[consumed:].copy()
        if len(pending):
            vad.accept_waveform(np.pad(pending, (0, window - len(pending))))
        vad.flush()
        drain()
        # 没有重叠音频，不做模糊去重，以免误删真实重复说话。
        return "\n".join(pieces)

    def close(self):
        """释放识别模型；VAD 是每次调用的局部资源。允许重复关闭。"""
        self._recognizer = None
