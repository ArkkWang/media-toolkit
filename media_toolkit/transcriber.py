"""独立文件转录入口，不导入 UI、不联网、不自动保存结果。"""
from pathlib import Path
from threading import Lock
from itertools import chain
import os

from .audio import audio_chunks
from .text import merge_by_text

DEFAULT_MODELS = {
    "sensevoice": Path("models/SenseVoiceSmall-int8"),
    "qwen": Path("models/Qwen3-ASR-1.7B"),
}


class Transcriber:
    """拥有一个可复用的本地模型。推荐 with 使用；单实例不支持并发转录。"""

    def __init__(self, model_dir: str | Path | None = None, *,
                 backend: str = "sensevoice", use_gpu: bool | None = None,
                 num_threads: int | None = None):
        if backend not in DEFAULT_MODELS:
            raise ValueError(f"未知转录后端：{backend}；可选 sensevoice、qwen")
        if use_gpu is not None and type(use_gpu) is not bool:
            raise ValueError("use_gpu 必须为 bool 或 None")
        if num_threads is not None and (type(num_threads) is not int or num_threads < 1):
            raise ValueError("num_threads 必须是正整数")
        if backend == "sensevoice" and use_gpu is True:
            raise ValueError("SenseVoice 后端仅支持 CPU；GPU 转录请选择 backend='qwen'")
        if backend == "qwen" and num_threads is not None:
            raise ValueError("num_threads 目前仅适用于 SenseVoice 后端")
        self.backend = backend
        self.model_dir = Path(model_dir if model_dir is not None else DEFAULT_MODELS[backend]).expanduser().resolve()
        self.use_gpu = (use_gpu is not False) if backend == "qwen" else False
        self.num_threads = num_threads if num_threads is not None else min(12, os.cpu_count() or 1)
        self._backend = None
        self._closed = False
        self._lock = Lock()

    def transcribe(self, path: str | Path) -> str:
        """返回模型生成的带标点文本。损坏媒体或推理失败会抛异常。"""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("同一 Transcriber 正在使用中，请串行调用或使用另一实例")
        try:
            if self._closed:
                raise RuntimeError("Transcriber 已关闭，请创建新实例")
            text = ""
            # 两个后端需要的切分语义不同，不把 VAD 段送进重叠文本合并。
            if self.backend == "sensevoice":
                with audio_chunks(path, overlap_seconds=0) as chunks:
                    first = next(chunks, None)
                    if first is None:
                        return ""
                    if self._backend is None:
                        from .backends.sensevoice import SenseVoiceBackend
                        self._backend = SenseVoiceBackend(self.model_dir, num_threads=self.num_threads)
                    return self._backend.transcribe_chunks(chain((first,), chunks))
            with audio_chunks(path) as chunks:
                for audio in chunks:
                    if self._backend is None:
                        from .backends.qwen import QwenBackend
                        self._backend = QwenBackend(self.model_dir, use_gpu=self.use_gpu)
                    # 原始静音直接略过，避免纯数字静音触发模型幻觉。
                    if not audio.any():
                        continue
                    segment = self._backend.recognize(audio).strip()
                    text = merge_by_text(text, segment)
            return text.strip()
        finally:
            self._lock.release()

    def close(self) -> None:
        """释放模型；重复调用无副作用，不中断其他线程中的推理。"""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("转录仍在进行，不能同时关闭模型")
        try:
            if self._backend is not None:
                self._backend.close()
                self._backend = None
            self._closed = True
        finally:
            self._lock.release()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Transcriber 已关闭，请创建新实例")
        return self

    def __exit__(self, *_):
        self.close()
