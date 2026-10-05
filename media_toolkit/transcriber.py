"""独立文件转录入口，不导入 UI、不联网、不自动保存结果。"""
from pathlib import Path
from threading import Lock

from .audio import audio_chunks
from .text import merge_by_text


class Transcriber:
    """拥有一个可复用的本地模型。推荐 with 使用；单实例不支持并发转录。"""

    def __init__(self, model_dir: str | Path, *, use_gpu: bool = True):
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.use_gpu = use_gpu
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
