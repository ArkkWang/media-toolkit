"""独立 Qwen3-ASR：CPU ONNX 编码器与随附 llama.cpp 解码器。

输入必须为 16 kHz 单声道 float32 PCM，每次不超过 60 秒。
模型路径显式提供；不下载模型，也不导入 CapsWriter-Offline。
"""
from __future__ import annotations

import codecs
from pathlib import Path
import threading

import numpy as np

from ._qwen.prompt import PromptBuilder

_NATIVE_INIT_LOCK = threading.Lock()


class QwenBackend(PromptBuilder):
    MAX_AUDIO_SECONDS = 60
    SAMPLE_RATE = 16000
    MAX_NEW_TOKENS = 512
    CONTEXT_SIZE = 2048

    def __init__(self, model_dir, use_gpu=True):
        self._lock = threading.RLock()
        self._closed = True
        self.encoder = self.model = self.ctx = self.embedding_table = None
        model_dir = Path(model_dir).expanduser().resolve()
        paths = [model_dir / name for name in (
            "qwen3_asr_encoder_frontend.onnx",
            "qwen3_asr_encoder_backend.onnx",
            "qwen3_asr_llm.gguf",
        )]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Qwen model file not found: {path}")
        from ._qwen.encoder import QwenAudioEncoder
        from ._qwen import llama
        self._llama = llama
        try:
            # 安装 CPU 版 ORT 即可；GPU 开关仅控制 GGUF 解码器。
            self.encoder = QwenAudioEncoder(str(paths[0]), str(paths[1]),
                                            onnx_provider="CPU", dml_pad_to=0,
                                            verbose=False)
            with _NATIVE_INIT_LOCK:
                self.model = llama.LlamaModel(str(paths[2]), use_gpu=use_gpu)
            self.embedding_table = llama.get_token_embeddings_gguf(str(paths[2]))
            self.ctx = llama.LlamaContext(self.model, n_ctx=self.CONTEXT_SIZE,
                                          n_batch=4096, embeddings=False,
                                          offload_kqv=bool(use_gpu))
            for name, token in (
                ("ID_IM_START", "<|im_start|>"), ("ID_IM_END", "<|im_end|>"),
                ("ID_AUDIO_START", "<|audio_start|>"),
                ("ID_AUDIO_END", "<|audio_end|>"), ("ID_ASR_TEXT", "<asr_text>"),
            ):
                ids = self.model.tokenize(token)
                if len(ids) != 1:
                    raise ValueError(f"Model lacks Qwen ASR special token: {token}")
                setattr(self, name, ids[0])
            self._closed = False
        except BaseException:
            self.close()
            raise

    def recognize(self, audio: np.ndarray) -> str:
        with self._lock:
            if self._closed:
                raise RuntimeError("QwenBackend is closed")
            if not isinstance(audio, np.ndarray) or audio.dtype != np.float32:
                raise TypeError("audio must be a numpy float32 array (16 kHz mono)")
            if audio.ndim != 1:
                raise ValueError("audio must be mono (one-dimensional)")
            if audio.size > self.MAX_AUDIO_SECONDS * self.SAMPLE_RATE:
                raise ValueError("QwenBackend accepts at most 60 seconds per call; split longer audio")
            if not np.isfinite(audio).all():
                raise ValueError("audio contains NaN or infinity")
            if not audio.size:
                return ""
            features, _ = self.encoder.encode(np.ascontiguousarray(audio))
            full_embd = self._build_prompt_embd(features, "", None, None)
            if len(full_embd) + self.MAX_NEW_TOKENS > self.CONTEXT_SIZE:
                raise ValueError("Audio prompt exceeds decoder context capacity")
            # 保留上游的重复熔断重试，最多四次；失败抛异常，
            # 不把诊断标记或静默截断的文本当作成功结果。
            for temperature in (0.4, 0.7, 1.0, 1.3):
                text = self._decode(full_embd, temperature)
                if text is not None:
                    return text.strip()
            raise RuntimeError("Qwen decoder repeatedly generated a token loop")

    def _decode(self, full_embd, temperature):
        """复用上游 embedding 预填充、M-RoPE 和采样循环，去掉流式输出。"""
        llama = self._llama
        total_len = len(full_embd)
        positions = np.arange(total_len, dtype=np.int32)
        positions = np.concatenate([positions, positions, positions,
                                    np.zeros(total_len, dtype=np.int32)])
        batch = sampler = None
        try:
            # 上游使用四组位置坐标，分配足够的位置存储空间。
            batch = llama.LlamaBatch(max(total_len * 4, 8192), self.model.n_embd, 1)
            batch.set_embd(full_embd, pos=positions)
            self.ctx.clear_kv_cache()
            code = self.ctx.decode(batch)
            if code != 0:
                raise RuntimeError(f"llama prefill failed (code {code})")
            batch.close()
            batch = None
            sampler = llama.LlamaSampler(temperature=temperature)
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            tokens, pieces = [], []
            for _ in range(self.MAX_NEW_TOKENS):
                token = sampler.sample(self.ctx.ptr)
                if token in (self.model.eos_token, self.ID_IM_END):
                    pieces.append(decoder.decode(b"", final=True))
                    return "".join(pieces)
                code = self.ctx.decode_token(token)
                if code != 0:
                    raise RuntimeError(f"llama token decode failed (code {code})")
                tokens.append(token)
                pieces.append(decoder.decode(self.model.token_to_bytes(token)))
                if len(tokens) > 15 and len(set(tokens[-15:])) <= 3:
                    return None
            raise RuntimeError("Qwen exceeded 512 output tokens; use shorter audio chunks")
        finally:
            if sampler is not None:
                sampler.free()
            if batch is not None:
                batch.close()
            self.ctx.clear_kv_cache()

    def close(self):
        """依次释放上下文、模型、内存映射和 ONNX 会话；允许重复调用。"""
        with self._lock:
            self._closed = True
            for name in ("ctx", "model", "embedding_table"):
                resource = getattr(self, name, None)
                if resource is not None:
                    resource.close()
                    setattr(self, name, None)
            if self.encoder is not None:
                self.encoder.sess_fe = self.encoder.sess_be = None
                self.encoder = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        if hasattr(self, "_lock"):
            self.close()
