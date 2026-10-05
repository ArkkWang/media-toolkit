"""有限内存解码音视频，输出 16 kHz 单声道 float32 分片。"""
from contextlib import contextmanager
from pathlib import Path
import subprocess
import tempfile
from collections.abc import Iterator

import numpy as np

SAMPLE_RATE = 16_000
CHUNK_SECONDS = 60
OVERLAP_SECONDS = 4


class MediaDecodeError(RuntimeError):
    """媒体无音轨、损坏，或 FFmpeg 解码失败。"""


def ffmpeg_executable() -> str:
    # wheel 自带二进制，不要求用户手工安装 FFmpeg 或配置 PATH。
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _read_up_to(stream, size: int) -> bytes:
    """管道可能短读，不能把一次 read 当作完整分片。"""
    parts = []
    remaining = size
    while remaining:
        data = stream.read(remaining)
        if not data:
            break
        parts.append(data)
        remaining -= len(data)
    return b"".join(parts)


def _chunks(stream, chunk_samples: int, overlap_samples: int) -> Iterator[np.ndarray]:
    data = _read_up_to(stream, chunk_samples * 4)
    while data:
        if len(data) % 4:
            raise MediaDecodeError("FFmpeg 返回的音频数据不完整")
        yield np.frombuffer(data, dtype="<f4").copy()
        if len(data) < chunk_samples * 4:
            return
        extra = _read_up_to(stream, (chunk_samples - overlap_samples) * 4)
        if not extra:
            return  # 不单独重复提交最后的 overlap
        tail = data[-overlap_samples * 4:] if overlap_samples else b""
        data = tail + extra


@contextmanager
def audio_chunks(path: str | Path, *, overlap_seconds: int = OVERLAP_SECONDS) -> Iterator[Iterator[np.ndarray]]:
    """用 with 消费分片；overlap_seconds=0 返回连续音频，供 VAD 使用。"""
    if not isinstance(overlap_seconds, int) or not 0 <= overlap_seconds < CHUNK_SECONDS:
        raise ValueError("分片重叠秒数必须为非负整数且小于分片长度")
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"媒体文件不存在：{source}")
    command = [
        ffmpeg_executable(), "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror",
        "-i", str(source), "-map", "0:a:0", "-vn", "-ac", "1", "-ar",
        str(SAMPLE_RATE), "-f", "f32le", "pipe:1",
    ]
    # stderr 写临时文件，避免管道写满死锁，也不把日志与文本结果混在一起。
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        exhausted = False

        def consume():
            nonlocal exhausted
            found = False
            for chunk in _chunks(process.stdout, CHUNK_SECONDS * SAMPLE_RATE,
                                 overlap_seconds * SAMPLE_RATE):
                found = True
                yield chunk
            code = process.wait()
            exhausted = True
            if code:
                errors.seek(0, 2)
                errors.seek(max(0, errors.tell() - 8000))
                detail = errors.read().decode("utf-8", errors="replace").strip()
                raise MediaDecodeError(f"FFmpeg 解码失败（{code}）：{detail}")
            if not found:
                raise MediaDecodeError("媒体没有可解码的音频采样")

        iterator = consume()
        try:
            yield iterator
        finally:
            iterator.close()
            if not exhausted and process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            process.stdout.close()
