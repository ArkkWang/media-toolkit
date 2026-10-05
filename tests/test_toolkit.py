"""不依赖模型的自动测试；包含真实 FFmpeg 解码。"""
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from contextlib import contextmanager

import numpy as np

from media_toolkit import Transcriber, MediaDecodeError
from media_toolkit.audio import _chunks, audio_chunks, ffmpeg_executable, SAMPLE_RATE
from media_toolkit.cli import main
from media_toolkit.text import merge_by_text


class ChunkTests(unittest.TestCase):
    def test_overlap_and_tail(self):
        data = np.arange(19, dtype="<f4")
        chunks = list(_chunks(io.BytesIO(data.tobytes()), 8, 2))
        self.assertEqual([x.tolist() for x in chunks],
                         [list(range(8)), list(range(6, 14)), list(range(12, 19))])

    def test_exact_boundary_no_duplicate_tail(self):
        data = np.arange(14, dtype="<f4")
        self.assertEqual(len(list(_chunks(io.BytesIO(data.tobytes()), 8, 2))), 2)

    def test_short_reads(self):
        class ShortReader(io.BytesIO):
            def read(self, size):
                return super().read(min(3, size))
        data = np.arange(9, dtype="<f4")
        chunks = list(_chunks(ShortReader(data.tobytes()), 8, 2))
        self.assertEqual(chunks[-1].tolist(), [6, 7, 8])

    def test_partial_sample_raises(self):
        with self.assertRaises(MediaDecodeError):
            list(_chunks(io.BytesIO(b"abc"), 8, 2))


class MergeTests(unittest.TestCase):
    def test_overlap(self):
        self.assertEqual(merge_by_text("今天我们一起学习，", "我们一起学习，下一部分。"),
                         "今天我们一起学习，下一部分。")

    def test_no_overlap_preserves_both(self):
        self.assertEqual(merge_by_text("第一段。", "完全不同。"), "第一段。\n完全不同。")

    def test_empty(self):
        self.assertEqual(merge_by_text("", "你好。"), "你好。")
        self.assertEqual(merge_by_text("你好。", ""), "你好。")


class TranscriberTests(unittest.TestCase):
    def test_reuse_result_and_close(self):
        class Backend:
            closed = False
            def recognize(self, audio):
                return "测试结果。"
            def close(self):
                self.closed = True
        @contextmanager
        def fake_chunks(path):
            yield iter([np.ones(1600, dtype=np.float32)])
        backend = Backend()
        with patch("media_toolkit.transcriber.audio_chunks", fake_chunks):
            with Transcriber("models/test") as transcriber:
                transcriber._backend = backend
                self.assertEqual(transcriber.transcribe("test.mp4"), "测试结果。")
                self.assertEqual(transcriber.transcribe("test.mp4"), "测试结果。")
            self.assertTrue(backend.closed)
            transcriber.close()
            with self.assertRaises(RuntimeError):
                transcriber.transcribe("test.mp4")

    def test_missing_file(self):
        with Transcriber("missing-model") as transcriber:
            with self.assertRaises(FileNotFoundError):
                transcriber.transcribe("does-not-exist.mp4")
            self.assertIsNone(transcriber._backend)

    def test_error_releases_lock_and_decoder(self):
        closed = []
        @contextmanager
        def fake_chunks(path):
            try:
                yield iter([np.ones(10, dtype=np.float32)])
            finally:
                closed.append(True)
        class Backend:
            def recognize(self, audio):
                raise RuntimeError("推理失败")
            def close(self):
                pass
        with patch("media_toolkit.transcriber.audio_chunks", fake_chunks):
            with Transcriber("unused") as transcriber:
                transcriber._backend = Backend()
                with self.assertRaisesRegex(RuntimeError, "推理失败"):
                    transcriber.transcribe("unused")
                self.assertFalse(transcriber._lock.locked())
        self.assertEqual(closed, [True])


class FFmpegTests(unittest.TestCase):
    def test_real_video(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "测试视频.mkv"
            subprocess.run([
                ffmpeg_executable(), "-v", "error", "-f", "lavfi", "-i",
                "color=c=black:s=32x32:r=1:d=2", "-f", "lavfi", "-i",
                "sine=frequency=440:sample_rate=44100:duration=2",
                "-c:v", "ffv1", "-c:a", "pcm_s16le", "-shortest", str(video),
            ], check=True)
            with audio_chunks(video) as chunks:
                audio = list(chunks)
            self.assertEqual(len(audio), 1)
            self.assertEqual(len(audio[0]), 2 * SAMPLE_RATE)
            self.assertEqual(audio[0].dtype, np.float32)

    def test_corrupt_file(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "bad.mp4"
            video.write_bytes(b"not a video")
            with self.assertRaises(MediaDecodeError):
                with audio_chunks(video) as chunks:
                    list(chunks)

    def test_no_audio_track(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "silent.mkv"
            subprocess.run([ffmpeg_executable(), "-v", "error", "-f", "lavfi",
                            "-i", "color=s=32x32:r=1:d=1", "-c:v", "ffv1", str(video)], check=True)
            with self.assertRaises(MediaDecodeError):
                with audio_chunks(video) as chunks:
                    list(chunks)


class CliTests(unittest.TestCase):
    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.txt"
            output.write_text("保留", encoding="utf-8")
            with self.assertRaises(SystemExit) as raised:
                main(["unused.mp4", "-o", str(output)])
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(output.read_text(encoding="utf-8"), "保留")

    def test_output_is_optional(self):
        with patch("media_toolkit.cli.Transcriber") as cls:
            cls.return_value.__enter__.return_value.transcribe.return_value = "你好。"
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                self.assertEqual(main(["file.mp4"]), 0)
                self.assertEqual(stdout.getvalue(), "你好。\n")


if __name__ == "__main__":
    unittest.main()
