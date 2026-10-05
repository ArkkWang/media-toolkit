"""双后端选择与 SenseVoice 生命周期测试；不加载真实模型。"""
from contextlib import contextmanager
import io
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from media_toolkit import Transcriber
from media_toolkit.backends.sensevoice import SenseVoiceBackend
from media_toolkit.cli import main
from media_toolkit.audio import _chunks, audio_chunks


class SelectionTests(unittest.TestCase):
    def test_defaults_and_explicit_qwen(self):
        t = Transcriber()
        self.assertEqual(t.backend, "sensevoice")
        self.assertFalse(t.use_gpu)
        self.assertEqual(t.model_dir.name, "SenseVoiceSmall-int8")
        self.assertGreaterEqual(t.num_threads, 1)
        self.assertLessEqual(t.num_threads, 12)
        q = Transcriber(backend="qwen")
        self.assertTrue(q.use_gpu)
        self.assertEqual(q.model_dir.name, "Qwen3-ASR-1.7B")
        self.assertFalse(Transcriber(backend="qwen", use_gpu=False).use_gpu)

    def test_invalid_options_fail_before_loading(self):
        for options in ({"backend": "unknown"}, {"num_threads": 0},
                        {"num_threads": -1}, {"num_threads": True},
                        {"num_threads": 1.5}, {"use_gpu": True},
                        {"use_gpu": "false"}, {"backend": "qwen", "num_threads": 4}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Transcriber(**options)

    def test_sensevoice_selection_reuse_and_cleanup(self):
        blocks = [np.ones(4, np.float32), np.ones(3, np.float32)]
        closed = []
        @contextmanager
        def fake_chunks(path, *, overlap_seconds):
            self.assertEqual(overlap_seconds, 0)
            try:
                yield iter(blocks)
            finally:
                closed.append(True)
        backend = Mock()
        def consume(chunks):
            self.assertEqual([len(x) for x in chunks], [4, 3])
            return "完整文本。"
        backend.transcribe_chunks.side_effect = consume
        with patch("media_toolkit.transcriber.audio_chunks", fake_chunks), \
             patch("media_toolkit.backends.sensevoice.SenseVoiceBackend", return_value=backend) as factory, \
             patch("media_toolkit.backends.qwen.QwenBackend") as qwen:
            with Transcriber("models/test", num_threads=8) as transcriber:
                self.assertEqual(transcriber.transcribe("one"), "完整文本。")
                self.assertEqual(transcriber.transcribe("two"), "完整文本。")
            factory.assert_called_once_with(Path("models/test").resolve(), num_threads=8)
            qwen.assert_not_called()
            backend.close.assert_called_once()
        self.assertEqual(closed, [True, True])

    def test_qwen_selection_and_no_automatic_fallback(self):
        @contextmanager
        def fake_chunks(path):
            yield iter([np.ones(4, np.float32)])
        with patch("media_toolkit.transcriber.audio_chunks", fake_chunks), \
             patch("media_toolkit.backends.qwen.QwenBackend") as qwen, \
             patch("media_toolkit.backends.sensevoice.SenseVoiceBackend") as sense:
            qwen.return_value.recognize.return_value = "回退结果。"
            with Transcriber(backend="qwen", use_gpu=False) as t:
                self.assertEqual(t.transcribe("one"), "回退结果。")
                qwen.assert_called_once_with(t.model_dir, use_gpu=False)
                qwen.return_value.recognize.side_effect = RuntimeError("模型失败")
                with self.assertRaisesRegex(RuntimeError, "模型失败"):
                    t.transcribe("two")
                self.assertFalse(t._lock.locked())
            sense.assert_not_called()

    def test_sensevoice_error_closes_audio_and_does_not_fallback(self):
        closed = []
        @contextmanager
        def broken_audio(path, **kwargs):
            def blocks():
                yield np.ones(4, np.float32)
                raise RuntimeError("音频尾部损坏")
            try:
                yield blocks()
            finally:
                closed.append(True)
        backend = Mock()
        backend.transcribe_chunks.side_effect = lambda chunks: list(chunks)
        with patch("media_toolkit.transcriber.audio_chunks", broken_audio), \
             patch("media_toolkit.backends.sensevoice.SenseVoiceBackend", return_value=backend), \
             patch("media_toolkit.backends.qwen.QwenBackend") as qwen:
            with Transcriber() as t:
                with self.assertRaisesRegex(RuntimeError, "尾部损坏"):
                    t.transcribe("one")
                self.assertFalse(t._lock.locked())
            qwen.assert_not_called()
        self.assertEqual(closed, [True])

    def test_concurrent_call_and_close_are_rejected(self):
        t = Transcriber()
        t._lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, "串行"):
                t.transcribe("unused")
            with self.assertRaisesRegex(RuntimeError, "仍在进行"):
                t.close()
        finally:
            t._lock.release()
        t.close()
        t.close()
        with self.assertRaisesRegex(RuntimeError, "已关闭"):
            t.transcribe("unused")

    def test_continuous_chunks_preserve_all_samples(self):
        data = np.arange(19, dtype="<f4")
        result = np.concatenate(list(_chunks(io.BytesIO(data.tobytes()), 8, 0)))
        np.testing.assert_array_equal(result, data)
        for overlap in (-1, 60, 0.5):
            with self.assertRaises(ValueError), audio_chunks("unused", overlap_seconds=overlap):
                pass


class FakeVad:
    """在 flush 才产生片段，以检查 EOF 和跨音频块的余数。"""
    instances = []

    def __init__(self, config, buffer_size_in_seconds):
        self.windows = []
        self.queue = []
        self.flushed = False
        self.instances.append(self)

    def accept_waveform(self, audio):
        self.windows.append(audio.copy())

    def flush(self):
        self.flushed = True
        samples = np.concatenate(self.windows) if self.windows else np.empty(0)
        if np.any(samples):
            self.queue.append(SimpleNamespace(samples=samples))

    def empty(self):
        return not self.queue

    @property
    def front(self):
        return self.queue[0]

    def pop(self):
        self.queue.pop(0)


class SenseVoiceTests(unittest.TestCase):
    def setUp(self):
        FakeVad.instances = []
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        for name in ("model.int8.onnx", "tokens.txt", "silero_vad.onnx"):
            Path(self.directory.name, name).touch()
        self.recognizer = Mock()
        self.recognizer.create_stream.side_effect = lambda: Mock(result=SimpleNamespace(text="  测试。  "))
        config = SimpleNamespace(silero_vad=SimpleNamespace(window_size=4))
        self.sherpa = SimpleNamespace(
            VadModelConfig=lambda: config,
            VoiceActivityDetector=FakeVad,
            OfflineRecognizer=SimpleNamespace(from_sense_voice=Mock(return_value=self.recognizer)),
        )
        with patch.dict("sys.modules", {"sherpa_onnx": self.sherpa}):
            self.backend = SenseVoiceBackend(self.directory.name, num_threads=12)
        self.addCleanup(self.backend.close)

    def test_cpu_itn_and_thread_configuration(self):
        kwargs = self.sherpa.OfflineRecognizer.from_sense_voice.call_args.kwargs
        self.assertEqual(kwargs['provider'], 'cpu')
        self.assertEqual(kwargs['num_threads'], 12)
        self.assertTrue(kwargs['use_itn'])
        self.assertEqual(self.backend._vad_config.provider, 'cpu')

    def test_tail_flush_and_no_cross_file_state(self):
        audio = np.arange(1, 8, dtype=np.float32)
        for _ in range(2):
            self.assertEqual(self.backend.transcribe_chunks([audio[:3], audio[3:]]), "测试。")
        self.assertEqual(len(FakeVad.instances), 2)
        for vad in FakeVad.instances:
            self.assertTrue(vad.flushed)
            np.testing.assert_array_equal(np.concatenate(vad.windows), np.pad(audio, (0, 1)))
            self.assertTrue(vad.empty())

    def test_empty_and_silence_skip_recognition(self):
        self.assertEqual(self.backend.transcribe_chunks([]), "")
        self.assertEqual(self.backend.transcribe_chunks([np.zeros(7, np.float32)]), "")
        self.recognizer.decode_stream.assert_not_called()

    def test_invalid_audio(self):
        for audio in ([1.0], np.ones(2, np.float64), np.ones((2, 2), np.float32),
                      np.array([np.nan], np.float32), np.array([np.inf], np.float32)):
            with self.subTest(audio=audio), self.assertRaises((TypeError, ValueError)):
                self.backend.transcribe_chunks([audio])

    def test_failed_recognition_does_not_pollute_next_file(self):
        self.recognizer.decode_stream.side_effect = RuntimeError("推理失败")
        with self.assertRaisesRegex(RuntimeError, "推理失败"):
            self.backend.transcribe_chunks([np.ones(5, np.float32)])
        self.recognizer.decode_stream.side_effect = None
        self.assertEqual(self.backend.transcribe_chunks([np.ones(4, np.float32)]), "测试。")
        self.assertEqual(len(FakeVad.instances[-1].windows), 1)

    def test_real_repetitions_are_not_removed(self):
        class RepeatingVad(FakeVad):
            def flush(self):
                super().flush()
                self.queue *= 2
        self.sherpa.VoiceActivityDetector = RepeatingVad
        self.assertEqual(self.backend.transcribe_chunks([np.ones(4, np.float32)]), "测试。\n测试。")

    def test_missing_files_and_invalid_threads(self):
        with self.assertRaises(ValueError):
            SenseVoiceBackend(self.directory.name, num_threads=0)
        for name in ("silero_vad.onnx", "tokens.txt", "model.int8.onnx"):
            Path(self.directory.name, name).unlink()
            with self.assertRaisesRegex(FileNotFoundError, name):
                SenseVoiceBackend(self.directory.name, num_threads=4)

    def test_close_is_idempotent(self):
        self.backend.close()
        self.backend.close()
        self.assertIsNone(self.backend._recognizer)
        with self.assertRaisesRegex(RuntimeError, "已关闭"):
            self.backend.transcribe_chunks([])


class BackendCliTests(unittest.TestCase):
    def test_default_and_fallback_arguments(self):
        cases = [([], 'sensevoice', None, None),
                 (['--threads', '8'], 'sensevoice', None, 8),
                 (['--backend', 'qwen'], 'qwen', None, None),
                 (['--backend', 'qwen', '--cpu'], 'qwen', False, None)]
        for extra, backend, use_gpu, threads in cases:
            with self.subTest(extra=extra), patch('media_toolkit.cli.Transcriber') as cls, \
                 patch('sys.stdout', new_callable=io.StringIO):
                cls.return_value.__enter__.return_value.transcribe.return_value = '完成。'
                self.assertEqual(main(['file.m4a', *extra]), 0)
                cls.assert_called_once_with(None, backend=backend, use_gpu=use_gpu, num_threads=threads)

    def test_invalid_options(self):
        for extra in (['--threads', '0'], ['--backend', 'qwen', '--threads', '8'],
                      ['--backend', 'bad']):
            with self.subTest(extra=extra), patch('sys.stderr', new_callable=io.StringIO), \
                 self.assertRaises(SystemExit) as raised:
                main(['file', *extra])
            self.assertEqual(raised.exception.code, 2)

    def test_failure_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'result.txt'
            with patch('media_toolkit.cli.Transcriber') as cls, \
                 patch('sys.stderr', new_callable=io.StringIO):
                cls.return_value.__enter__.return_value.transcribe.side_effect = RuntimeError('中途失败')
                self.assertEqual(main(['file', '-o', str(output)]), 1)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
