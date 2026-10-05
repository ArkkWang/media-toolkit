"""后端契约测试：使用 mock，不加载模型或原生 DLL。"""
import tempfile
import threading
import unittest
from unittest.mock import Mock

import numpy as np

from media_toolkit.backends.qwen import QwenBackend


class QwenBackendTests(unittest.TestCase):
    def setUp(self):
        self.backend = QwenBackend.__new__(QwenBackend)
        b = self.backend
        b._lock = threading.RLock()
        b._closed = False
        b.encoder = Mock()
        b.encoder.encode.return_value = (np.zeros((13, 4), np.float32), 0.0)
        b.model = Mock(n_embd=4, eos_token=99)
        b.ctx = Mock()
        b.embedding_table = Mock()
        b._build_prompt_embd = Mock(return_value=np.zeros((33, 4), np.float32))
        b._decode = Mock(return_value="  测试结果  ")
        self.addCleanup(b.close)

    def test_short_audio(self):
        audio = np.zeros(160, np.float32)
        self.assertEqual(self.backend.recognize(audio), "测试结果")
        self.backend.encoder.encode.assert_called_once()
        np.testing.assert_array_equal(self.backend.encoder.encode.call_args.args[0], audio)
        self.backend._decode.assert_called_once()

    def test_empty_audio_does_not_encode(self):
        self.assertEqual(self.backend.recognize(np.empty(0, np.float32)), "")
        self.backend.encoder.encode.assert_not_called()

    def test_sixty_seconds_accepted(self):
        self.assertEqual(self.backend.recognize(np.zeros(960000, np.float32)), "测试结果")

    def test_invalid_audio(self):
        cases = [
            ([0.0], TypeError),
            (np.zeros(10, np.float64), TypeError),
            (np.zeros((2, 10), np.float32), ValueError),
            (np.array([np.nan], np.float32), ValueError),
            (np.array([np.inf], np.float32), ValueError),
            (np.zeros(960001, np.float32), ValueError),
        ]
        for audio, error in cases:
            with self.subTest(error=error, shape=getattr(audio, "shape", None)):
                with self.assertRaises(error):
                    self.backend.recognize(audio)
        self.backend.encoder.encode.assert_not_called()

    def test_close_releases_once_in_order(self):
        b = self.backend
        owner = Mock()
        ctx, model, table, encoder = b.ctx, b.model, b.embedding_table, b.encoder
        owner.attach_mock(ctx, "ctx")
        owner.attach_mock(model, "model")
        owner.attach_mock(table, "table")
        b.close()
        b.close()
        self.assertEqual([call[0] for call in owner.mock_calls],
                         ["ctx.close", "model.close", "table.close"])
        self.assertIsNone(encoder.sess_fe)
        self.assertIsNone(encoder.sess_be)
        self.assertIsNone(b.encoder)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            b.recognize(np.zeros(160, np.float32))

    def test_model_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "qwen3_asr_encoder_frontend.onnx"):
                QwenBackend(directory)

    def test_repetition_retries_are_bounded_and_raise(self):
        self.backend._decode.return_value = None
        with self.assertRaisesRegex(RuntimeError, "token loop"):
            self.backend.recognize(np.zeros(160, np.float32))
        self.assertEqual(self.backend._decode.call_count, 4)

    def test_decoder_error_is_not_success(self):
        self.backend._decode.side_effect = RuntimeError("decoder aborted")
        with self.assertRaisesRegex(RuntimeError, "decoder aborted"):
            self.backend.recognize(np.zeros(160, np.float32))
        self.backend._decode.assert_called_once()

    def test_context_overflow(self):
        self.backend._build_prompt_embd.return_value = np.zeros((2048, 4), np.float32)
        with self.assertRaisesRegex(ValueError, "context capacity"):
            self.backend.recognize(np.zeros(160, np.float32))
        self.backend._decode.assert_not_called()

    def _native_mocks(self):
        b = self.backend
        b._llama = Mock()
        b.ctx.decode.return_value = 0
        b.ctx.decode_token.return_value = 0
        b.ID_IM_END = 100
        b.model.token_to_bytes.return_value = b"x"
        return b._llama.LlamaBatch.return_value, b._llama.LlamaSampler.return_value

    def test_native_decode_failure_frees_batch(self):
        batch, sampler = self._native_mocks()
        self.backend.ctx.decode.return_value = -1
        with self.assertRaisesRegex(RuntimeError, "prefill failed"):
            QwenBackend._decode(self.backend, np.zeros((2, 4), np.float32), 0.4)
        batch.close.assert_called_once()
        self.backend._llama.LlamaSampler.assert_not_called()
        self.assertEqual(self.backend.ctx.clear_kv_cache.call_count, 2)

    def test_token_budget_exhaustion_raises_and_frees_sampler(self):
        batch, sampler = self._native_mocks()
        self.backend.MAX_NEW_TOKENS = 2
        sampler.sample.side_effect = [1, 2]
        with self.assertRaisesRegex(RuntimeError, "output tokens"):
            QwenBackend._decode(self.backend, np.zeros((2, 4), np.float32), 0.4)
        batch.close.assert_called_once()
        sampler.free.assert_called_once()


if __name__ == "__main__":
    unittest.main()
