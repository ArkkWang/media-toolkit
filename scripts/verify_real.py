"""本机真实模型验证：独立启动后台运行，可观察分片耗时。"""
import argparse
import json
import logging
from pathlib import Path
import time

from media_toolkit import Transcriber
from media_toolkit.backends.qwen import QwenBackend

parser = argparse.ArgumentParser()
parser.add_argument("input", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=args.output.with_suffix(".log"), level=logging.INFO,
                    format="%(asctime)s %(message)s", encoding="utf-8")
started = time.perf_counter()
original = QwenBackend.recognize
count = 0


def timed_recognize(self, audio):
    global count
    count += 1
    at = time.perf_counter()
    logging.info("开始分片 %s, %.2f 秒", count, len(audio) / 16000)
    text = original(self, audio)
    logging.info("完成分片 %s, 耗时 %.2f 秒, 字数 %s", count, time.perf_counter() - at, len(text))
    return text


QwenBackend.recognize = timed_recognize
try:
    with Transcriber("models/Qwen3-ASR-1.7B") as transcriber:
        text = transcriber.transcribe(args.input)
    args.output.write_text(text, encoding="utf-8")
    report = {"elapsed_seconds": round(time.perf_counter() - started, 2),
              "chunks": count, "characters": len(text), "success": True}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logging.info("完成：%s", report)
except BaseException:
    logging.exception("验证失败")
    raise
