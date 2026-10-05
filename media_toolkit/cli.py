"""可选命令行外壳，核心库不读取命令行或全局配置。"""
import argparse
from pathlib import Path
import sys

from . import Transcriber
from .transcriber import DEFAULT_MODELS


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="离线转录视频或音频，输出带标点的纯文本")
    parser.add_argument("file", type=Path, help="视频或音频文件")
    parser.add_argument("--backend", choices=tuple(DEFAULT_MODELS), default="sensevoice",
                        help="识别后端（默认 sensevoice：纯 CPU；qwen：默认 GPU 解码）")
    parser.add_argument("--model", type=Path, help="覆盖所选后端的默认模型目录，相对当前目录")
    parser.add_argument("--threads", type=int, help="SenseVoice 推理线程数（默认最多 12，不超过逻辑 CPU 数）")
    parser.add_argument("--cpu", action="store_true", help="禁用 Qwen 解码器 GPU 加速；SenseVoice 始终使用 CPU")
    parser.add_argument("-o", "--output", type=Path, help="保存 UTF-8 文本；省略则输出到终端")
    args = parser.parse_args(argv)
    if args.threads is not None and (args.threads < 1 or args.backend != "sensevoice"):
        parser.error("--threads 必须为正整数，且仅适用于 sensevoice 后端")
    if args.output and args.output.exists():
        parser.error(f"输出文件已存在，不会覆盖：{args.output}")
    try:
        with Transcriber(args.model, backend=args.backend,
                         use_gpu=False if args.cpu else None,
                         num_threads=args.threads) as transcriber:
            text = transcriber.transcribe(args.file)
        if args.output:
            # x 模式也防止转录期间其他进程创建的文件被覆盖。
            with args.output.open("x", encoding="utf-8", newline="\n") as output:
                output.write(text + "\n")
            print(f"已保存：{args.output}", file=sys.stderr)
        else:
            print(text)
        return 0
    except KeyboardInterrupt:
        print("转录已取消", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        print(f"转录失败：{error}", file=sys.stderr)
        return 1
