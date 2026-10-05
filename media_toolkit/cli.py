"""可选命令行外壳，核心库不读取命令行或全局配置。"""
import argparse
from pathlib import Path
import sys

from . import Transcriber


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="离线转录视频或音频，输出带标点的纯文本")
    parser.add_argument("file", type=Path, help="视频或音频文件")
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3-ASR-1.7B"),
                        help="模型目录（默认：models/Qwen3-ASR-1.7B，相对当前目录）")
    parser.add_argument("--cpu", action="store_true", help="禁用解码器 GPU 加速")
    parser.add_argument("-o", "--output", type=Path, help="保存 UTF-8 文本；省略则输出到终端")
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        parser.error(f"输出文件已存在，不会覆盖：{args.output}")
    try:
        with Transcriber(args.model, use_gpu=not args.cpu) as transcriber:
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
