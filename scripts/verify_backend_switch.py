"""在全新进程中检查两个后端的两种加载顺序，均仅使用 CPU。

uv run python scripts/verify_backend_switch.py 音频短片路径
用独立进程分别验证 SenseVoice→Qwen 与 Qwen→SenseVoice，避免 DLL 加载顺序掩盖兼容问题。
"""
import argparse
import subprocess
import sys

from media_toolkit import Transcriber


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio')
    parser.add_argument('--first', choices=['sensevoice', 'qwen'])
    args = parser.parse_args()
    if args.first is None:
        for first in ('sensevoice', 'qwen'):
            subprocess.run([sys.executable, __file__, args.audio, '--first', first], check=True)
        return
    second = 'qwen' if args.first == 'sensevoice' else 'sensevoice'
    for name in (args.first, second):
        with Transcriber(backend=name, use_gpu=False) as transcriber:
            text = transcriber.transcribe(args.audio)
            if not text:
                raise AssertionError(f'{name} 未返回文字')
        print(f'{args.first}优先加载：{name} 识别成功，{len(text)}字', flush=True)


if __name__ == '__main__':
    main()
