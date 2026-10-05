"""本机双后端真实集成验证；输出只写入指定的新实验目录。

uv run python scripts/verify_backend_integration.py 音频路径 --output .local/backend-integration --qwen-cpu
验证默认CLI完整转录、SenseVoice API复用/静音/EOF及可选Qwen CPU回退。
不运行Qwen GPU，不做准确率断言。计时不包含模型下载或依赖安装。
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from media_toolkit import Transcriber
from media_toolkit.audio import ffmpeg_executable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qwen-cpu', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {}
    output = args.output / 'full-sensevoice.txt'
    start = time.perf_counter()
    subprocess.run([sys.executable, '-m', 'media_toolkit', str(args.input),
                    '-o', str(output)], check=True)
    full = output.read_text(encoding='utf-8').strip()
    if not full:
        raise AssertionError('完整文件未产生文本')
    report['cli_sensevoice'] = {'wall_seconds': time.perf_counter() - start,
                               'characters': len(full), 'threads': '默认最多12',
                               'provider': 'cpu'}
    clip = args.output / 'clip.wav'
    silence = args.output / 'silence.wav'
    subprocess.run([ffmpeg_executable(), '-v', 'error', '-nostdin', '-i', str(args.input),
                    '-t', '12', '-ac', '1', '-ar', '16000', str(clip)], check=True)
    subprocess.run([ffmpeg_executable(), '-v', 'error', '-nostdin', '-f', 'lavfi',
                    '-i', 'anullsrc=r=16000:cl=mono', '-t', '2', str(silence)], check=True)
    start = time.perf_counter()
    with Transcriber() as transcriber:
        first = transcriber.transcribe(clip)
        model = transcriber._backend
        empty = transcriber.transcribe(silence)
        second = transcriber.transcribe(clip)
        if not first or first != second or empty != '' or model is not transcriber._backend:
            raise AssertionError('模型复用或文件间隔离验证失败')
    if model._recognizer is not None:
        raise AssertionError('关闭后仍持有SenseVoice模型')
    (args.output / 'clip-sensevoice.txt').write_text(first, encoding='utf-8')
    report['api_sensevoice'] = {'wall_seconds': time.perf_counter() - start,
                               'repeat_output_equal': True, 'silence_empty': True,
                               'model_reused_and_released': True}
    if args.qwen_cpu:
        qwen_output = args.output / 'clip-qwen-cpu.txt'
        start = time.perf_counter()
        subprocess.run([sys.executable, '-m', 'media_toolkit', str(clip), '--backend', 'qwen',
                        '--cpu', '-o', str(qwen_output)], check=True)
        qwen_text = qwen_output.read_text(encoding='utf-8').strip()
        if not qwen_text:
            raise AssertionError('Qwen CPU回退未产生文本')
        report['cli_qwen_cpu'] = {'wall_seconds': time.perf_counter() - start,
                                  'audio_seconds': 12, 'characters': len(qwen_text)}
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
