"""SenseVoice INT8 纯 CPU 文件转录实验，不修改正式后端。

uv run --with sherpa-onnx==1.13.8 --with psutil python scripts/benchmark_sensevoice.py 音频路径
模型来源和散列另存于实验报告；输出默认在 .local/ 下。
"""
from __future__ import annotations

import argparse
import json
import platform
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import sherpa_onnx

from media_toolkit.audio import SAMPLE_RATE, OVERLAP_SECONDS, audio_chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio', type=Path)
    parser.add_argument('--model', type=Path, default=Path('models/SenseVoiceSmall-int8'))
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, default=Path('.local/sensevoice-benchmark'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.txt', '.json', '.segments.jsonl'):
        if args.output.with_suffix(suffix).exists():
            raise FileExistsError(args.output.with_suffix(suffix))

    process = psutil.Process()
    stop = threading.Event()
    memory = {'sampled_peak_rss_bytes': 0, 'sampled_peak_tree_rss_bytes': 0}

    def monitor():
        while not stop.is_set():
            rss = process.memory_info().rss
            tree_rss = rss
            for child in process.children(recursive=True):
                try:
                    tree_rss += child.memory_info().rss
                except psutil.Error:
                    pass
            memory['sampled_peak_rss_bytes'] = max(memory['sampled_peak_rss_bytes'], rss)
            memory['sampled_peak_tree_rss_bytes'] = max(memory['sampled_peak_tree_rss_bytes'], tree_rss)
            stop.wait(0.05)

    timer = time.perf_counter()
    cpu_start = process.cpu_times()
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    segments = []
    total_samples = 0
    recognition_seconds = 0.0
    try:
        recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(args.model / 'model.int8.onnx'),
            tokens=str(args.model / 'tokens.txt'),
            num_threads=args.threads, provider='cpu', language='auto', use_itn=True,
        )
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(args.model / 'silero_vad.onnx')
        config.silero_vad.min_silence_duration = 0.5
        config.silero_vad.min_speech_duration = 0.25
        config.silero_vad.max_speech_duration = 25.0
        config.sample_rate = SAMPLE_RATE
        config.num_threads = 1
        config.provider = 'cpu'
        vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=60)
        window = config.silero_vad.window_size
        load_seconds = time.perf_counter() - timer
        print(f'模型加载 {load_seconds:.3f}s，CPU线程 {args.threads}', flush=True)
        pending = np.empty(0, dtype=np.float32)

        with args.output.with_suffix('.segments.jsonl').open('w', encoding='utf-8') as log:
            def drain():
                nonlocal recognition_seconds
                while not vad.empty():
                    speech = vad.front
                    samples = np.asarray(speech.samples, dtype=np.float32).copy()
                    start = speech.start / SAMPLE_RATE
                    vad.pop()
                    t = time.perf_counter()
                    stream = recognizer.create_stream()
                    stream.accept_waveform(SAMPLE_RATE, samples)
                    recognizer.decode_stream(stream)
                    elapsed = time.perf_counter() - t
                    recognition_seconds += elapsed
                    result = stream.result
                    entry = {'start': start, 'duration': len(samples) / SAMPLE_RATE,
                             'seconds': elapsed, 'text': result.text,
                             'language': getattr(result, 'lang', '')}
                    segments.append(entry)
                    log.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    log.flush()
                    print(f'{len(segments):03d} {start:7.1f}s +{entry["duration"]:5.1f}s / {elapsed:.3f}s {result.text[:70]}', flush=True)

            with audio_chunks(args.audio) as chunks:
                for index, audio in enumerate(chunks):
                    # 正式库分片有重叠；还原连续音频，避免重复送给 VAD。
                    if index:
                        audio = audio[OVERLAP_SECONDS * SAMPLE_RATE:]
                    total_samples += len(audio)
                    pending = np.concatenate((pending, audio))
                    consumed = 0
                    while consumed + window <= len(pending):
                        vad.accept_waveform(pending[consumed:consumed + window])
                        consumed += window
                        drain()
                    pending = pending[consumed:].copy()
            if len(pending):
                vad.accept_waveform(np.pad(pending, (0, window - len(pending))))
            vad.flush()
            drain()

        wall_seconds = time.perf_counter() - timer
        cpu_end = process.cpu_times()
        cpu_seconds = cpu_end.user + cpu_end.system - cpu_start.user - cpu_start.system
        native_peak = getattr(process.memory_info(), 'peak_wset', None)
        report = {
            'source': str(args.audio.resolve()), 'audio_seconds': total_samples / SAMPLE_RATE,
            'model': 'SenseVoiceSmall INT8 2024-07-17', 'provider': 'cpu',
            'sherpa_onnx': sherpa_onnx.__version__, 'psutil': psutil.__version__,
            'python': platform.python_version(), 'platform': platform.platform(),
            'logical_cpus': psutil.cpu_count(), 'asr_threads': args.threads, 'vad_threads': 1,
            'vad_max_speech_seconds': 25, 'vad_min_silence_seconds': 0.5,
            'wall_seconds_including_load_and_audio_decode': wall_seconds,
            'model_load_seconds': load_seconds, 'asr_seconds': recognition_seconds,
            'rtf': wall_seconds / (total_samples / SAMPLE_RATE),
            'process_cpu_seconds': cpu_seconds,
            'average_used_cpu_cores': cpu_seconds / wall_seconds,
            'native_peak_working_set_bytes': native_peak,
            **memory,
            'segment_count': len(segments),
            'detected_speech_seconds': sum(s['duration'] for s in segments),
            'notes': '计时从第三方模块导入完成后开始，含模型加载、FFmpeg解码、VAD、ASR和逐段日志；不含下载和依赖安装。CPU时间不含FFmpeg子进程；RSS为工作集，不等于独占内存。无人工参考稿，未测准确率。',
        }
        text = '\n'.join(s['text'].strip() for s in segments if s['text'].strip()) + '\n'
        args.output.with_suffix('.txt').write_text(text, encoding='utf-8')
        args.output.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    finally:
        stop.set()
        watcher.join()


if __name__ == '__main__':
    main()
