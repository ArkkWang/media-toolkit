"""检查实验环境与模型 API，不加载识别模型。"""
import inspect
import json
import platform
import subprocess
import psutil
import sherpa_onnx
from media_toolkit.audio import ffmpeg_executable

print('sherpa-onnx', sherpa_onnx.__version__)
print('platform', platform.platform(), platform.processor())
print('cpu', psutil.cpu_count(logical=False), psutil.cpu_count(), 'ram', psutil.virtual_memory().total)
print(inspect.signature(sherpa_onnx.OfflineRecognizer.from_sense_voice))
print(sherpa_onnx.VadModelConfig.__doc__)
print(sherpa_onnx.VoiceActivityDetector.__doc__)
r = subprocess.run([ffmpeg_executable(), '-hide_banner', '-i', r'D:\Downloads\youtube\Zuqb-YFnhG4.m4a'], capture_output=True)
print(r.stderr.decode('utf-8', errors='replace'))
