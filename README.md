# media-toolkit

将视频或音频离线转成带标点的纯文本，提供 Python API 和命令行入口。适用于提取视频文稿、整理录音，不生成字幕或时间戳。

当前支持 **Windows x64**。默认使用 **SenseVoiceSmall INT8（CPU）**；保留 **Qwen3-ASR-1.7B** 作为可手动切换的备选，支持 Vulkan 加速解码或纯 CPU。不会自动切换后端。

## 安装与准备

需要 [uv](https://docs.astral.sh/uv/getting-started/installation/) 管理 Python 环境。在终端执行：

```powershell
git clone https://github.com/ArkkWang/media-toolkit.git
cd media-toolkit
uv sync --locked
```

依赖包含 `sherpa-onnx==1.13.8`、配套的 `sherpa-onnx-core==1.13.8` 和 FFmpeg（音视频解码工具），无需单独配置 FFmpeg。仓库不包含模型；按所选后端准备文件即可。

### 默认：SenseVoiceSmall INT8

从 [sherpa-onnx 模型发布包](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2) 解压取得 `model.int8.onnx`、`tokens.txt`，另下载 [Silero VAD 模型](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx)。VAD 用于检测语音片段。将三个文件直接放在：

```text
models/SenseVoiceSmall-int8/
├── model.int8.onnx
├── tokens.txt
└── silero_vad.onnx
```

固定使用 CPU，默认线程数为 `min(12, os.cpu_count() or 1)`。无需准备 Qwen 模型或其 DLL。

### 可选：Qwen3-ASR-1.7B

1. 从[模型发布页](https://github.com/HaujetZhao/CapsWriter-Offline/releases/tag/models)下载 `Qwen3-ASR-1.7B-q5_k.zip`（ONNX + GGUF 版本），解压到 `models/Qwen3-ASR-1.7B/`，确保该目录直接包含：

   ```text
   qwen3_asr_encoder_frontend.onnx
   qwen3_asr_encoder_backend.onnx
   qwen3_asr_llm.gguf
   ```

2. 下载 [llama.cpp b10621 Windows Vulkan x64](https://github.com/ggml-org/llama.cpp/releases/download/b10621/llama-b10621-bin-win-vulkan-x64.zip)，将 `llama.dll`、`ggml.dll`、`ggml-base.dll`、`ggml-vulkan.dll`、`libomp.dll` 和全部 `ggml-cpu-*.dll` 放入 `media_toolkit/backends/_qwen/bin/`，目录不存在时创建。必须使用此版本，其他版本可能与 Python 绑定不兼容。

默认通过 Vulkan 加速解码，无需安装 CUDA；编码器始终使用 CPU。两套模型可以同时保留，切换后端不删除原有文件。

## 转录文件

在项目目录中执行：

```powershell
uv run media-toolkit "D:\videos\demo.mp4" -o "D:\videos\demo.txt"
```

成功后生成 UTF-8 文本文件。已有输出文件不会被覆盖；失败时显示原因并返回非零退出码。

- 省略 `-o`：直接在终端输出文本，不保存文件。
- `--backend sensevoice|qwen`：选择后端，默认 `sensevoice`。
- `--model "模型目录"`：覆盖所选后端的默认目录，不自动识别或切换模型种类。
- `--threads 8`：指定正整数线程数，仅适用于 SenseVoice。
- `--cpu`：仅对 Qwen 生效，禁用解码器 GPU 加速；配合 SenseVoice 无额外效果。

例如，手动切换到 Qwen 纯 CPU 模式：

```powershell
uv run media-toolkit "D:\videos\demo.mp4" --backend qwen --cpu
```

不需要编辑配置文件。模型准备好后，转录过程不联网。

## Python 调用

```python
from media_toolkit import Transcriber

with Transcriber() as transcriber:
    text = transcriber.transcribe("视频.mp4")
    print(text)
```

接口：`Transcriber(model_dir=None, *, backend="sensevoice", use_gpu=None, num_threads=None)`。

| 参数 | 行为 |
| --- | --- |
| `backend` | `"sensevoice"`（默认）或 `"qwen"`；不自动回退 |
| `model_dir` | 省略时分别使用 `models/SenseVoiceSmall-int8` 或 `models/Qwen3-ASR-1.7B`；路径相对当前工作目录，也可用绝对路径 |
| `use_gpu` | SenseVoice 固定 CPU，传 `True` 报错；Qwen 的 `None` 等同 `True`，传 `False` 使用纯 CPU |
| `num_threads` | 正整数，仅 SenseVoice 可用；省略时使用上述默认线程数 |

**旧 API 迁移**：原来仅传 Qwen 目录的调用必须补充 `backend="qwen"`，不会根据目录自动识别模型：

```python
with Transcriber("models/Qwen3-ASR-1.7B", backend="qwen", use_gpu=False) as transcriber:
    text = transcriber.transcribe("视频.mp4")
```

返回值是字符串，库不会自动保存文件，也不读取全局配置或修改工作目录。后端和模型按需加载，退出 `with` 时释放模型。同一实例可以串行处理多个文件，复用模型，不支持并发调用。

## 识别限制与性能参考

- 只处理第一条音轨，逐块读取音频，不将整段音频载入内存。
- SenseVoice：连续音频经 Silero VAD 切段后逐段识别，以换行拼接；不做重叠去重，保留真实重复。VAD 的 `max_speech_duration=25` 是切段目标，不是严格的 25 秒上限，实测出现过约 30 秒的片段。
- Qwen：保持 60 秒分片、4 秒重叠及文本重叠合并；拼接边界仍可能重复或遗漏。
- 标点由模型生成，不代表真实停顿。专有名词可能识别错误，静音或背景音乐也可能产生错误文本；当前不做热词纠正或说话人区分。
- 缺少文件、无音轨、解码或推理失败会报错，不返回截断文本冒充完整结果，也不会自动改用另一后端。

本机正式 CLI 使用默认 12 线程，完整处理 17 分 02 秒音频耗时 **22.4 秒**（含进程启动、模型加载、解码和保存）。此前独立实验的主进程峰值内存约 467 MiB。用户阅读实验文本认为结果可用，但未测量准确率；这不是与 Qwen 的同条件比较，也不是其他机器上的性能保证。

## 开发

```powershell
uv run python -m unittest discover -s tests -v
```

无模型测试与真实模型验证分开记录，不将模拟测试通过视为识别成功。真实模型结果、复现命令和验证范围见[验证记录](VERIFICATION.md)。

## 致谢

感谢 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)、[SenseVoice](https://github.com/FunAudioLLM/SenseVoice)、[Silero VAD](https://github.com/snakers4/silero-vad) 和 [CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline) 提供的开源实现。部分 Qwen 推理和文本合并代码在 CapsWriter-Offline 基础上改造，来源与许可证边界见[第三方说明](THIRD_PARTY_NOTICES.md)。
