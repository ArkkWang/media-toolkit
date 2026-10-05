# media-toolkit

将视频或音频离线转成带标点的纯文本，提供 Python API 和命令行入口。适用于提取视频文稿、整理录音，不生成字幕或时间戳。

当前支持 **Windows x64**，使用 Qwen3-ASR-1.7B 本地语音识别模型。默认通过 Vulkan 加速解码（无需安装 CUDA），也可使用 CPU。

## 安装与准备

需要 [uv](https://docs.astral.sh/uv/getting-started/installation/) 管理 Python 环境。在终端执行：

```powershell
git clone https://github.com/ArkkWang/media-toolkit.git
cd media-toolkit
uv sync --locked
```

仓库不包含模型和原生运行库，首次使用还需准备以下文件：

1. **模型**：从[模型发布页](https://github.com/HaujetZhao/CapsWriter-Offline/releases/tag/models)下载 `Qwen3-ASR-1.7B-q5_k.zip`（ONNX + GGUF 版本），解压到 `models/Qwen3-ASR-1.7B/`，确保该目录直接包含：

   ```text
   qwen3_asr_encoder_frontend.onnx
   qwen3_asr_encoder_backend.onnx
   qwen3_asr_llm.gguf
   ```

2. **运行库**：下载 [llama.cpp b10621 Windows Vulkan x64](https://github.com/ggml-org/llama.cpp/releases/download/b10621/llama-b10621-bin-win-vulkan-x64.zip)，将 `llama.dll`、`ggml.dll`、`ggml-base.dll`、`ggml-vulkan.dll`、`libomp.dll` 和全部 `ggml-cpu-*.dll` 放入 `media_toolkit/backends/_qwen/bin/`，目录不存在时创建。必须使用此版本，其他版本可能与 Python 绑定不兼容。

FFmpeg（音视频解码工具）随依赖安装，无需单独配置。已有上述文件时可直接开始转录。

## 转录文件

在项目目录中执行：

```powershell
uv run media-toolkit "D:\videos\demo.mp4" -o "D:\videos\demo.txt"
```

成功后生成 UTF-8 文本文件。已有输出文件不会被覆盖；失败时显示原因并返回非零退出码。

- 省略 `-o`：直接在终端输出文本，不保存文件。
- `--model "模型目录"`：指定其他模型存放位置，不用于切换模型种类。
- `--cpu`：禁用解码器 GPU 加速；编码器始终使用 CPU。

不需要编辑配置文件。模型准备好后，转录过程不联网。

## Python 调用

```python
from media_toolkit import Transcriber

with Transcriber("models/Qwen3-ASR-1.7B") as transcriber:
    text = transcriber.transcribe("视频.mp4")
    print(text)
```

返回值是字符串，库不会自动保存文件。路径相对当前工作目录，也可使用绝对路径；纯 CPU 模式传入 `use_gpu=False`。

模型在首次转录时加载，退出 `with` 时释放。同一实例可以串行处理多个文件，复用模型，不支持并发调用。

## 识别限制

- 只处理第一条音轨。长文件按 60 秒分片、4 秒重叠处理，不会将整段音频载入内存；拼接边界仍可能重复或遗漏。
- 标点由模型生成，不代表真实停顿。专有名词可能识别错误，静音或背景音乐也可能产生错误文本；当前不做热词纠正或说话人区分。
- 缺少文件、无音轨、解码或推理失败会报错，不返回截断文本冒充完整结果。

## 开发

```powershell
uv run python -m unittest discover -s tests -v
```

测试不需要模型或 GPU，覆盖音频解码、分片合并、资源释放和错误处理。真实模型验证结果见 [验证记录](VERIFICATION.md)。

## 致谢

感谢 [CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline) 提供的开源实现。部分推理和文本合并代码在其基础上改造，来源与许可证见 [第三方说明](THIRD_PARTY_NOTICES.md)。
