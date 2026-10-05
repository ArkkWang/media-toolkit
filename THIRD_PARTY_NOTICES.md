# 第三方归属与许可说明

## 改造的 Python 推理源码

`media_toolkit/backends/_qwen/encoder.py`、`prompt.py`、`llama.py` 及
`backends/qwen.py` 的解码流程复制改造自 **CapsWriter-Offline**。
版权所有：Copyright (c) 2026 Haujet Zhao，采用 MIT 许可证；完整原文保留于
`licenses/CapsWriter-Offline-MIT.txt`。

来源检出目录：`../CapsWriter-Offline`，提交：
`784f3da58559d5bfce0ccc4ad92567ea8db99cbd`。
原始路径：`core/server/engines/qwen_asr_gguf/inference/{encoder,asr}.py`
与 `core/server/engines/llama/llama.py`；使用共享的 **b10621** 绑定，而非
inference 目录中的旧版绑定。

改动：删除服务器、后台 worker、对齐器、导出器、CLI、文件解码、中文 ITN
和流式输出；保留 NumPy mel 特征提取、拆分 ONNX 编码、GGUF embedding
提取与原生解码。新增显式模型路径、原生错误检查、输入长度限制、显式资源释放
和本地 DLL 加载；不改变工作目录或 PATH。运行时不依赖原项目目录。

## 改造的文本重叠合并代码

`media_toolkit/text.py` 改自 CapsWriter-Offline 的
`core/server/merger/text_merger.py`，同样采用 MIT 许可证，
版权所有：Copyright (c) 2026 Haujet Zhao。
许可证原文见 `licenses/CapsWriter-Offline-MIT.txt`。

## Qwen 本机原生运行库：Windows x64

GitHub 仓库不上传 DLL，以下说明记录本机副本的来源与授权核验范围。
新机器需按 README 从上游自行准备匹配版本；本机现有 DLL 不会因发布而删除。

`media_toolkit/backends/_qwen/bin/` 是原项目
`core/server/engines/llama/bin/` 的选定子集。原目录下载说明标注的来源：

https://github.com/ggml-org/llama.cpp/releases/download/b10621/llama-b10621-bin-win-vulkan-x64.zip

保留 `llama.dll`、`ggml.dll`、`ggml-base.dll`、各 CPU 后端变体、
`ggml-vulkan.dll` 与 `libomp.dll`；不包含 CLI、服务器、基准测试、导出和 RPC
相关 DLL。文件校验值见 `licenses/native-sha256.txt`。

- llama.cpp / ggml：MIT；原文见 `licenses/llama.cpp-MIT.txt`，取得自上游
  b10621 标签：https://github.com/ggml-org/llama.cpp/blob/b10621/LICENSE
- LLVM OpenMP（`libomp.dll`）：适用独立 LLVM 条款，**不是 CapsWriter 的 MIT**。
  原文见 `licenses/LLVM-OpenMP.txt`，取得自
  https://github.com/llvm/llvm-project/blob/main/openmp/LICENSE.TXT 。
  原项目未提供该二进制对应的准确 LLVM 构建版本。
- Vulkan 后端可能包含着色器、编译器或其他构建依赖，分别适用各上游许可证。
  本地原项目没有提供二进制依赖清单或完整发行许可说明，其根目录 MIT 许可证
  **不能证明所有随附二进制内容均采用 MIT**。对外再分发前，需审核官方 b10621
  发行包及构建依赖，补齐必要许可说明。本文件记录来源，不宣称已完成二进制许可审计。
- GPU 驱动、Vulkan 加载器及 Microsoft 系统运行库属于平台前置条件，
  本项目不提供它们，也不改变其许可。

## 单独安装的依赖与模型权重

NumPy、ONNX Runtime、gguf、imageio-ffmpeg 均为单独安装的依赖，各自适用其发行许可证，
未将它们的 Python 包源码复制进本项目。随依赖提供的 FFmpeg 二进制还需遵守其具体
发行构建的许可，不能仅以 Python 包的许可证代替。Qwen 编码器仅使用 CPU；
`use_gpu=True`（或默认的 `None`）通过 Vulkan 运行库允许 GGUF 卸载至 GPU，
实际能力取决于硬件和驱动。

Qwen3-ASR 权重及 ONNX/GGUF 转换产物适用各自模型与转换条款，
不属于 CapsWriter 源码 MIT 许可证的授权范围。后端不下载或复制权重；
调用方需准备本地模型，可使用默认目录 `models/Qwen3-ASR-1.7B` 或显式指定目录。

### SenseVoice、sherpa-onnx 与 Silero VAD

默认 SenseVoice 后端通过单独安装的 `sherpa-onnx==1.13.8` 与
`sherpa-onnx-core==1.13.8` 在 CPU 上运行。显式锁定 core，避免跨平台 wheel
元数据差异导致原生运行库漏装；它们不依赖 Qwen 的本地 DLL。
本项目不提交该依赖的 wheel、原生二进制或模型权重。来源与上游许可说明入口：

- **sherpa-onnx**：https://github.com/k2-fsa/sherpa-onnx ，上游项目采用
  Apache-2.0；其 wheel 内原生组件和构建依赖仍适用各自条款。
- **SenseVoice**：https://github.com/FunAudioLLM/SenseVoice ，上游项目采用
  Apache-2.0。本项目使用 SenseVoiceSmall 的 INT8 ONNX 转换产物，不将源码许可
  自动视为所有模型、转换产物和分发包的完整授权证明。
- **Silero VAD**：https://github.com/snakers4/silero-vad ，上游项目采用 MIT。
  本地使用的 ONNX 文件来自 sherpa-onnx 发布渠道，具体产物仍需结合该版本的
  模型与分发许可说明核对。

SenseVoice 模型发布来源：

- Hugging Face：
  https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17
- sherpa-onnx INT8 发布包：
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2
- 本机实际取得模型的 ModelScope 仓库：
  https://modelscope.cn/models/chriscrs/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17

本地 `model.int8.onnx` 的 SHA256 为
`c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51`，
与上述 Hugging Face 仓库对应文件一致。这仅证明该文件内容一致，不证明
`tokens.txt`、其他文件或整个分发包均已核验，也不代替许可审计。

本地 `silero_vad.onnx` 下载来源：
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx 。
使用时与 `model.int8.onnx`、`tokens.txt` 一起放入
`models/SenseVoiceSmall-int8`，也可指定其他模型目录。

以上记录来源和许可证边界，**不宣称已完成模型、wheel 或其原生依赖的许可审计**。
这些组件不属于 CapsWriter 源码 MIT 许可证的授权范围；对外再分发前需核对
具体版本的权重、转换产物和二进制条款，并保留必要的版权、许可证和 NOTICE。

## Qwen 运行限制与资源生命周期

`QwenBackend.recognize` 接受一维 float32、16 kHz 音频，每次最多 60 秒；
解码器上下文为 2048 token，最多生成 512 token。更长音频须由调用方分片。
耗尽输出 token 预算会报错，不会返回静默截断结果。重复熔断最多尝试四次，
全部失败后抛出异常。ASR 模型可能对静音产生幻觉；本后端未实现 VAD。

`close()` 释放实例持有的上下文、模型、embedding 内存映射和 ONNX 会话。
每次推理的 batch 与 sampler 在发生异常时也会释放。原生库、后端注册与 DLL
搜索目录句柄保留至进程结束，避免关闭某个识别实例时破坏其他仍存活的实例。
同一实例的调用串行执行。随附绑定与 DLL 是配套的 b10621 ABI，不能单独替换 DLL。
