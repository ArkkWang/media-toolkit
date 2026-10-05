# 第一版验证记录

日期：2026-10-05。

## 环境

- Windows x64；uv 独立环境，Python 3.12.14。
- NVIDIA GeForce RTX 5080 Laptop GPU，16 GB；驱动 610.78。
- 编码器：ONNX Runtime CPU；解码器：随项目附带的 llama.cpp b10621 Vulkan。
- Python 依赖由 `uv.lock` 固定；模型复制到新项目自身目录。
- 不导入原项目、不连接原服务、不借用原项目虚拟环境。

## 自动验证

- `uv run python -m unittest discover -s tests -v`：26 项通过。
- `uv sync --locked`：通过。
- `uv build --wheel` 和 `uv run python scripts/check_package.py`：通过，DLL 与许可文件已包含，未夹带模型或缓存。
- 静态检查未发现对 `core`、`config_client`、`config_server` 的导入。
- 原 CapsWriter-Offline 项目工作区未修改。

## 真实视频验证

使用用户此前已成功转录的本机视频，时长约 **469.9 秒（7 分 50 秒）**。

1. Python API 完整处理 9 个分片，输出约 2,858 字；中途 GPU 有其他任务竞争，总耗时 241.64 秒，不作为性能基准。
2. 用户释放 GPU 后，使用实际 CLI 重新完整转录，含模型加载和文本保存，墙钟耗时 **20.906 秒**（单次本机测量，非通用保证）。
3. 输出包含全文开头与结尾、中文标点，没有推理诊断标记；人工阅读未见明显大段重复。没有逐字人工对照音频，因此不声称准确率或零遗漏。专有名词仍有识别错误，第一版不含热词与文本纠错。
4. 验证输出保存在 `.local/cli-verified.txt`；API 验证日志、报告和输出保存在 `.local/verified-full.*`。这些文件不纳入 Git。

## 已知限制

- 仅验证 Windows x64 和当前这套 Qwen3-ASR 模型。
- 当前只返回带标点的纯文本，不检测真实停顿，不生成时间戳或字幕。
- 不具备 VAD，背景音乐和静音可能引发模型幻觉；纯数字全零静音会跳过。
- 重叠文本合并是启发式；快速语音若触发 512-token 输出上限会明确报错，不悄悄截断。
- 原生 DLL 的完整再分发授权审计尚未完成，详见第三方说明；本次是本机使用交付，不是公开发行。
