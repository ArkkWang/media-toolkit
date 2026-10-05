# 验证记录

## 双后端版本：SenseVoice 默认 CPU，Qwen 显式回退

日期：2026-10-05。

### 工程选择与边界

- 保持同一个 `Transcriber.transcribe()` 能力，新增 `backend="sensevoice" | "qwen"`；用户要求保留 Qwen 以便手动回退，不建插件注册机制、不自动降级。
- SenseVoice 已在用户提供的真实音频上通过速度测试，用户阅读认为可用，因此作为默认；没有与 Qwen 做同条件准确率比较，不声称其全面优于 Qwen。
- 两者复用文件解码和生命周期管理；SenseVoice 接收无重叠音频、逐文件创建 VAD、按段换行拼接，Qwen 保留原有重叠分片和文本合并。
- 旧 API 仅传 Qwen 目录的调用需补 `backend="qwen"`。模型目录不用于猜测后端；线程覆盖仅适用于 SenseVoice。

### 环境与自动验证

- Windows x64，Python 3.12.14；Intel Core Ultra 9 275HX，24 逻辑 CPU，约 32 GB 内存。
- `sherpa-onnx==1.13.8`、`sherpa-onnx-core==1.13.8`，SenseVoice 和 Silero VAD 都显式使用 CPU；ASR 默认最多 12 线程，VAD 1 线程。
- `uv sync --locked`、`uv pip check`：通过。
- `uv run python -m unittest discover -s tests -v`：**45 项通过**。包括默认选择、Qwen 路由、无自动回退、参数校验、VAD 尾段与文件间隔离、失败关闭音频、保留真实重复、并发拒绝及 CLI 不保存失败结果。这些使用模拟模型，不算识别成功。
- 真实 FFmpeg 测试包含可部分解码的损坏 FLAC：启用 `-xerror` 后报错，避免忽略损坏帧后返回不完整结果。
- `uv build --wheel`、`uv run python scripts/check_package.py`：通过；本机包含新增后端、现有 DLL 和许可说明，无模型或缓存。未做新的原生二进制发行许可审计。

### 正式真实模型验证

源文件：用户提供的 `Zuqb-YFnhG4.m4a`，解码时长 **1022.28175 秒（17 分 02 秒）**。

1. 默认正式 CLI 完整处理文件，墙钟 **22.403 秒**，含 Python 进程启动、模型加载、FFmpeg 解码、VAD、识别和文本保存；输出 6,159 个字符（包含标点与换行，不是汉字计数）。这是单次本机测量，下载与环境安装不计入。
2. SenseVoice API 同一实例依次处理 12 秒片段、2 秒静音、同一片段：两次片段输出一致，静音返回空文本；模型复用且关闭后不再持有识别器。
3. 显式 `--backend qwen --cpu` 识别同一 12 秒片段成功，耗时 **4.804 秒**（含加载）；只证明回退可用，不与完整音频测速直接比较。
4. 两个全新进程分别验证 SenseVoice→Qwen、Qwen→SenseVoice 的 API 加载顺序，都输出非空识别结果；均使用 CPU，没有借用 GPU。
5. 正式版本未重跑 Qwen GPU 性能；其原有验证保留在下方。没有逐字听校、人工参考稿或 CER/WER，因此不声明准确率、无遗漏或无幻觉。

复现命令（输出目录必须不存在）：

```powershell
uv run python scripts/verify_backend_integration.py "D:\Downloads\youtube\Zuqb-YFnhG4.m4a" --output .local/backend-integration-new --qwen-cpu
uv run python scripts/verify_backend_switch.py .local/backend-integration-new/clip.wav
```

本次产物位于 `.local/backend-integration-fixed/`（报告、完整文本和片段文本），加载顺序日志为 `.local/backend-switch.log`，不纳入 Git。

### 实验与集成问题

此前独立 SenseVoice INT8 实验：8 线程 26.6 秒、12 线程 24.0 秒；主进程峰值工作集约 467 MiB。正式 CLI 本次没有重新测内存，不将旧实验内存值冒充正式 CLI 测量。

初次正式环境验证发现：仅锁定 sherpa-onnx 时，跨平台依赖解析漏装 core，导致 ONNX Runtime API 不匹配和原生崩溃。已显式锁定同版 sherpa-onnx-core；修复后重跑完整 CLI、API 及双向加载测试通过。通过单元测试不足以证明原生依赖可运行，故保留真实验证。

---

# 第一版 Qwen 验证记录（历史）

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
