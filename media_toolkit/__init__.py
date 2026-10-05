"""本地音视频工具库：文件输入，纯文本输出。"""
from .transcriber import Transcriber
from .audio import MediaDecodeError

__all__ = ["Transcriber", "MediaDecodeError"]
