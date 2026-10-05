"""检查源码独立性和 wheel 所需数据，避免仅可在开发目录运行。"""
import ast
from pathlib import Path
from zipfile import ZipFile

root = Path(__file__).resolve().parents[1]
for source in (root / "media_toolkit").rglob("*.py"):
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            modules = [node.module or ""]
        else:
            continue
        for module in modules:
            assert module.split(".")[0] not in {"core", "config_client", "config_server"}, (source, module)

wheel = max((root / "dist").glob("*.whl"), key=lambda path: path.stat().st_mtime)
with ZipFile(wheel) as archive:
    names = archive.namelist()
    for filename in ("llama.dll", "ggml.dll", "ggml-base.dll", "ggml-vulkan.dll", "libomp.dll"):
        assert any(name.endswith("/" + filename) for name in names), filename
    assert any(name.endswith("/licenses/CapsWriter-Offline-MIT.txt") for name in names)
    assert any(name.endswith("/THIRD_PARTY_NOTICES.md") for name in names)
    assert not any("models/" in name or "__pycache__/" in name for name in names)
print("源码无原项目导入；wheel 包含所需 DLL 和许可说明，未夹带模型或缓存。")
