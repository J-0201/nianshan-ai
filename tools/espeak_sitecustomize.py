# -*- coding: utf-8 -*-
"""本地新增：让 phonemizer 自动找到 espeakng-loader 自带的 espeak-ng。

背景：YingMusic-Singer 的文本前端（tokenizer/g2p）通过 phonemizer 调用
espeak-ng 库做 IPA 转换。Windows 上没有系统级 espeak-ng 时会报
"espeak not installed on your system"。这里用 PyPI 的 espeakng-loader 包
（自带 espeak-ng.dll 与 espeak-ng-data）替代系统安装。

phonemizer 读取的环境变量（见 phonemizer/backend/espeak/wrapper.py）：
  - PHONEMIZER_ESPEAK_LIBRARY   -> 共享库文件路径
  - PHONEMIZER_ESPEAK_DATA_PATH -> espeak-ng-data 目录

sitecustomize 由 Python 启动时自动导入，因此本文件对 venv 内所有脚本生效。
"""
import os

try:
    import espeakng_loader

    _lib = str(espeakng_loader.get_library_path())
    _data = str(espeakng_loader.get_data_path())

    if os.path.isfile(_lib):
        os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", _lib)
    if os.path.isdir(_data):
        os.environ.setdefault("PHONEMIZER_ESPEAK_DATA_PATH", _data)
        # espeak 自身也会读 ESPEAK_DATA_PATH，一并设置更稳妥
        os.environ.setdefault("ESPEAK_DATA_PATH", _data)

    # DLL 所在目录加入 PATH，便于其依赖被解析
    _dll_dir = os.path.dirname(_lib)
    if os.path.isdir(_dll_dir) and _dll_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _dll_dir + os.pathsep + os.environ.get("PATH", "")
except Exception:
    # 缺失时不阻断解释器启动；没有 espeak 时由 phonemizer 自行报错
    pass
