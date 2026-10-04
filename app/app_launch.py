# -*- coding: utf-8 -*-
"""
Seed-VC 网页界面启动器。

直接运行官方 app_svc.py 在本机会立刻失败，原因有两个，这里一并修掉：

1) torchaudio fbank 在 CUDA 上崩溃
   同 svc_launch.py 的问题：compliance.kaldi.fbank 接收 2D CUDA 张量时抛
   UnicodeDecodeError（伪装错误，实际位置在 torch.fft.rfft 内部）。
   而 app_svc.py 第 287 行正好用它提取说话人特征 —— 也就是说网页界面上
   点"提交"就会崩。这里把 fbank 强制放到 CPU 计算。

2) pydub 找不到 ffmpeg
   官方界面用 pydub 把结果编码成 mp3 用于流式播放（app_svc.py 第 351~384 行），
   而本机原本没有 ffmpeg。这里把已部署的 ffmpeg/ffprobe 指给 pydub。

用法（一般通过 run_webapp.ps1 调用）:
    python app_launch.py
"""

import os
import runpy
import sys

import torch
import torchaudio.compliance.kaldi as _kaldi

# ---------- 修复 1: fbank 走 CPU ----------
_orig_fbank = _kaldi.fbank


def fbank_cpu_safe(waveform, *args, **kwargs):
    if torch.is_tensor(waveform) and waveform.is_cuda:
        return _orig_fbank(waveform.cpu(), *args, **kwargs).to(waveform.device)
    return _orig_fbank(waveform, *args, **kwargs)


_kaldi.fbank = fbank_cpu_safe

# ---------- 修复 2: 把 ffmpeg 指给 pydub ----------
_FFMPEG_BIN = os.path.join(WORKSPACE, "tools", "ffmpeg\bin")
_ffmpeg_exe = os.path.join(_FFMPEG_BIN, "ffmpeg.exe")
_ffprobe_exe = os.path.join(_FFMPEG_BIN, "ffprobe.exe")

if os.path.isfile(_ffmpeg_exe):
    os.environ["PATH"] = _FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")
    # pydub 读取这两个模块级属性（也兼容环境变量写法）
    os.environ.setdefault("FFMPEG_BINARY", _ffmpeg_exe)
    os.environ.setdefault("FFPROBE_BINARY", _ffprobe_exe)
    try:
        from pydub import AudioSegment

        AudioSegment.converter = _ffmpeg_exe
        AudioSegment.ffmpeg = _ffmpeg_exe
        AudioSegment.ffprobe = _ffprobe_exe
        print("[启动] pydub -> ffmpeg: %s" % _ffmpeg_exe)
    except Exception as exc:  # pragma: no cover
        print("[启动] pydub 配置失败: %s" % exc)
else:
    print("[启动] 警告: 未找到 %s，mp3 输出会失败" % _ffmpeg_exe)

# ---------- 启动官方界面 ----------
_here = os.path.dirname(os.path.abspath(__file__))
_target = os.path.join(_here, "app_svc.py")
sys.argv = [_target] + sys.argv[1:]
runpy.run_path(_target, run_name="__main__")