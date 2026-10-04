# -*- coding: utf-8 -*-
"""
Seed-VC 推理启动器 —— 修复 torchaudio fbank 在 CUDA 上的崩溃。

问题
----
torchaudio 2.4.0 的 torchaudio.compliance.kaldi.fbank() 在接收 **2D CUDA 张量** 时抛出：

    UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc0 in position 117: invalid start byte

报错位置在 torch.fft.rfft() 内部，看起来像是 fft 的问题，实际不是。

实测界定（本机 RTX 3050 / torch 2.4.0+cu124）
--------------------------------------------
    同一张量放 CPU 上              -> 正常，输出 (1078, 80)
    torch.fft.rfft  (CUDA)        -> 正常
    torch.fft.fft   (CUDA)        -> 正常
    torch.stft      (CUDA)        -> 正常
    kaldi.fbank     (CUDA)        -> 崩溃

所以不是 CUDA FFT 的问题，而是 fbank 这条路径上的问题。该 UnicodeDecodeError
很可能是把某个底层错误信息按 UTF-8 解码失败后抛出的"伪装错误"，掩盖了真实原因。

处理
----
fbank 只用于给 campplus 说话人嵌入模型提取 80 维梅尔特征，本身计算量极小
（十几秒音频在 CPU 上只需毫秒级），因此这里强制它在 CPU 上计算，结果再搬回原设备。
这样既绕开崩溃，又不影响数值结果，也不必修改上游仓库代码（便于日后 git pull）。

用法
----
    python svc_launch.py <与 inference.py 完全相同的参数>
"""

import os
import runpy
import sys

import torch
import torchaudio.compliance.kaldi as _kaldi

_orig_fbank = _kaldi.fbank


def fbank_cpu_safe(waveform, *args, **kwargs):
    """CUDA 张量先搬到 CPU 算 fbank，再把结果搬回去。"""
    if torch.is_tensor(waveform) and waveform.is_cuda:
        result = _orig_fbank(waveform.cpu(), *args, **kwargs)
        return result.to(waveform.device)
    return _orig_fbank(waveform, *args, **kwargs)


_kaldi.fbank = fbank_cpu_safe

_here = os.path.dirname(os.path.abspath(__file__))
_target = os.path.join(_here, "inference.py")

# 把命令行参数原样透传给 inference.py
sys.argv = [_target] + sys.argv[1:]

try:
    runpy.run_path(_target, run_name="__main__")
finally:
    # 报告 PyTorch 自身的真实显存占用，用于与 nvidia-smi 的全卡读数区分开：
    # nvidia-smi 报的是全卡占用（含桌面/浏览器等），这里的数字才是模型本身的成本。
    if torch.cuda.is_available():
        alloc = torch.cuda.max_memory_allocated() / 1024 ** 2
        reserved = torch.cuda.max_memory_reserved() / 1024 ** 2
        print("\n[显存] PyTorch 峰值分配: %.0f MiB" % alloc)
        print("[显存] PyTorch 峰值保留: %.0f MiB (含缓存块)" % reserved)