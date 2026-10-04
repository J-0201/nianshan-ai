# -*- coding: utf-8 -*-
"""YingMusic-Singer 4GB 显存可行性探测。

用仓库自带示例素材（MIDI + 音色音频），跑一次真实推理，
实测：模型加载耗时、推理耗时、显存峰值、输出规格。
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

os.environ.setdefault("TORCH_HOME", os.path.join(ROOT, "ckpts", "torch_hub"))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "ckpts", "hf"))

import numpy as np
import soundfile as sf
import torch

from singer.model import SAMPLE_RATE_48K, YingSinger

# 仓库自带示例（README 里验证过的组合）
TIMBRE = os.path.join(ROOT, "resources", "audios", "male.wav")
TIMBRE_CONTENT = "在爱的回归线，又期待相见。"
MIDI = os.path.join(ROOT, "resources", "audios", "female__Rnb_Funk__下等马_clip_001.mid")
LYRICS = (
    "头抬起来，你表情别太奇怪，无大碍。没伤到脑袋，如果我下手太重，私密马赛。"
    "习武十载，没下山没谈恋爱，吃光后山七八亩菜，练就这套拳脚，莫以貌取人哉。"
    "暮色压台，擂鼓未衰，下一个谁还要来？速来领拜，别耽误我热蒸铁揭盖。"
)
OUT = os.path.join(ROOT, "outputs", "probe_v1.wav")

NFE = int(os.environ.get("PROBE_NFE", "32"))
CFG = float(os.environ.get("PROBE_CFG", "4.0"))
PITCH = int(os.environ.get("PROBE_PITCH", "-1"))

print("=" * 62)
print("YingMusic-Singer 4GB 探测 | nfe_steps=%d cfg=%.1f pitch=%d" % (NFE, CFG, PITCH))
print("=" * 62)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("设备:", device, "|", torch.cuda.get_device_name(0) if device == "cuda" else "")
if device == "cuda":
    torch.cuda.reset_peak_memory_stats()
    print("基线显存: %.0f MiB" % (torch.cuda.memory_allocated() / 1024**2))

# ---- 模型加载 ----
t0 = time.time()
try:
    singer = YingSinger(singer_path=os.path.join(ROOT, "ckpts"), device=device)
except Exception as exc:
    print("\n[失败] 模型加载异常: %s: %s" % (type(exc).__name__, exc))
    import traceback
    traceback.print_exc()
    sys.exit(2)
t_load = time.time() - t0
alloc = torch.cuda.memory_allocated() / 1024**2 if device == "cuda" else 0
print("模型加载完成: %.1f 秒 | 权重占用 %.0f MiB" % (t_load, alloc))

# ---- 推理 ----
t1 = time.time()
try:
    wav = singer.inference(
        timbre_audio_path=TIMBRE,
        timbre_audio_content=TIMBRE_CONTENT,
        midi_file=MIDI,
        lyrics=LYRICS,
        pitch_shift=PITCH,
        cfg_strength=CFG,
        nfe_steps=NFE,
    )
except torch.cuda.OutOfMemoryError as exc:
    print("\n[OOM] 显存不足: %s" % exc)
    print("峰值显存: %.0f MiB" % (torch.cuda.max_memory_allocated() / 1024**2))
    sys.exit(3)
except Exception as exc:
    print("\n[失败] 推理异常: %s: %s" % (type(exc).__name__, exc))
    import traceback
    traceback.print_exc()
    sys.exit(4)
t_inf = time.time() - t1

peak = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0
reserved = torch.cuda.max_memory_reserved() / 1024**2 if device == "cuda" else 0

# ---- 保存（用 soundfile，绕开 torchcodec 依赖）----
os.makedirs(os.path.dirname(OUT), exist_ok=True)
arr = wav.detach().cpu().float().numpy()
if arr.ndim == 1:
    arr = arr[:, None]
elif arr.shape[0] <= 2 and arr.shape[1] > arr.shape[0]:
    arr = arr.T  # (channels, samples) -> (samples, channels)
sf.write(OUT, arr, SAMPLE_RATE_48K)

dur = arr.shape[0] / SAMPLE_RATE_48K
print("\n" + "=" * 62)
print("推理完成: %.1f 秒" % t_inf)
print("输出: %.2f 秒 | %d Hz | %d 声道 | %s" % (dur, SAMPLE_RATE_48K, arr.shape[1], OUT))
print("PyTorch 显存峰值: %.0f MiB (分配) / %.0f MiB (保留)" % (peak, reserved))
if device == "cuda":
    free, total = torch.cuda.mem_get_info()
    print("推理结束时空闲显存: %.0f MiB / %.0f MiB" % (free / 1024**2, total / 1024**2))
print("实时率 RTF: %.2f (%.2f 秒音频 / %.1f 秒耗时)" % (t_inf / max(dur, 1e-6), dur, t_inf))
print("=" * 62)
