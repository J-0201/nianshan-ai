# -*- coding: utf-8 -*-
"""实测：扩散步数对音色相似度的影响（Seed-VC / 念山AI 音色转换）。

方法：
  - 固定源音频与参考音频，只改变扩散步数；
  - 用 Seed-VC 自身的 CAM++ 说话人模型（app_svc.py:287-292 的同一套提取方式）
    计算 192 维音色嵌入；
  - 比较「输出 vs 参考」（越高越像目标音色）与「输出 vs 源」（越低转换越彻底）。
"""
import json
import os
import shutil
import sys
import time

import httpx

BASE = "http://127.0.0.1:7860"
SVC_ROOT = SVC_DIR
SRC = os.path.join(DEMO_DIR, "test_source_short.wav")
REF = os.path.join(DEMO_DIR, "test_ref.wav")
SAVE = os.path.join(WORKSPACE, "tools", "steps_test")
STEPS_LIST = [8, 15, 30, 50, 80]

os.makedirs(SAVE, exist_ok=True)
os.chdir(SVC_ROOT)
sys.path.insert(0, SVC_ROOT)
os.environ.setdefault("HF_HUB_CACHE", "./checkpoints/hf_cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.compliance.kaldi as _kaldi

# 与念山AI 相同：fbank 在 CUDA 上会崩，改在 CPU 算
_orig_fbank = _kaldi.fbank


def fbank_safe(wav, *a, **kw):
    if torch.is_tensor(wav) and wav.is_cuda:
        return _orig_fbank(wav.cpu(), *a, **kw).to(wav.device)
    return _orig_fbank(wav, *a, **kw)


_kaldi.fbank = fbank_safe

from hf_utils import load_custom_model_from_hf          # noqa: E402
from modules.campplus.DTDNN import CAMPPlus             # noqa: E402

# ---------- 路径解析（可用环境变量覆盖）----------
# 仓库根 = 本文件所在目录的上一级；上游项目默认与仓库同级。
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE = os.path.dirname(_REPO)
SVC_DIR = os.environ.get("NS_SEEDVC_DIR") or os.path.join(WORKSPACE, "seed-vc")
SVS_DIR = os.environ.get("NS_SVS_DIR") or os.path.join(WORKSPACE, "YingMusic-Singer")
DEMO_DIR = os.environ.get("NS_DEMO_AUDIO_DIR") or os.path.join(WORKSPACE, "demo_audio")
RESULTS = os.path.join(_REPO, "tools", "results")


CKPT = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", config_filename=None)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
_spk = CAMPPlus(feat_dim=80, embedding_size=192)
_spk.load_state_dict(torch.load(CKPT, map_location="cpu"))
_spk.eval().to(DEV)
print("CAM++ 说话人模型已加载:", CKPT)


def embed(path):
    """提取 192 维说话人嵌入（复刻 app_svc.py:287-292）。"""
    wav, sr = sf.read(path, always_2d=True)
    w = torch.from_numpy(wav.mean(axis=1).astype(np.float32))
    if sr != 16000:
        w = torchaudio.functional.resample(w, sr, 16000)
    feat = torchaudio.compliance.kaldi.fbank(w.unsqueeze(0), num_mel_bins=80,
                                             dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    with torch.no_grad():
        e = _spk(feat.unsqueeze(0).to(DEV))
    return e.squeeze(0).cpu().numpy()


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def centroid(path):
    wav, sr = sf.read(path, always_2d=True)
    m = wav.mean(axis=1)
    X = np.abs(np.fft.rfft(m * np.hanning(len(m))))
    f = np.fft.rfftfreq(len(m), 1 / sr)
    return float((X * f).sum() / max(X.sum(), 1e-9))


# ---- 基线 ----
e_src, e_ref = embed(SRC), embed(REF)
print("基线：源↔参考 相似度 %.4f（越高说明本来就像，任务越容易）" % cos(e_src, e_ref))
print("      源质心 %.0f Hz | 参考质心 %.0f Hz\n" % (centroid(SRC), centroid(REF)))

c = httpx.Client(timeout=httpx.Timeout(1200.0, connect=30.0))
ep = (c.get(BASE + "/gradio_api/info").json().get("named_endpoints") or {}).get("/convert")
names = [p.get("parameter_name") for p in ep["parameters"]]

uploaded = []
for p in (SRC, REF):
    with open(p, "rb") as fh:
        r = c.post(BASE + "/gradio_api/upload",
                   files={"files": (os.path.basename(p), fh, "audio/wav")})
    uploaded.append(r.json()[0])
fd = lambda p: {"path": p, "meta": {"_type": "gradio.FileData"}}

rows = []
print("%6s | %8s | %10s | %10s | %9s | %s" %
      ("步数", "耗时(秒)", "输出↔参考", "输出↔源", "质心Hz", "文件"))
print("-" * 78)

for steps in STEPS_LIST:
    values = {"source": fd(uploaded[0]), "target": fd(uploaded[1]), "is_song": False,
              "steps": steps, "pitch": 0, "length_adjust": 1.0, "cfg": 0.7, "auto_f0": False}
    data = [values[n] for n in names]
    t0 = time.time()
    r = c.post(BASE + "/gradio_api/call/convert", json={"data": data})
    eid = r.json()["event_id"]
    out_path = None
    with c.stream("GET", BASE + f"/gradio_api/call/convert/{eid}") as s:
        for line in s.iter_lines():
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload and payload != "null":
                    try:
                        obj = json.loads(payload)
                        if isinstance(obj, list):
                            for item in obj:
                                if isinstance(item, dict) and item.get("path", "").endswith(".wav"):
                                    out_path = item["path"]
                    except Exception:
                        pass
    dt = time.time() - t0
    if not out_path or not os.path.isfile(out_path):
        print("%6d | 失败（未取到产物）" % steps)
        continue
    saved = os.path.join(SAVE, "steps_%03d.wav" % steps)
    shutil.copy(out_path, saved)
    e_out = embed(saved)
    sim_ref = cos(e_out, e_ref)
    sim_src = cos(e_out, e_src)
    ce = centroid(saved)
    rows.append((steps, dt, sim_ref, sim_src, ce, saved))
    print("%6d | %8.1f | %10.4f | %10.4f | %9.0f | %s" %
          (steps, dt, sim_ref, sim_src, ce, os.path.basename(saved)))

print()
if rows:
    best = max(rows, key=lambda x: x[2])
    print("音色最相似（输出↔参考）的步数: %d  (相似度 %.4f)" % (best[0], best[2]))
    print("步数 8 → 80 的相似度变化: %.4f → %.4f (Δ %+.4f)" %
          (rows[0][2], rows[-1][2], rows[-1][2] - rows[0][2]))
