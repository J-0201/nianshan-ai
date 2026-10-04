# -*- coding: utf-8 -*-
"""验证：参考音频分离人声是否真的让音色更像。

设计：
  1. 取一段**带伴奏**的歌曲片段作为参考音频；
  2. 先用「参考分离」跑一次（应用内部会分离，产物可直接作为"干净人声真值"）；
  3. 再用「不分离」跑一次，参考仍是同一段带伴奏音频；
  4. 用 Seed-VC 自身的 CAM++ 说话人模型，比较两个输出与真值的音色相似度。
"""
import glob
import json
import os
import shutil
import sys
import time

import httpx

BASE = "http://127.0.0.1:7860"
SVC = SVC_DIR
SRC = os.path.join(DEMO_DIR, "test_source_short.wav")
REF_MIXED = os.path.join(WORKSPACE, "tools", "ref_sep_test\ref_with_music.wav")
OUT_DIR = os.path.join(SVC, "out")
SAVE = os.path.join(WORKSPACE, "tools", "ref_sep_test")

os.chdir(SVC)
sys.path.insert(0, SVC)
os.environ.setdefault("HF_HUB_CACHE", "./checkpoints/hf_cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.compliance.kaldi as _kaldi

_orig = _kaldi.fbank


def fbank_safe(w, *a, **kw):
    if torch.is_tensor(w) and w.is_cuda:
        return _orig(w.cpu(), *a, **kw).to(w.device)
    return _orig(w, *a, **kw)


_kaldi.fbank = fbank_safe

from hf_utils import load_custom_model_from_hf      # noqa: E402
from modules.campplus.DTDNN import CAMPPlus         # noqa: E402

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


def embed(path):
    wav, sr = sf.read(path, always_2d=True)
    w = torch.from_numpy(wav.mean(axis=1).astype(np.float32))
    if sr != 16000:
        w = torchaudio.functional.resample(w, sr, 16000)
    feat = torchaudio.compliance.kaldi.fbank(w.unsqueeze(0), num_mel_bins=80,
                                             dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    with torch.no_grad():
        return _spk(feat.unsqueeze(0).to(DEV)).squeeze(0).cpu().numpy()


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


c = httpx.Client(timeout=httpx.Timeout(1800.0, connect=30.0))
names = [p["parameter_name"] for p in
         c.get(BASE + "/gradio_api/info").json()["named_endpoints"]["/convert"]["parameters"]]


def upload(path):
    with open(path, "rb") as fh:
        r = c.post(BASE + "/gradio_api/upload",
                   files={"files": (os.path.basename(path), fh, "audio/wav")})
    return r.json()[0]


fd = lambda p: {"path": p, "meta": {"_type": "gradio.FileData"}}
up_src, up_ref = upload(SRC), upload(REF_MIXED)


def run(sep_ref, tag):
    values = {"source": fd(up_src), "target": fd(up_ref), "is_song": False,
              "sep_ref": sep_ref, "steps": 30, "pitch": 0,
              "length_adjust": 1.0, "cfg": 0.7, "auto_f0": False}
    t0 = time.time()
    r = c.post(BASE + "/gradio_api/call/convert", json={"data": [values[n] for n in names]})
    eid = r.json()["event_id"]
    out_path, status = None, ""
    with c.stream("GET", BASE + f"/gradio_api/call/convert/{eid}") as s:
        for line in s.iter_lines():
            if line.startswith("data:"):
                pl = line[5:].strip()
                if pl and pl != "null":
                    try:
                        obj = json.loads(pl)
                        if isinstance(obj, list):
                            for it in obj:
                                if isinstance(it, dict) and str(it.get("path", "")).endswith(".wav"):
                                    out_path = it["path"]
                                elif isinstance(it, str) and it.startswith(("⏳", "🔄", "✅", "❌", "⚠️")):
                                    status = it
                    except Exception:
                        pass
    dt = time.time() - t0
    saved = None
    if out_path and os.path.isfile(out_path):
        saved = os.path.join(SAVE, "out_%s.wav" % tag)
        shutil.copy(out_path, saved)
    print("  [%s] 耗时 %.1f 秒 -> %s" % (tag, dt, os.path.basename(saved) if saved else "失败"))
    print("       状态: %s" % status.split("\n")[0][:70])
    return saved


print("=== 测试 1：参考音频【分离人声】(sep_ref=True) ===")
before = set(glob.glob(os.path.join(OUT_DIR, "sep_ref_*")))
out_sep = run(True, "sep_ref")
after = set(glob.glob(os.path.join(OUT_DIR, "sep_ref_*")))
new_dirs = sorted(after - before)
gt_vocals = None
for d in new_dirs:
    v = os.path.join(d, "vocals.wav")
    if os.path.isfile(v):
        gt_vocals = v
gt_vocals = gt_vocals or os.path.join(SAVE, "gt_vocals.wav")
if gt_vocals and os.path.isfile(gt_vocals):
    shutil.copy(gt_vocals, os.path.join(SAVE, "gt_vocals.wav"))
    print("  干净人声真值: %s" % gt_vocals)

if not gt_vocals or not os.path.isfile(gt_vocals):
    sys.exit("未能取到分离出的人声，无法比较")

print("\n=== 测试 2：参考音频【不分离】(sep_ref=False，直接用带伴奏原声) ===")
out_raw = run(False, "no_sep")

print("\n=== 结果对比（CAM++ 音色相似度，越接近真值越好）===")
e_gt = embed(gt_vocals)
e_mixed = embed(REF_MIXED)
e_esrc = embed(SRC)
print("参考原始混音 vs 真值人声      : %.4f  <- 音乐污染程度（越低说明嵌入被严重带偏）" % cos(e_mixed, e_gt))
print("源音频         vs 真值人声      : %.4f  (基线)" % cos(e_esrc, e_gt))
if out_sep:
    e = embed(out_sep)
    print("【分离参考】输出 vs 真值人声    : %.4f" % cos(e, e_gt))
if out_raw:
    e = embed(out_raw)
    print("【未分离参考】输出 vs 真值人声  : %.4f" % cos(e, e_gt))
if out_sep and out_raw:
    a, b = cos(embed(out_sep), e_gt), cos(embed(out_raw), e_gt)
    print("\n分离参考带来的提升: %+.4f (%s)" % (a - b, "有改善" if a > b else "无改善"))
    print("两个输出彼此相似度: %.4f" % cos(embed(out_sep), embed(out_raw)))
