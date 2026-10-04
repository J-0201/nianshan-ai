# -*- coding: utf-8 -*-
"""验证「分离试听预览」：只做分离、不转换，并确认后续转换会复用缓存。

流程：
  1. 调 /sep_preview（模拟勾选）→ 应只分离，产出人声+伴奏，且不产生转换结果
  2. 再调 /convert（sep_ref=True，同一参考）→ 状态应显示"复用已预览的参考分离结果"
"""
import json
import os
import shutil
import time

import httpx
import soundfile as sf

# ---------- 路径解析（可用环境变量覆盖）----------
# 仓库根 = 本文件所在目录的上一级；上游项目默认与仓库同级。
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE = os.path.dirname(_REPO)
SVC_DIR = os.environ.get("NS_SEEDVC_DIR") or os.path.join(WORKSPACE, "seed-vc")
SVS_DIR = os.environ.get("NS_SVS_DIR") or os.path.join(WORKSPACE, "YingMusic-Singer")
DEMO_DIR = os.environ.get("NS_DEMO_AUDIO_DIR") or os.path.join(WORKSPACE, "demo_audio")
RESULTS = os.path.join(_REPO, "tools", "results")


BASE = "http://127.0.0.1:7860"
SRC = os.path.join(DEMO_DIR, "test_source_short.wav")
REF = os.path.join(WORKSPACE, "tools", "ref_sep_test\ref_with_music.wav")   # 25 秒带伴奏
SAVE = os.path.join(WORKSPACE, "tools", "ref_sep_test")

c = httpx.Client(timeout=httpx.Timeout(1800.0, connect=30.0))
info = c.get(BASE + "/gradio_api/info").json()["named_endpoints"]
print("端点:", sorted(info.keys()))


def upload(p):
    with open(p, "rb") as fh:
        return c.post(BASE + "/gradio_api/upload",
                      files={"files": (os.path.basename(p), fh, "audio/wav")}).json()[0]


fd = lambda p: {"path": p, "meta": {"_type": "gradio.FileData"}}
up_ref = upload(REF)


def call(ep_name, values, ninfo):
    names = [p["parameter_name"] for p in ninfo[ep_name]["parameters"]]
    data = [values[n] for n in names]
    r = c.post(BASE + f"/gradio_api/call/{ep_name.lstrip('/')}", json={"data": data})
    eid = r.json()["event_id"]
    frames = []
    with c.stream("GET", BASE + f"/gradio_api/call/{ep_name.lstrip('/')}/{eid}") as s:
        for line in s.iter_lines():
            if line.startswith("data:"):
                pl = line[5:].strip()
                if pl and pl != "null":
                    try:
                        frames.append(json.loads(pl))
                    except Exception:
                        pass
    return frames


def find_value(frame, idx):
    """从输出帧里取第 idx 个组件的路径（兼容 value 包装与直接 FileData）。"""
    if not isinstance(frame, list) or len(frame) <= idx:
        return None
    v = frame[idx]
    if isinstance(v, dict):
        if isinstance(v.get("value"), dict) and v["value"].get("path"):
            return v["value"]["path"]
        if v.get("path"):
            return v["path"]
    return None


print("\n=== 步骤 1：调 /sep_preview_run（只分离，不转换）===")
t0 = time.time()
frames = call("/sep_preview_run", {"target": fd(up_ref)}, info)
dt = time.time() - t0
last = frames[-1] if frames else None
v = find_value(last, 0)
a = find_value(last, 1)
print("耗时 %.1f 秒" % dt)
print("  分离人声: %s" % (os.path.basename(v) if v else "(空)"))
print("  分离伴奏: %s" % (os.path.basename(a) if a else "(空)"))
print("  状态: %s" % (last[2].split("\n")[0][:80] if isinstance(last, list) and len(last) > 2 else "?"))
if v and os.path.isfile(v):
    i = sf.info(v)
    shutil.copy(v, os.path.join(SAVE, "preview_vocals.wav"))
    print("  时长 %.2f 秒（参考 25 秒）-> %s" % (i.duration, "正确" if abs(i.duration - 25) < 1 else "异常"))
    gt = os.path.join(SAVE, "gt_vocals.wav")
    if os.path.isfile(gt):
        import numpy as np
        d1, _ = sf.read(v, always_2d=True)
        d2, _ = sf.read(gt, always_2d=True)
        n = min(len(d1), len(d2))
        # 与之前"转换流程内分离"的结果对比，确认是同一套分离结果
        corr = float(np.corrcoef(d1[:n, 0], d2[:n, 0])[0, 1])
        print("  与之前流程内分离结果的波形相关系数: %.4f（应接近 1）" % corr)

print("\n=== 步骤 2：调 /convert（sep_ref=True，同一参考）→ 应复用缓存 ===")
t0 = time.time()
frames2 = call("/convert", {"source": fd(upload(SRC)), "target": fd(up_ref),
                             "is_song": False, "sep_ref": True, "steps": 30,
                             "pitch": 0, "length_adjust": 1.0, "cfg": 0.7,
                             "auto_f0": False}, info)
dt2 = time.time() - t0
st = []
for f in frames2:
    if isinstance(f, list) and len(f) >= 5 and isinstance(f[4], str):
        st.append(f[4])
print("耗时 %.1f 秒" % dt2)
prev = None
for s in st:
    h = s.split("\n")[0]
    if h != prev:
        print("   ", h[:86])
        prev = h
print("\n复用提示是否出现:", "是 ✓" if any("复用" in s for s in st) else "否 ✗")
