# -*- coding: utf-8 -*-
"""验证：预览产物只进临时目录 + 保存按钮按需留存。"""
import json
import os
import time

import httpx

# ---------- 路径解析（可用环境变量覆盖）----------
# 仓库根 = 本文件所在目录的上一级；上游项目默认与仓库同级。
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE = os.path.dirname(_REPO)
SVC_DIR = os.environ.get("NS_SEEDVC_DIR") or os.path.join(WORKSPACE, "seed-vc")
SVS_DIR = os.environ.get("NS_SVS_DIR") or os.path.join(WORKSPACE, "YingMusic-Singer")
DEMO_DIR = os.environ.get("NS_DEMO_AUDIO_DIR") or os.path.join(WORKSPACE, "demo_audio")
RESULTS = os.path.join(_REPO, "tools", "results")


BASE = "http://127.0.0.1:7860"
SVC = SVC_DIR
WORK = os.path.join(SVC, "work")
SAVED = os.path.join(SVC, "out", "saved")
REF = os.path.join(WORKSPACE, "tools", "ref_sep_test\ref_with_music.wav")

c = httpx.Client(timeout=httpx.Timeout(900.0, connect=30.0))
info = c.get(BASE + "/gradio_api/info").json()["named_endpoints"]
print("端点:", sorted(info.keys()))


def upload(p):
    with open(p, "rb") as fh:
        return c.post(BASE + "/gradio_api/upload",
                      files={"files": (os.path.basename(p), fh, "audio/wav")}).json()[0]


def call(ep, values):
    names = [p["parameter_name"] for p in info[ep]["parameters"]]
    r = c.post(BASE + f"/gradio_api/call/{ep.lstrip('/')}",
               json={"data": [values[n] for n in names]})
    eid = r.json()["event_id"]
    frames = []
    with c.stream("GET", BASE + f"/gradio_api/call/{ep.lstrip('/')}/{eid}") as s:
        for line in s.iter_lines():
            if line.startswith("data:"):
                pl = line[5:].strip()
                if pl and pl != "null":
                    try:
                        frames.append(json.loads(pl))
                    except Exception:
                        pass
    return frames


def stem_path(frame, idx):
    if not isinstance(frame, list) or len(frame) <= idx:
        return None
    v = frame[idx]
    if isinstance(v, dict):
        if isinstance(v.get("value"), dict):
            return v["value"].get("path")
        return v.get("path")
    return None


fd = lambda p: {"path": p, "meta": {"_type": "gradio.FileData"}}

# 记录测试前的状态
before_work = set(os.listdir(WORK)) if os.path.isdir(WORK) else set()
before_saved = set(os.listdir(SAVED)) if os.path.isdir(SAVED) else set()
print("\n测试前 work/ 内容: %s" % (sorted(before_work) or "（空）"))
print("测试前 out/saved/ 内容: %s" % (sorted(before_saved) or "（空）"))

print("\n=== 步骤 1：分离预览 ===")
t0 = time.time()
frames = call("/sep_preview_run", {"target": fd(upload(REF))})
v, a = stem_path(frames[-1], 0), stem_path(frames[-1], 1)
print("耗时 %.1f 秒" % (time.time() - t0))
print("  人声: %s" % v)
print("  伴奏: %s" % a)
print("  界面拿到的路径属于 Gradio 临时区（正常）：%s" % (os.path.basename(os.path.dirname(v))))
preview_dirs = [d for d in os.listdir(WORK) if d.startswith("preview_")]
files_in_work = []
for d in preview_dirs:
    files_in_work += [os.path.join(d, f) for f in os.listdir(os.path.join(WORK, d))]
print("  临时目录 work/ 内的分离文件: %s" % files_in_work)
print("  永久 out/ 下是否新增 sep_* 目录: %s" % (
    "是 ✗" if [d for d in os.listdir(os.path.join(SVC,"out")) if d.startswith("sep_")] else "否 ✓"))
print("  预览后 work/ 内容: %s" % sorted(os.listdir(WORK)))
print("  预览后 out/saved/ 内容: %s" % (sorted(os.listdir(SAVED)) if os.path.isdir(SAVED) else "（目录尚未创建，符合预期）"))

print("\n=== 步骤 2：点「💾 保存分离结果」 ===")
frames2 = call("/save_stems", {})
msg = frames2[-1][0] if isinstance(frames2[-1], list) else str(frames2[-1])
print(msg[:400])
after_saved = set(os.listdir(SAVED)) if os.path.isdir(SAVED) else set()
new_dirs = sorted(after_saved - before_saved)
print("\n新增保存目录: %s" % (new_dirs or "（无）"))
for d in new_dirs:
    for f in sorted(os.listdir(os.path.join(SAVED, d))):
        p = os.path.join(SAVED, d, f)
        print("   %s  %.2f MB" % (f, os.path.getsize(p) / 1024**2))
