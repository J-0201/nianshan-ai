# -*- coding: utf-8 -*-
"""验证：只勾选「参考音频也分离人声」（is_song=False, sep_ref=True）时，
界面应回传「参考音频」的分离人声与伴奏。"""
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
REF = os.path.join(WORKSPACE, "tools", "ref_sep_test\ref_with_music.wav")   # 25 秒带伴奏参考
SAVE = os.path.join(WORKSPACE, "tools", "ref_sep_test")

c = httpx.Client(timeout=httpx.Timeout(1800.0, connect=30.0))
names = [p["parameter_name"] for p in
         c.get(BASE + "/gradio_api/info").json()["named_endpoints"]["/convert"]["parameters"]]


def upload(p):
    with open(p, "rb") as fh:
        return c.post(BASE + "/gradio_api/upload",
                      files={"files": (os.path.basename(p), fh, "audio/wav")}).json()[0]


fd = lambda p: {"path": p, "meta": {"_type": "gradio.FileData"}}
values = {"source": fd(upload(SRC)), "target": fd(upload(REF)),
          "is_song": False, "sep_ref": True, "steps": 30, "pitch": 0,
          "length_adjust": 1.0, "cfg": 0.7, "auto_f0": False}

t0 = time.time()
r = c.post(BASE + "/gradio_api/call/convert", json={"data": [values[n] for n in names]})
eid = r.json()["event_id"]
raw = []
with c.stream("GET", BASE + f"/gradio_api/call/convert/{eid}") as s:
    for line in s.iter_lines():
        raw.append(line)

final = None
for line in raw:
    if line.startswith("data:"):
        pl = line[5:].strip()
        if pl and pl != "null":
            try:
                obj = json.loads(pl)
                if isinstance(obj, list) and len(obj) >= 5 and isinstance(obj[4], str):
                    final = obj
            except Exception:
                pass

print("耗时 %.1f 秒" % (time.time() - t0))
labels = ["成品混音", "转换后人声", "分离出的人声", "分离出的伴奏"]
os.makedirs(SAVE, exist_ok=True)
for i, lbl in enumerate(labels):
    v = final[i] if final else None
    if isinstance(v, dict) and v.get("path") and os.path.isfile(v["path"]):
        info = sf.info(v["path"])
        dst = os.path.join(SAVE, "refonly_%d_%s.wav" % (i, lbl))
        shutil.copy(v["path"], dst)
        print("  %-12s: %.2f 秒 | %d 声道   -> %s" % (lbl, info.duration, info.channels, os.path.basename(dst)))
    else:
        print("  %-12s: (空)" % lbl)
print("\n参照: 参考音频 25.00 秒、源音频 %.2f 秒" % sf.info(SRC).duration)
