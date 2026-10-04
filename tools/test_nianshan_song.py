# -*- coding: utf-8 -*-
"""念山AI 歌曲模式端到端验证：30 秒歌曲片段 -> 分离 -> 转换 -> 混音。"""
import json
import os
import shutil
import sys
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
SRC = os.path.join(DEMO_DIR, "song_clip30.wav")
REF = os.path.join(DEMO_DIR, "test_ref.wav")
SAVE_DIR = os.path.join(WORKSPACE, "tools", "webapp_result")

c = httpx.Client(timeout=httpx.Timeout(1200.0, connect=30.0))

info = c.get(BASE + "/gradio_api/info").json()
ep = (info.get("named_endpoints") or {}).get("/convert")
if not ep:
    sys.exit("未找到 /convert 端点")
param_names = [p.get("parameter_name") for p in (ep.get("parameters") or [])]
print("入参顺序:", param_names)

paths = []
for p in (SRC, REF):
    with open(p, "rb") as fh:
        r = c.post(BASE + "/gradio_api/upload",
                   files={"files": (os.path.basename(p), fh, "audio/wav")})
    r.raise_for_status()
    paths.append(r.json()[0])
print("已上传:", [os.path.basename(p) for p in paths])

file_arg = lambda p: {"path": p, "meta": {"_type": "gradio.FileData"}}
values = {
    "source": file_arg(paths[0]),
    "target": file_arg(paths[1]),
    "is_song": True,
    "steps": 30,
    "pitch": 0,
    "length_adjust": 1.0,
    "cfg": 0.7,
    "auto_f0": False,
}
data = [values[n] for n in param_names]

t0 = time.time()
r = c.post(BASE + "/gradio_api/call/convert", json={"data": data})
r.raise_for_status()
event_id = r.json()["event_id"]

raw = []
statuses = []
with c.stream("GET", BASE + f"/gradio_api/call/convert/{event_id}") as s:
    for line in s.iter_lines():
        raw.append(line)
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload and payload != "null":
                try:
                    obj = json.loads(payload)
                    if isinstance(obj, list) and len(obj) >= 3 and isinstance(obj[2], str):
                        statuses.append(obj[2])
                except Exception:
                    pass

dt = time.time() - t0
events = [l[6:].strip() for l in raw if l.startswith("event:")]
print("\n=== 事件数: %d (generating=%d)  总耗时 %.1f 秒 ===" %
      (len(events), events.count("generating"), dt))

print("\n=== 状态刷新序列（去重相邻） ===")
prev = None
for s_ in statuses:
    head = s_.split("\n")[0]
    if head != prev:
        print("  ", head[:80])
        prev = head

final = statuses[-1] if statuses else ""
print("\n=== 最终状态 ===")
print(final)

# 提取两个 wav
found = []


def walk(o):
    if isinstance(o, dict):
        if o.get("meta", {}).get("_type") == "gradio.FileData" and isinstance(o.get("path"), str):
            if o["path"].lower().endswith(".wav") and not o.get("is_stream"):
                found.append(o["path"])
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)


for line in raw:
    if line.startswith("data:"):
        payload = line[5:].strip()
        if payload and payload != "null":
            try:
                walk(json.loads(payload))
            except Exception:
                pass

print("\n=== 产物 ===")
os.makedirs(SAVE_DIR, exist_ok=True)
for f in dict.fromkeys(found):
    ok = os.path.isfile(f)
    print("  %s  存在=%s" % (f, ok))
    if ok:
        tag = "song" if "song" in os.path.basename(f) else ("vocal" if "vocal" in os.path.basename(f) else "out")
        dst = os.path.join(SAVE_DIR, "nianshan_%s.wav" % tag)
        shutil.copy(f, dst)
        print("    -> %s (%d 字节)" % (dst, os.path.getsize(dst)))
