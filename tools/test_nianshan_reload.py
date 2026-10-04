# -*- coding: utf-8 -*-
"""验证显存仲裁的反向路径：乐谱唱歌跑完后，音色转换能否自动重载 Seed-VC。"""
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
SRC = os.path.join(DEMO_DIR, "test_source_short.wav")
REF = os.path.join(DEMO_DIR, "test_ref.wav")

c = httpx.Client(timeout=httpx.Timeout(1800.0, connect=30.0))
ep = (c.get(BASE + "/gradio_api/info").json().get("named_endpoints") or {}).get("/convert")
names = [p.get("parameter_name") for p in ep["parameters"]]
print("入参顺序:", names)

paths = []
for p in (SRC, REF):
    with open(p, "rb") as fh:
        r = c.post(BASE + "/gradio_api/upload",
                   files={"files": (os.path.basename(p), fh, "audio/wav")})
    r.raise_for_status()
    paths.append(r.json()[0])

fd = lambda p: {"path": p, "meta": {"_type": "gradio.FileData"}}
values = {
    "source": fd(paths[0]), "target": fd(paths[1]), "is_song": False,
    "steps": 30, "pitch": 0, "length_adjust": 1.0, "cfg": 0.7, "auto_f0": False,
}
data = [values[n] for n in names]

t0 = time.time()
r = c.post(BASE + "/gradio_api/call/convert", json={"data": data})
r.raise_for_status()
eid = r.json()["event_id"]

raw, statuses = [], []
with c.stream("GET", BASE + f"/gradio_api/call/convert/{eid}") as s:
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
print("\n事件数 %d (generating %d) | 耗时 %.1f 秒" % (len(events), events.count("generating"), dt))

prev = None
for s_ in statuses:
    head = s_.split("\n")[0]
    if head != prev:
        print("   ", head[:80])
        prev = head
print("\n最终状态:")
print(statuses[-1] if statuses else "(无)")
