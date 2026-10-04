# -*- coding: utf-8 -*-
"""念山AI「乐谱唱歌」模式端到端验证（走 Gradio API）。

验证要点：
  1. /svs 端点存在且参数正确
  2. 上传 MIDI + 音色音频后能跑通，产出音频
  3. 进度事件是多次刷新的（不是一条静态提示）
"""
import json
import os
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
MIDI = os.path.join(SVS_DIR, "resources\audios\female__Rnb_Funk__下等马_clip_001.mid")
TIMBRE = os.path.join(SVS_DIR, "resources\audios\male.wav")
SAVE_DIR = os.path.join(WORKSPACE, "tools", "webapp_result")
LYRICS = (
    "头抬起来，你表情别太奇怪，无大碍。没伤到脑袋，如果我下手太重，私密马赛。"
    "习武十载，没下山没谈恋爱，吃光后山七八亩菜，练就这套拳脚，莫以貌取人哉。"
    "暮色压台，擂鼓未衰，下一个谁还要来？速来领拜，别耽误我热蒸铁揭盖。"
)

c = httpx.Client(timeout=httpx.Timeout(2400.0, connect=30.0))

info = c.get(BASE + "/gradio_api/info").json()
named = info.get("named_endpoints") or {}
print("可用端点:", sorted(named.keys()))
ep = named.get("/svs")
if not ep:
    sys.exit("未找到 /svs 端点")
param_names = [p.get("parameter_name") for p in (ep.get("parameters") or [])]
print("入参顺序:", param_names)

paths = []
for p in (MIDI, TIMBRE):
    with open(p, "rb") as fh:
        r = c.post(BASE + "/gradio_api/upload",
                   files={"files": (os.path.basename(p), fh, "application/octet-stream")})
    r.raise_for_status()
    paths.append(r.json()[0])
print("已上传:", [os.path.basename(p) for p in paths])

fd = lambda p: {"path": p, "meta": {"_type": "gradio.FileData"}}
values = {
    "midi_file": fd(paths[0]),
    "lyrics": LYRICS,
    "timbre": fd(paths[1]),
    "timbre_content": "在爱的回归线，又期待相见。",
    "chunk_sec": 30,
    "pitch_shift": -4,
    "cfg": 4.0,
    "nfe": 32,
}
data = [values[n] for n in param_names]

t0 = time.time()
r = c.post(BASE + "/gradio_api/call/svs", json={"data": data})
r.raise_for_status()
eid = r.json()["event_id"]
print("event_id:", eid)

raw, statuses = [], []
with c.stream("GET", BASE + f"/gradio_api/call/svs/{eid}") as s:
    for line in s.iter_lines():
        raw.append(line)
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload and payload != "null":
                try:
                    obj = json.loads(payload)
                    if isinstance(obj, list) and len(obj) >= 2 and isinstance(obj[1], str):
                        statuses.append(obj[1])
                except Exception:
                    pass

dt = time.time() - t0
events = [l[6:].strip() for l in raw if l.startswith("event:")]
print("\n=== 事件数 %d (generating %d) | 总耗时 %.1f 秒 ===" %
      (len(events), events.count("generating"), dt))

print("\n=== 状态刷新序列（相邻去重）===")
prev = None
shown = 0
for s_ in statuses:
    head = s_.split("\n")[0]
    if head != prev:
        print("   ", head[:90])
        prev = head
        shown += 1
    if shown > 12:
        print("    ...（其余略）")
        break

print("\n=== 最终状态 ===")
print(statuses[-1] if statuses else "(无)")

found = []


def walk(o):
    if isinstance(o, dict):
        if o.get("meta", {}).get("_type") == "gradio.FileData" and isinstance(o.get("path"), str):
            if o["path"].lower().endswith(".wav"):
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
    print("  %s 存在=%s" % (f, ok))
    if ok:
        dst = os.path.join(SAVE_DIR, "nianshan_svs.wav")
        import shutil
        shutil.copy(f, dst)
        print("    -> %s (%d 字节)" % (dst, os.path.getsize(dst)))
