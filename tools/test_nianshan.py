# -*- coding: utf-8 -*-
"""念山AI 网页应用端到端验证：上传 -> 调用 /convert -> 收产物 -> 客观分析。"""
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
SRC = os.path.join(DEMO_DIR, "test_source_short.wav")
REF = os.path.join(DEMO_DIR, "test_ref.wav")
SAVE_DIR = os.path.join(WORKSPACE, "tools", "webapp_result")

c = httpx.Client(timeout=httpx.Timeout(900.0, connect=30.0))

# 1) 端点与参数
info = c.get(BASE + "/gradio_api/info").json()
named = info.get("named_endpoints") or {}
print("命名端点:", list(named.keys()))
ep = named.get("/convert")
if not ep:
    sys.exit("未找到 /convert 端点")
param_names = [p.get("parameter_name") for p in (ep.get("parameters") or [])]
print("入参顺序:", param_names)

# 2) 上传
paths = []
for p in (SRC, REF):
    with open(p, "rb") as fh:
        r = c.post(BASE + "/gradio_api/upload",
                   files={"files": (os.path.basename(p), fh, "audio/wav")})
    r.raise_for_status()
    paths.append(r.json()[0])
print("已上传:", [os.path.basename(p) for p in paths])

# 3) 按参数名组装 data
file_arg = lambda p: {"path": p, "meta": {"_type": "gradio.FileData"}}
values = {
    "source": file_arg(paths[0]),
    "target": file_arg(paths[1]),
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
print("event_id =", event_id)

raw = []
with c.stream("GET", BASE + f"/gradio_api/call/convert/{event_id}") as s:
    for line in s.iter_lines():
        raw.append(line)
dt = time.time() - t0

events = [l for l in raw if l.startswith("event:")]
print("\n=== 事件: %s  总耗时 %.1f 秒 ===" % ([e[6:].strip() for e in events], dt))

# 4) 提取最终 complete 数据
final = None
for line in raw:
    if line.startswith("data:"):
        payload = line[5:].strip()
        if payload and payload != "null":
            final = payload  # 取最后一个有效 data
if not final:
    sys.exit("没有收到任何数据")

obj = json.loads(final)
print("\n=== 最终输出 ===")
print(json.dumps(obj, ensure_ascii=False)[:600])

# 5) 提取音频路径与状态文本
audio_path, status = None, None


def walk(o):
    global audio_path, status
    if isinstance(o, dict):
        if o.get("meta", {}).get("_type") == "gradio.FileData" and o.get("path"):
            if not o.get("is_stream") and str(o["path"]).lower().endswith(".wav"):
                audio_path = o["path"]
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)
    elif isinstance(o, str) and "转换完成" in o:
        status = o


walk(obj)

print("\n=== 状态文本 ===")
print(status if status else "(未捕获)")

if audio_path and os.path.isfile(audio_path):
    os.makedirs(SAVE_DIR, exist_ok=True)
    dst = os.path.join(SAVE_DIR, "nianshan_out.wav")
    shutil.copy(audio_path, dst)
    print("\n产物: %s -> %s (%d 字节)" % (audio_path, dst, os.path.getsize(dst)))
else:
    sys.exit("未找到输出音频: %s" % audio_path)
