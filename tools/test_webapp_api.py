# -*- coding: utf-8 -*-
"""通过 Gradio 的 HTTP API 真实触发一次转换，验证网页应用后端可用。

注意：必须用 /gradio_api/info 里列出的命名端点（这里是 /predict）。
/config 里的 dependencies 还包含 Gradio 的内部 lambda，取第一个会调错东西。
"""
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

# 1) 取正确的命名端点
info = c.get(BASE + "/gradio_api/info").json()
named = list((info.get("named_endpoints") or {}).keys())
print("命名端点:", named)
endpoint = named[0] if named else "/predict"
call_name = endpoint.lstrip("/")
print("使用端点:", endpoint, "->", call_name)

# 2) 上传音频
paths = []
for p in (SRC, REF):
    with open(p, "rb") as fh:
        r = c.post(BASE + "/gradio_api/upload",
                   files={"files": (os.path.basename(p), fh, "audio/wav")})
    r.raise_for_status()
    paths.append(r.json()[0])
print("已上传:", [os.path.basename(p) for p in paths])

# 3) 触发转换（入参顺序与界面 inputs 一致）
#    Gradio 5 要求文件入参是带 meta 标记的字典，不能直接传路径字符串，
#    否则报 "The 'meta' field must be explicitly provided"。
file_arg = lambda p: {"path": p, "meta": {"_type": "gradio.FileData"}}
data = [file_arg(paths[0]), file_arg(paths[1]), 20, 1.0, 0.7, False, 0]
t0 = time.time()
r = c.post(BASE + f"/gradio_api/call/{call_name}", json={"data": data})
r.raise_for_status()
event_id = r.json()["event_id"]
print("event_id =", event_id)

# 4) 读 SSE，完整保留每一行
raw = []
with c.stream("GET", BASE + f"/gradio_api/call/{call_name}/{event_id}") as s:
    for line in s.iter_lines():
        raw.append(line)

elapsed = time.time() - t0
print("\n=== 原始事件流 (%d 行, 耗时 %.1f 秒) ===" % (len(raw), elapsed))
for line in raw:
    print("  " + (line if len(line) <= 400 else line[:400] + " ..."))

# 5) 从结果里提取产物路径
print("\n=== 提取到的文件 ===")
found = []


def walk(o):
    if isinstance(o, dict):
        if "path" in o and isinstance(o["path"], str):
            found.append(o["path"])
        if "url" in o and isinstance(o["url"], str) and "/tmp/" in o["url"]:
            found.append(o["url"])
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)


for line in raw:
    if line.startswith("data:"):
        payload = line[5:].strip()
        if not payload or payload == "null":
            continue
        try:
            walk(json.loads(payload))
        except Exception:
            pass

if found:
    os.makedirs(SAVE_DIR, exist_ok=True)
    for i, f in enumerate(dict.fromkeys(found)):
        src_path = f if os.path.isfile(f) else None
        print("  产物[%d]: %s  存在=%s" % (i, f, bool(src_path)))
        if src_path:
            ext = os.path.splitext(src_path)[1] or ".wav"
            dst = os.path.join(SAVE_DIR, "webapp_out_%d%s" % (i, ext))
            shutil.copy(src_path, dst)
            print("       已另存为: %s (%d 字节)" % (dst, os.path.getsize(dst)))
else:
    print("  未从事件流中提取到文件路径")