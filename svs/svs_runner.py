# -*- coding: utf-8 -*-
"""YingMusic-Singer 子进程运行器（供念山AI 网页端调用）。

用法:
    python svs_runner.py <job.json>

job.json 字段:
    midi          乐谱 MIDI 路径
    lyrics        目标歌词
    timbre        音色参考音频路径
    timbre_content 音色参考片段所唱的词
    out           输出 wav 路径
    chunk_sec     每段秒数（默认 30）
    pitch_shift   升降调半音（默认 0）
    cfg           CFG 强度（默认 4.0）
    nfe           扩散步数（默认 32）
    seed          随机种子（默认 666）

输出协议（stdout，每行一条，供父进程解析）:
    PROGRESS|<阶段>|<说明>
    DONE|<输出路径>
    ERROR|<错误信息>

之所以用 JSON 文件而不是命令行参数：Windows 下命令行传中文歌词极易乱码。
"""
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "src"))


def emit(kind, *parts):
    print("%s|%s" % (kind, "|".join(str(p) for p in parts)), flush=True)


def main():
    if len(sys.argv) < 2:
        emit("ERROR", "缺少 job.json 参数")
        return 2
    # 用 utf-8-sig 读取：兼容带 BOM（PowerShell Out-File 会写 BOM）与不带 BOM 的两种 JSON
    with open(sys.argv[1], "r", encoding="utf-8-sig") as fh:
        job = json.load(fh)

    os.environ.setdefault("TORCH_HOME", os.path.join(HERE, "ckpts", "torch_hub"))
    os.environ.setdefault("HF_HOME", os.path.join(HERE, "ckpts", "hf"))
    os.environ.setdefault("PYTHONUTF8", "1")

    import numpy as np
    import soundfile as sf
    import torch

    from long_song import load_notes, plan_chunks, synthesize_long

    # 先规划分段，把总时长/段数报给父进程（用于显示预计耗时）
    try:
        _notes = load_notes(job["midi"])
        _chunks = plan_chunks(_notes, chunk_sec=float(job.get("chunk_sec", 30)))
        _total = max(n[1] for n in _notes)
        emit("PLAN", "%.2f" % _total, "%d" % len(_chunks), "%d" % len(_notes))
    except Exception as exc:
        emit("ERROR", "乐谱解析失败: %s: %s" % (type(exc).__name__, str(exc)[:200]))
        return 6

    t_start = time.time()
    emit("PROGRESS", "load", "正在加载歌声合成模型（约 20 秒）")
    try:
        from singer.model import YingSinger
        singer = YingSinger(singer_path=os.path.join(HERE, "ckpts"), device="cuda")
    except Exception as exc:
        emit("ERROR", "模型加载失败: %s: %s" % (type(exc).__name__, str(exc)[:300]))
        traceback.print_exc()
        return 3

    chunk_sec = float(job.get("chunk_sec", 30))
    last = {"i": 0, "n": 0}
    t_chunks = {"t": time.time()}

    def cb(i, n, msg):
        last["i"], last["n"] = i, n
        t_chunks["t"] = time.time()
        emit("PROGRESS", "chunk", "%d/%d|%s" % (i + 1, n, msg))

    try:
        audio, sr, info = synthesize_long(
            singer,
            timbre_audio_path=job["timbre"],
            timbre_audio_content=job["timbre_content"],
            midi_path=job["midi"],
            lyrics=job["lyrics"],
            chunk_sec=chunk_sec,
            pitch_shift=int(job.get("pitch_shift", 0)),
            cfg_strength=float(job.get("cfg", 4.0)),
            nfe_steps=int(job.get("nfe", 32)),
            seed=int(job.get("seed", 666)),
            progress_cb=cb,
        )
    except torch.cuda.OutOfMemoryError as exc:
        emit("ERROR", "显存不足: %s" % str(exc).split("\n")[0][:200])
        return 4
    except Exception as exc:
        emit("ERROR", "合成失败: %s: %s" % (type(exc).__name__, str(exc)[:300]))
        traceback.print_exc()
        return 5

    out = job["out"]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    sf.write(out, audio, sr)

    emit("PROGRESS", "save", "写入文件")
    emit("STATS",
         "%.1f" % (time.time() - t_start),
         "%d" % info["chunks"],
         "%.2f" % info["duration"],
         "%d" % info["total_notes"])
    emit("DONE", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
