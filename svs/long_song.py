# -*- coding: utf-8 -*-
"""YingMusic-Singer 长歌分段生成 + 无缝拼接。

背景
----
模型单次生成有内存上限（本机 RTX 3050 4GB 实测：40 秒可行，55 秒在 VAE 解码
阶段 OOM）。官方 README 也建议总时长不超过约 45 秒。为支持整首歌，本模块：

  1. 把乐谱按"音符间隙"切成若干段（默认每段 30 秒，优先切在乐句之间的空白处，
     避免切断延音音符）；
  2. 按各段音符数量比例切分歌词；
  3. 逐段用同一音色参考生成；
  4. 采样级精确拼接 —— 模型输出恒为「0.08 秒静音 + 旋律」，且旋律长度
     = MIDI 时长 × 25 帧（DEFAULT_VAE_FRAME_HZ=25），因此去掉每段的 0.08 秒
     引导后，各段即可按其起始时间精确落位，交界处再做短线交叉淡化。

单独运行（自测）：
    python long_song.py --midi <路径> --lyrics <歌词> --timbre <音色wav> \
        --timbre-content <音色片段所唱的词> --out <输出wav> [--chunk 30]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from typing import Callable, List, Sequence, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("TORCH_HOME", os.path.join(ROOT, "ckpts", "torch_hub"))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "ckpts", "hf"))

import numpy as np
import pretty_midi
import soundfile as sf
import torch

# 模型输出的固定前导静音：silence_frames = int(0.1 * 25) = 2 帧 = 0.08 秒
LEAD_SILENCE_SEC = 2 / 25.0
# 段间接缝交叉淡化长度
DEFAULT_FADE_MS = 30.0


# --------------------------------------------------------------------------
# 乐谱切分
# --------------------------------------------------------------------------
def load_notes(midi_path: str) -> List[Tuple[float, float, int]]:
    """读取 MIDI，返回按起始时间排序的 (start, end, pitch)。"""
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = [(float(n.start), float(n.end), int(n.pitch))
             for inst in pm.instruments for n in inst.notes]
    notes.sort()
    if not notes:
        raise ValueError("MIDI 中没有音符: %s" % midi_path)
    return notes


def find_cut(notes: Sequence[Tuple[float, float, int]], lo: float, hi: float) -> float:
    """在 [lo, hi] 内寻找最大音符间隙的中点作为切点；找不到则返回 hi。

    切在间隙中间可避免截断延音音符，也让分段边界落在乐句呼吸处。
    """
    best_gap, best_pos = 0.0, None
    for i in range(len(notes) - 1):
        gap_start = notes[i][1]
        gap_end = notes[i + 1][0]
        if gap_end <= lo or gap_start >= hi or gap_end <= gap_start:
            continue
        gs, ge = max(gap_start, lo - 1.0), min(gap_end, hi + 1.0)
        if ge - gs > best_gap:
            best_gap, best_pos = ge - gs, (gap_start + gap_end) / 2.0
    if best_pos is None:
        return hi
    return float(min(max(best_pos, lo), hi))


def plan_chunks(notes: Sequence[Tuple[float, float, int]],
                chunk_sec: float = 30.0,
                search_sec: float = 3.0) -> List[Tuple[float, float]]:
    """把时间轴切成 [(start, end), ...]，每段约 chunk_sec 秒。"""
    total_end = max(n[1] for n in notes)
    bounds = [0.0]
    while total_end - bounds[-1] > chunk_sec * 1.15:
        target = bounds[-1] + chunk_sec
        cut = find_cut(notes, target - search_sec, target + search_sec)
        if cut <= bounds[-1] + 1.0:          # 兜底，避免死循环
            cut = target
        bounds.append(cut)
    bounds.append(total_end)
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def write_chunk_midi(notes: Sequence[Tuple[float, float, int]],
                     start: float, end: float, out_path: str) -> int:
    """把落在 [start, end) 的音符平移到 0 起点写出，返回音符数量。"""
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    count = 0
    for s, e, p in notes:
        if s < start or s >= end:
            continue
        ns = s - start
        ne = min(e, end) - start
        if ne - ns < 0.03:
            continue
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=p, start=ns, end=ne))
        count += 1
    if count == 0:
        return 0
    pm.instruments.append(inst)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    pm.write(out_path)
    return count


# --------------------------------------------------------------------------
# 歌词切分
# --------------------------------------------------------------------------
_CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff]")


def lyric_units(text: str) -> List[str]:
    """把歌词切成"音节单元"：中文按字，西文按词；标点/空白附着到前一个单元。"""
    units: List[str] = []
    buf = ""
    for ch in text:
        if _CJK.match(ch):
            if buf.strip():
                units.append(buf)
            buf = ""
            units.append(ch)
        elif ch.isalnum() or ch in "'’":
            buf += ch
        else:
            if buf:
                units.append(buf)
                buf = ""
            if units:
                units[-1] += ch
            elif ch.strip():
                units.append(ch)
    if buf:
        units.append(buf)
    return units


def split_lyrics(lyrics: str, weights: Sequence[int]) -> List[str]:
    """按权重（各段音符数）把歌词切成 len(weights) 份。"""
    units = lyric_units(lyrics)
    total_w = sum(weights) or 1
    total_u = len(units)
    parts: List[str] = []
    idx = 0
    for i, w in enumerate(weights):
        if i == len(weights) - 1:
            take = total_u - idx
        else:
            take = int(round(total_u * w / total_w))
            take = max(0, min(take, total_u - idx))
        parts.append("".join(units[idx:idx + take]))
        idx += take
    return parts


# --------------------------------------------------------------------------
# 拼接
# --------------------------------------------------------------------------
def _equal_power_fade(n: int) -> Tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
    return np.sin(t * np.pi / 2), np.cos(t * np.pi / 2)


def stitch(pieces: Sequence[Tuple[float, np.ndarray]], sr: int,
           total_len: int, guard_ms: float = DEFAULT_FADE_MS) -> np.ndarray:
    """把 (起始秒, 音频[帧,声道]) 精确落位并平滑拼接。

    说明：切点已选在音符间隙（静音处），各段音频首尾相接、通常没有重叠，
    因此这里不做长交叉淡化（那会与空白淡化、反而削弱音头），而是：
      - 每段入/出口各加一段极短（默认 30ms）等功率淡化，消除拼接咔哒声；
      - 用 `+=` 叠加，若两段恰好有重叠，则自然形成等功率交叉淡化。
    """
    ch = max(p[1].shape[1] for p in pieces) if pieces else 1
    out = np.zeros((total_len, ch), dtype=np.float32)
    guard_n = int(sr * guard_ms / 1000.0)
    n_pieces = len(pieces)
    covered_end = 0            # 已写入内容的末端样本位置

    for idx, (start_sec, audio) in enumerate(pieces):
        s = int(round(start_sec * sr))
        if s >= total_len:
            continue
        a = audio[:, :ch]
        if a.shape[1] < ch:                       # 声道数补齐
            a = np.repeat(a, ch, axis=1)
        e = min(s + len(a), total_len)
        seg = a[:e - s].copy()
        pos = s

        # 与已有内容重叠时做等功率交叉淡化（正常设计下不重叠，
        # 但保留此分支以避免任何情况下出现电平叠加爆音）
        ov = max(0, covered_end - s)
        if ov > 0:
            n = min(ov, len(seg))
            rise, decay = _equal_power_fade(n)
            out[pos:pos + n] = out[pos:pos + n] * decay[:, None] + seg[:n] * rise[:, None]
            seg = seg[n:]
            pos += n

        n = min(guard_n, len(seg) // 2)
        if n > 0:
            rise, decay = _equal_power_fade(n)
            if idx > 0 and ov == 0:
                seg[:n] *= rise[:, None]          # 入口防咔哒（前段未延入此处）
            if idx < n_pieces - 1:
                seg[-n:] *= decay[:, None]        # 出口防咔哒
        out[pos:pos + len(seg)] = seg
        covered_end = max(covered_end, pos + len(seg))
    return out


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def synthesize_long(
    singer,
    timbre_audio_path: str,
    timbre_audio_content: str,
    midi_path: str,
    lyrics: str,
    chunk_sec: float = 30.0,
    pitch_shift: int = 0,
    cfg_strength: float = 4.0,
    nfe_steps: int = 32,
    seed: int = 666,
    fade_ms: float = DEFAULT_FADE_MS,
    work_dir: str | None = None,
    use_cache: bool = False,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> Tuple[np.ndarray, int, dict]:
    """分段合成整首歌。返回 (音频[帧,声道], 采样率, 统计信息)。

    work_dir 中会缓存每段生成结果（chunk_XX.npy），use_cache=True 时命中即复用，
    便于在不重新推理的情况下调整拼接/后处理参数。
    """
    from singer.model import SAMPLE_RATE_48K

    notes = load_notes(midi_path)
    chunks = plan_chunks(notes, chunk_sec=chunk_sec)
    counts = []
    midi_files = []
    work_dir = work_dir or os.path.join(ROOT, "outputs", "chunks")
    os.makedirs(work_dir, exist_ok=True)

    for i, (cs, ce) in enumerate(chunks):
        p = os.path.join(work_dir, "chunk_%02d.mid" % i)
        n = write_chunk_midi(notes, cs, ce, p)
        counts.append(n)
        midi_files.append(p)

    lyric_parts = split_lyrics(lyrics, counts)
    total_notes = len(notes)

    pieces: List[Tuple[float, np.ndarray]] = []
    chunk_stats = []
    for i, ((cs, ce), mp, ly) in enumerate(zip(chunks, midi_files, lyric_parts)):
        n_notes = counts[i]
        msg = "第 %d/%d 段（%.1f~%.1f 秒，%d 个音符）" % (
            i + 1, len(chunks), cs, ce, n_notes)
        if progress_cb:
            progress_cb(i, len(chunks), msg)
        if n_notes == 0:
            chunk_stats.append({"index": i, "notes": 0, "skipped": True})
            continue

        cache_path = os.path.join(work_dir, "chunk_%02d.npy" % i)
        if use_cache and os.path.isfile(cache_path):
            arr = np.load(cache_path)
            dt = 0.0
            if progress_cb:
                progress_cb(i, len(chunks), msg + " —— 命中缓存")
        else:
            t0 = time.time()
            wav = singer.inference(
                timbre_audio_path=timbre_audio_path,
                timbre_audio_content=timbre_audio_content,
                midi_file=mp,
                lyrics=ly,
                pitch_shift=pitch_shift,
                cfg_strength=cfg_strength,
                nfe_steps=nfe_steps,
                seed=seed,
            )
            dt = time.time() - t0
            arr = wav.detach().cpu().float().numpy()
            if arr.ndim == 1:
                arr = arr[:, None]
            elif arr.shape[0] <= 2 and arr.shape[1] > arr.shape[0]:
                arr = arr.T

            # 去掉固定前导静音，使音频起点精确对应本段 MIDI 的 0 时刻
            lead = int(round(LEAD_SILENCE_SEC * SAMPLE_RATE_48K))
            if len(arr) > lead:
                arr = arr[lead:]
            np.save(cache_path, arr)

            # 及时释放显存（本机靠驱动回退到内存，必须主动清理）
            del wav
            torch.cuda.empty_cache()

        pieces.append((cs, arr))
        chunk_stats.append({"index": i, "notes": n_notes, "start": cs, "end": ce,
                            "seconds": len(arr) / SAMPLE_RATE_48K, "gen_sec": dt})

    total_sec = max(n[1] for n in notes)
    total_len = int(round(total_sec * SAMPLE_RATE_48K)) + int(0.5 * SAMPLE_RATE_48K)
    audio = stitch(pieces, SAMPLE_RATE_48K, total_len, guard_ms=fade_ms)
    peak = float(np.abs(audio).max())
    if peak > 0.99:
        audio = audio * (0.99 / peak)

    info = {
        "chunks": len(chunks),
        "total_notes": total_notes,
        "duration": total_len / SAMPLE_RATE_48K,
        "chunk_stats": chunk_stats,
    }
    return audio, SAMPLE_RATE_48K, info


# --------------------------------------------------------------------------
# 命令行自测
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="YingMusic-Singer 长歌分段合成")
    ap.add_argument("--midi", required=True)
    ap.add_argument("--lyrics", required=True)
    ap.add_argument("--timbre", required=True)
    ap.add_argument("--timbre-content", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=float, default=30.0, help="每段秒数")
    ap.add_argument("--pitch-shift", type=int, default=0)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--seed", type=int, default=666)
    ap.add_argument("--fade-ms", type=float, default=DEFAULT_FADE_MS)
    args = ap.parse_args()

    from singer.model import YingSinger

    singer = YingSinger(singer_path=os.path.join(ROOT, "ckpts"), device="cuda")
    t0 = time.time()

    def cb(i, n, msg):
        print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)

    audio, sr, info = synthesize_long(
        singer,
        timbre_audio_path=args.timbre,
        timbre_audio_content=args.timbre_content,
        midi_path=args.midi,
        lyrics=args.lyrics,
        chunk_sec=args.chunk,
        pitch_shift=args.pitch_shift,
        cfg_strength=args.cfg,
        nfe_steps=args.nfe,
        seed=args.seed,
        fade_ms=args.fade_ms,
        progress_cb=cb,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sf.write(args.out, audio, sr)

    print("\n完成: %.2f 秒音频 | 共 %d 段 | 总耗时 %.1f 秒" % (
        info["duration"], info["chunks"], time.time() - t0))
    for c in info["chunk_stats"]:
        if c.get("skipped"):
            print("  段%-2d 跳过（无音符）" % (c["index"] + 1))
        else:
            print("  段%-2d %6.1f~%6.1f 秒 | %3d 音符 | 生成 %.1f 秒 | 耗时 %.1f 秒" % (
                c["index"] + 1, c["start"], c["end"], c["notes"],
                c["seconds"], c["gen_sec"]))
    print("输出: %s" % args.out)


if __name__ == "__main__":
    main()
