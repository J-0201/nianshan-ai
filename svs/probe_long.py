# -*- coding: utf-8 -*-
"""实测 YingMusic-Singer 的时长上限与长序列退化情况。

做法：把官方示例 MIDI 重复 N 次拼成长乐谱（歌词同比重复），
生成后按"每个重复段"分别验证音高跟随度——
若后段精度明显下降，说明长序列会退化，必须靠分段生成解决。

环境变量：
  REPEATS   重复段数（默认 2，即约 55 秒）
  PROBE_NFE 扩散步数（默认 32）
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("TORCH_HOME", os.path.join(ROOT, "ckpts", "torch_hub"))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "ckpts", "hf"))

import numpy as np
import pretty_midi
import soundfile as sf
import torch

from singer.model import SAMPLE_RATE_48K, YingSinger

REPEATS = int(os.environ.get("REPEATS", "2"))
NFE = int(os.environ.get("PROBE_NFE", "32"))
SRC_MIDI = os.path.join(ROOT, "resources", "audios", "female__Rnb_Funk__下等马_clip_001.mid")
TIMBRE = os.path.join(ROOT, "resources", "audios", "male.wav")
TIMBRE_CONTENT = "在爱的回归线，又期待相见。"
LYRIC_ONE = (
    "头抬起来，你表情别太奇怪，无大碍。没伤到脑袋，如果我下手太重，私密马赛。"
    "习武十载，没下山没谈恋爱，吃光后山七八亩菜，练就这套拳脚，莫以貌取人哉。"
    "暮色压台，擂鼓未衰，下一个谁还要来？速来领拜，别耽误我热蒸铁揭盖。"
)
OUT = os.path.join(ROOT, "outputs", "probe_long_%d.wav" % REPEATS)
MIDI_OUT = os.path.join(ROOT, "outputs", "long_%d.mid" % REPEATS)


def build_long_midi(repeats):
    """把示例 MIDI 依次平移拼接，返回 (路径, 单段时长, 总时长)。"""
    pm = pretty_midi.PrettyMIDI(SRC_MIDI)
    seg_dur = pm.get_end_time()
    out = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for r in range(repeats):
        off = r * seg_dur
        for i in pm.instruments:
            for n in i.notes:
                inst.notes.append(pretty_midi.Note(
                    velocity=n.velocity, pitch=n.pitch,
                    start=n.start + off, end=n.end + off))
    out.instruments.append(inst)
    os.makedirs(os.path.dirname(MIDI_OUT), exist_ok=True)
    out.write(MIDI_OUT)
    return MIDI_OUT, seg_dur, seg_dur * repeats


def main():
    midi_path, seg_dur, total_dur = build_long_midi(REPEATS)
    lyrics = LYRIC_ONE * REPEATS

    print("=" * 64)
    print("长序列探测 | 重复 %d 段 | 单段 %.2f 秒 | 乐谱总长 %.2f 秒" %
          (REPEATS, seg_dur, total_dur))
    print("预计旋律帧数 %.0f（上限 4096）" % (total_dur * 25))
    print("=" * 64)

    device = "cuda"
    singer = YingSinger(singer_path=os.path.join(ROOT, "ckpts"), device=device)

    t0 = time.time()
    wav = singer.inference(
        timbre_audio_path=TIMBRE,
        timbre_audio_content=TIMBRE_CONTENT,
        midi_file=midi_path,
        lyrics=lyrics,
        pitch_shift=-1,
        cfg_strength=4.0,
        nfe_steps=NFE,
    )
    dt = time.time() - t0

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    arr = wav.detach().cpu().float().numpy()
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.shape[0] <= 2 and arr.shape[1] > arr.shape[0]:
        arr = arr.T
    sf.write(OUT, arr, SAMPLE_RATE_48K)
    dur = arr.shape[0] / SAMPLE_RATE_48K

    peak = torch.cuda.max_memory_allocated() / 1024**2
    reserved = torch.cuda.max_memory_reserved() / 1024**2
    print("\n生成完成: %.1f 秒耗时 | 产出 %.2f 秒音频" % (dt, dur))
    print("期望时长 %.2f 秒 | 实际 %.2f 秒 | 差异 %+.2f 秒" %
          (total_dur, dur, dur - total_dur))
    print("PyTorch 显存 分配峰值 %.0f MiB / 保留峰值 %.0f MiB" % (peak, reserved))
    print("RTF %.2f" % (dt / max(dur, 1e-6)))
    print("OUTPUT=%s" % OUT)
    print("SEG=%.4f" % seg_dur)
    print("MELODY_OFFSET=0.1")


if __name__ == "__main__":
    main()
