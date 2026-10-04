# -*- coding: utf-8 -*-
"""定位本机可承受的最大单次生成时长（乐谱秒数 -> 能否跑完）。

按目标时长拼接/截断示例乐谱，歌词同比重复。
环境变量：
  TARGET_SEC  目标乐谱时长（秒），默认 40
  PROBE_NFE   扩散步数，默认 32
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

TARGET = float(os.environ.get("TARGET_SEC", "40"))
NFE = int(os.environ.get("PROBE_NFE", "32"))
SRC_MIDI = os.path.join(ROOT, "resources", "audios", "female__Rnb_Funk__下等马_clip_001.mid")
TIMBRE = os.path.join(ROOT, "resources", "audios", "male.wav")
TIMBRE_CONTENT = "在爱的回归线，又期待相见。"
LYRIC_ONE = (
    "头抬起来，你表情别太奇怪，无大碍。没伤到脑袋，如果我下手太重，私密马赛。"
    "习武十载，没下山没谈恋爱，吃光后山七八亩菜，练就这套拳脚，莫以貌取人哉。"
    "暮色压台，擂鼓未衰，下一个谁还要来？速来领拜，别耽误我热蒸铁揭盖。"
)
OUT = os.path.join(ROOT, "outputs", "probe_%ds.wav" % int(TARGET))
MIDI_OUT = os.path.join(ROOT, "outputs", "dur_%ds.mid" % int(TARGET))


def build_midi(target_sec):
    pm = pretty_midi.PrettyMIDI(SRC_MIDI)
    seg = pm.get_end_time()
    out = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    t = 0.0
    while t < target_sec:
        for i in pm.instruments:
            for n in i.notes:
                s, e = n.start + t, n.end + t
                if s >= target_sec:
                    continue
                e = min(e, target_sec)
                if e - s < 0.05:
                    continue
                inst.notes.append(pretty_midi.Note(
                    velocity=n.velocity, pitch=n.pitch, start=s, end=e))
        t += seg
    out.instruments.append(inst)
    os.makedirs(os.path.dirname(MIDI_OUT), exist_ok=True)
    out.write(MIDI_OUT)
    return MIDI_OUT, out.get_end_time()


def main():
    midi_path, real_dur = build_midi(TARGET)
    reps = max(1, int(np.ceil(real_dur / 27.81)))
    lyrics = LYRIC_ONE * reps

    print("=" * 64)
    print("时长上限定位 | 目标 %.0f 秒 | 实际乐谱 %.2f 秒 | 旋律帧数约 %.0f" %
          (TARGET, real_dur, real_dur * 25))
    print("=" * 64)

    singer = YingSinger(singer_path=os.path.join(ROOT, "ckpts"), device="cuda")

    t0 = time.time()
    try:
        wav = singer.inference(
            timbre_audio_path=TIMBRE,
            timbre_audio_content=TIMBRE_CONTENT,
            midi_file=midi_path,
            lyrics=lyrics,
            pitch_shift=-1,
            cfg_strength=4.0,
            nfe_steps=NFE,
        )
    except torch.cuda.OutOfMemoryError as exc:
        print("\n[OOM] %.0f 秒不可行" % TARGET)
        print("  PyTorch 分配峰值 %.0f MiB" % (torch.cuda.max_memory_allocated() / 1024**2))
        print("  %s" % str(exc).split("\n")[0][:200])
        sys.exit(3)

    dt = time.time() - t0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    arr = wav.detach().cpu().float().numpy()
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.shape[0] <= 2 and arr.shape[1] > arr.shape[0]:
        arr = arr.T
    sf.write(OUT, arr, SAMPLE_RATE_48K)
    dur = arr.shape[0] / SAMPLE_RATE_48K

    print("\n[OK] %.0f 秒可行" % TARGET)
    print("  耗时 %.1f 秒 | 产出 %.2f 秒 | RTF %.2f" % (dt, dur, dt / max(dur, 1e-6)))
    print("  PyTorch 分配峰值 %.0f MiB | 保留峰值 %.0f MiB" %
          (torch.cuda.max_memory_allocated() / 1024**2,
           torch.cuda.max_memory_reserved() / 1024**2))
    print("  OUTPUT=%s" % OUT)


if __name__ == "__main__":
    main()
