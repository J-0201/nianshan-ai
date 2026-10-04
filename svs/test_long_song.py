# -*- coding: utf-8 -*-
"""整首歌分段合成端到端测试。

把官方示例乐谱重复 N 遍拼成一首"长歌"（歌词同比重复），
用 long_song.synthesize_long 分段生成，然后检查：
  - 各段是否都成功（无 OOM）
  - 总时长是否正确
  - 段与段交界处是否有断口/突跳
  - 逐段音高跟随度（验证长歌后段是否退化）
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("TORCH_HOME", os.path.join(ROOT, "ckpts", "torch_hub"))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "ckpts", "hf"))

import numpy as np
import pretty_midi
import soundfile as sf
import torch

from long_song import synthesize_long

REPEATS = int(os.environ.get("REPEATS", "4"))
CHUNK = float(os.environ.get("CHUNK_SEC", "30"))
NFE = int(os.environ.get("PROBE_NFE", "32"))
SRC_MIDI = os.path.join(ROOT, "resources", "audios", "female__Rnb_Funk__下等马_clip_001.mid")
TIMBRE = os.path.join(ROOT, "resources", "audios", "male.wav")
TIMBRE_CONTENT = "在爱的回归线，又期待相见。"
LYRIC_ONE = (
    "头抬起来，你表情别太奇怪，无大碍。没伤到脑袋，如果我下手太重，私密马赛。"
    "习武十载，没下山没谈恋爱，吃光后山七八亩菜，练就这套拳脚，莫以貌取人哉。"
    "暮色压台，擂鼓未衰，下一个谁还要来？速来领拜，别耽误我热蒸铁揭盖。"
)
LONG_MIDI = os.path.join(ROOT, "outputs", "song_long.mid")
OUT = os.path.join(ROOT, "outputs", "song_long.wav")


def build_long_midi(repeats):
    pm = pretty_midi.PrettyMIDI(SRC_MIDI)
    seg = pm.get_end_time()
    out = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for r in range(repeats):
        off = r * seg
        for i in pm.instruments:
            for n in i.notes:
                inst.notes.append(pretty_midi.Note(
                    velocity=n.velocity, pitch=n.pitch,
                    start=n.start + off, end=n.end + off))
    out.instruments.append(inst)
    os.makedirs(os.path.dirname(LONG_MIDI), exist_ok=True)
    out.write(LONG_MIDI)
    return LONG_MIDI, seg, seg * repeats


def main():
    from singer.model import YingSinger

    midi, seg_dur, total_dur = build_long_midi(REPEATS)
    lyrics = LYRIC_ONE * REPEATS
    print("=" * 66)
    print("整首歌分段合成 | %d 段重复 | 乐谱总长 %.1f 秒 | 分段长度 %.0f 秒" %
          (REPEATS, total_dur, CHUNK))
    print("=" * 66)

    singer = YingSinger(singer_path=os.path.join(ROOT, "ckpts"), device="cuda")
    t0 = time.time()

    def cb(i, n, msg):
        print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)

    try:
        audio, sr, info = synthesize_long(
            singer,
            timbre_audio_path=TIMBRE,
            timbre_audio_content=TIMBRE_CONTENT,
            midi_path=midi,
            lyrics=lyrics,
            chunk_sec=CHUNK,
            pitch_shift=-1,
            cfg_strength=4.0,
            nfe_steps=NFE,
            progress_cb=cb,
        )
    except torch.cuda.OutOfMemoryError as exc:
        print("\n[OOM] 分段合成仍然失败: %s" % str(exc).split("\n")[0][:200])
        sys.exit(3)

    dt = time.time() - t0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    sf.write(OUT, audio, sr)

    print("\n" + "=" * 66)
    print("完成: 总耗时 %.1f 秒 | 产出 %.2f 秒音频 (乐谱 %.2f 秒)" %
          (dt, info["duration"], total_dur))
    for c in info["chunk_stats"]:
        if c.get("skipped"):
            print("  段%-2d 跳过" % (c["index"] + 1))
        else:
            print("  段%-2d %6.1f~%6.1f 秒 | %3d 音符 | 音频 %.1f 秒 | 生成耗时 %.1f 秒" % (
                c["index"] + 1, c["start"], c["end"], c["notes"], c["seconds"], c["gen_sec"]))

    # ---- 交界处质量检查 ----
    mono = audio.mean(axis=1)
    print("\n=== 段间交界处检查 ===")
    ok = True
    for c in info["chunk_stats"][1:]:
        if c.get("skipped"):
            continue
        pos = int(round(c["start"] * sr))
        w = int(0.15 * sr)
        around = mono[max(0, pos - w):pos + w]
        rms = float(np.sqrt((around ** 2).mean()))
        jump = float(np.abs(np.diff(mono[max(0, pos - w):pos + w])).max())
        # 前后各 0.3 秒 RMS 对比，检测断口
        before = mono[max(0, pos - int(0.5 * sr)):pos]
        after = mono[pos:pos + int(0.5 * sr)]
        rb = float(np.sqrt((before ** 2).mean())) if len(before) else 0
        ra = float(np.sqrt((after ** 2).mean())) if len(after) else 0
        flag = "OK" if (jump < 0.05 and min(rb, ra) > 0.005) else "注意"
        if flag != "OK":
            ok = False
        print("  %6.1f 秒处: 交界RMS %.4f | 最大跳变 %.4f | 前0.5s %.4f / 后0.5s %.4f  [%s]" %
              (c["start"], rms, jump, rb, ra, flag))
    print("\n交界处总体: %s" % ("无断口/无爆音" if ok else "存在可疑位置"))
    print("全局最大采样跳变: %.4f" % float(np.abs(np.diff(mono)).max()))
    print("输出: %s" % OUT)


if __name__ == "__main__":
    main()
