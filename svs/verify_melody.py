# -*- coding: utf-8 -*-
"""验证生成歌声是否跟随 MIDI 乐谱（客观指标）。

思路：解析 MIDI 音符 -> 用 pyin 提取生成音频的 F0 ->
逐音符比较音高（半音）误差，并计算两条音高轮廓的相关系数。
"""
import os
import sys

import numpy as np
import pretty_midi
import soundfile as sf

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_WAV = os.path.join(ROOT, "outputs", "probe_v1.wav")
MIDI = os.path.join(ROOT, "resources", "audios", "female__Rnb_Funk__下等马_clip_001.mid")


def hz_to_midi(f):
    return 69 + 12 * np.log2(np.maximum(f, 1e-6) / 440.0)


def main():
    pm = pretty_midi.PrettyMIDI(MIDI)
    notes = []
    for inst in pm.instruments:
        for n in inst.notes:
            notes.append((n.start, n.end, n.pitch))
    notes.sort()
    print("MIDI 音符数: %d | 时长 %.2f 秒 | 音高范围 %d~%d" % (
        len(notes), pm.get_end_time(), min(n[2] for n in notes), max(n[2] for n in notes)))

    audio, sr = sf.read(OUT_WAV, always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)
    print("生成音频: %.2f 秒 @ %d Hz" % (len(mono) / sr, sr))

    import librosa
    f0, voiced, _ = librosa.pyin(
        mono, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
        sr=sr, frame_length=2048, hop_length=512,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=512)
    print("F0 提取完成: 有声帧占比 %.1f%%" % (100.0 * np.mean(voiced)))

    # 生成音频开头有约 0.1 秒静音间隔（对应中间分隔），补偿后再对齐
    best = None
    for off in np.arange(-0.30, 0.31, 0.05):
        errs, pairs = [], []
        for s, e, p in notes:
            m = (times >= s + off) & (times < e + off) & voiced
            if m.sum() == 0:
                continue
            f = np.median(f0[m])
            if not np.isfinite(f):
                continue
            g = hz_to_midi(f)
            errs.append(g - p)
            pairs.append((p, g))
        if len(errs) >= 5:
            mae = float(np.mean(np.abs(errs)))
            if best is None or mae < best[0]:
                pairs_a = np.array(pairs)
                corr = float(np.corrcoef(pairs_a[:, 0], pairs_a[:, 1])[0, 1])
                best = (mae, off, len(errs), corr, float(np.mean(errs)))

    if best is None:
        print("无法比较（F0 或音符太少）")
        return
    mae, off, cnt, corr, bias = best
    print("\n=== 音高跟随验证（对齐偏移 %.2f 秒）===" % off)
    print("可比对音符数: %d" % cnt)
    print("平均绝对音高误差: %.2f 半音" % mae)
    print("系统性偏差(生成-乐谱): %+.2f 半音" % bias)
    print("音高轮廓相关系数: %.3f" % corr)
    print()
    if mae < 1.0:
        print("结论: 生成歌声的音高与乐谱高度吻合（<1 半音）")
    elif mae < 2.0:
        print("结论: 音高基本跟随乐谱（1~2 半音），有可察觉偏差")
    else:
        print("结论: 音高跟随较差（>2 半音）")


if __name__ == "__main__":
    main()
