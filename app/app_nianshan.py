# -*- coding: utf-8 -*-
"""
念山AI · 歌声转换  —— 本地网页应用

在官方 app_svc.py 的后端（load_models / voice_conversion）之上，
提供布局完善、中文文案、默认值正确的自定义界面。

启动方式：通过 run_webapp.ps1（已固化工作目录、缓存、代理、端口等）。
"""

import argparse
import os
import shutil
import time

import torch

# ---------- 路径解析 ----------
# 部署后的目录布局（由 install.ps1 生成，也可自行摆放后用环境变量覆盖）：
#   <workspace>/
#     ├── seed-vc/            上游 Seed-VC（本项目文件被复制进此目录）
#     ├── YingMusic-Singer/   上游 YingMusic-Singer
#     ├── tools/ffmpeg/bin/   ffmpeg 静态构建（提供 ffmpeg.exe / ffprobe.exe）
#     └── demo_audio/         可选：界面示例音频
_HERE = os.path.dirname(os.path.abspath(__file__))        # .../seed-vc
_WORKSPACE = os.path.dirname(_HERE)                       # 上游项目所在目录

# 环境变量（都可选，用于自定义布局）：
#   NS_SVS_DIR        YingMusic-Singer 目录
#   NS_FFMPEG_DIR     含 ffmpeg.exe 的目录
#   NS_DEMO_AUDIO_DIR 示例音频目录
SVS_ROOT = os.environ.get("NS_SVS_DIR") or os.path.join(_WORKSPACE, "YingMusic-Singer")
FFMPEG_BIN = os.environ.get("NS_FFMPEG_DIR") or os.path.join(_WORKSPACE, "tools", "ffmpeg", "bin")
DEMO_AUDIO_DIR = os.environ.get("NS_DEMO_AUDIO_DIR") or os.path.join(_WORKSPACE, "demo_audio")

# ---------- 修复 1: torchaudio fbank 在 CUDA 上崩溃（详见 svc_launch.py）----------
import torchaudio.compliance.kaldi as _kaldi

_orig_fbank = _kaldi.fbank


def _fbank_cpu_safe(waveform, *args, **kwargs):
    if torch.is_tensor(waveform) and waveform.is_cuda:
        return _orig_fbank(waveform.cpu(), *args, **kwargs).to(waveform.device)
    return _orig_fbank(waveform, *args, **kwargs)


_kaldi.fbank = _fbank_cpu_safe

# ---------- 修复 2: 把 ffmpeg 指给 pydub（官方代码用它编码 mp3）----------
_FFMPEG_EXE = os.path.join(FFMPEG_BIN, "ffmpeg.exe")
if os.path.isfile(_FFMPEG_EXE):
    os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("FFMPEG_BINARY", _FFMPEG_EXE)
    os.environ.setdefault("FFPROBE_BINARY", os.path.join(FFMPEG_BIN, "ffprobe.exe"))
    from pydub import AudioSegment

    AudioSegment.converter = _FFMPEG_EXE
    AudioSegment.ffmpeg = _FFMPEG_EXE
    AudioSegment.ffprobe = os.path.join(FFMPEG_BIN, "ffprobe.exe")
else:
    # ffmpeg 是硬依赖（上游 app_svc 用它编码 mp3）。缺失时给出明确指引而不是莫名报错。
    print("[警告] 未找到 ffmpeg: %s" % _FFMPEG_EXE)
    print("        请运行 install.ps1 自动下载，或设置环境变量 NS_FFMPEG_DIR 指向含 ffmpeg.exe 的目录。")

import gradio as gr
import numpy as np
import soundfile as sf

# ---------- 复用官方后端 ----------
import app_svc

OUT_DIR = os.path.join(_HERE, "out")
os.makedirs(OUT_DIR, exist_ok=True)

# 临时工作区：分离产物默认只放这里，应用启动时清空（不做永久保存）。
# 用户若要留存，用界面上的「💾 保存分离结果」按钮另存到 out\saved\<时间戳>\。
WORK_DIR = os.path.join(_HERE, "work")
if os.path.isdir(WORK_DIR):
    shutil.rmtree(WORK_DIR, ignore_errors=True)
os.makedirs(WORK_DIR, exist_ok=True)
SAVE_DIR = os.path.join(OUT_DIR, "saved")

# 当前展示给用户的分离结果（供「保存」按钮使用）
_last_stems = {"vocals": None, "accomp": None, "source": ""}

# demucs 模型缓存也放 F 盘（默认会写 C 盘用户目录，且本机会因此报错）
os.environ.setdefault("TORCH_HOME", os.path.join(_HERE, "checkpoints", "torch_hub"))

_args = argparse.Namespace(fp16=True, checkpoint=None, config=None, share=False, gpu=0)
app_svc.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed-VC 的模型全局量（模块级），用于加载/卸载
_SEEDVC_GLOBALS = (
    "model_f0", "semantic_fn", "vocoder_fn", "campplus_model",
    "to_mel_f0", "mel_fn_args", "f0_fn",
)
_seedvc_loaded = False


def load_seedvc():
    """加载 Seed-VC 模型（首次启动与从乐谱唱歌切回时调用）。"""
    global _seedvc_loaded
    if _seedvc_loaded:
        return
    (
        app_svc.model_f0,
        app_svc.semantic_fn,
        app_svc.vocoder_fn,
        app_svc.campplus_model,
        app_svc.to_mel_f0,
        app_svc.mel_fn_args,
        app_svc.f0_fn,
    ) = app_svc.load_models(_args)
    app_svc.max_context_window = app_svc.sr // app_svc.hop_length * 30
    app_svc.overlap_wave_len = app_svc.overlap_frame_len * app_svc.hop_length
    _seedvc_loaded = True
    print("念山AI 模型加载完成: sr=%d, device=%s" % (app_svc.sr, app_svc.device))


def unload_seedvc():
    """释放 Seed-VC 全部模型，把 4GB 显存让给乐谱唱歌子进程。

    本机只有 4GB 显存，Seed-VC（约 2.7GB）与 YingMusic-Singer（约 2.7GB）
    无法共存，必须互斥使用。
    """
    global _seedvc_loaded
    if not _seedvc_loaded:
        return
    import gc

    for name in _SEEDVC_GLOBALS:
        setattr(app_svc, name, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    _seedvc_loaded = False
    print("已释放 Seed-VC 模型，显存交还给乐谱唱歌")


load_seedvc()


# ---------- 人声分离与混音 ----------
_demucs_model = None


def separate_vocals(song_path, work_dir):
    """Demucs 分离。返回 (人声路径, 伴奏路径)，均为 44.1kHz WAV。

    实测（本机，应用常驻显存约 3GB 时）：30 秒片段 GPU 分离 7.5 秒，
    全卡峰值 3856 MiB，4GB 可承受。
    """
    global _demucs_model
    from demucs.apply import apply_model
    from demucs.audio import AudioFile, save_audio
    from demucs.pretrained import get_model

    if _demucs_model is None:
        _demucs_model = get_model("htdemucs")
        _demucs_model.eval()
        if torch.cuda.is_available():
            _demucs_model.cuda()

    model = _demucs_model
    wav = AudioFile(song_path).read(streams=0, samplerate=model.samplerate,
                                    channels=model.audio_channels)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        sources = apply_model(model, wav[None], device=device, split=True,
                              overlap=0.25, progress=False)[0]
    sources = sources * ref.std() + ref.mean()
    mix_denorm = wav * ref.std() + ref.mean()

    vocals = sources[model.sources.index("vocals")]
    accomp = mix_denorm - vocals

    os.makedirs(work_dir, exist_ok=True)
    vocals_path = os.path.join(work_dir, "vocals.wav")
    accomp_path = os.path.join(work_dir, "accompaniment.wav")
    save_audio(vocals.cpu(), vocals_path, model.samplerate)
    save_audio(accomp.cpu(), accomp_path, model.samplerate)

    del sources, vocals, accomp, mix_denorm, wav
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return vocals_path, accomp_path


def mix_tracks(vocal_wave, vocal_sr, accomp_path, out_path):
    """把转换后人声叠回伴奏，峰值规整到 0.95 防削波。返回混音文件路径。"""
    accomp, sr_a = sf.read(accomp_path, always_2d=True)
    v = np.asarray(vocal_wave, dtype=np.float32)
    if sr_a != vocal_sr:
        import librosa
        v = librosa.resample(v, orig_sr=vocal_sr, target_sr=sr_a)
    n = min(len(v), accomp.shape[0])
    mixed = accomp[:n] + v[:n, None]
    peak = np.abs(mixed).max()
    if peak > 0.95:
        mixed = mixed * (0.95 / peak)
    sf.write(out_path, mixed, sr_a)
    return out_path


# ---------- 乐谱唱歌（YingMusic-Singer，子进程隔离）----------
# 两套环境不兼容（本应用 py3.10+torch2.4，YingMusic 需 py3.12+torch2.9），
# 且 4GB 显存装不下两个模型，故：调用前卸载 Seed-VC，用子进程跑合成。
_SVS_ROOT = SVS_ROOT
_SVS_PY = os.path.join(_SVS_ROOT, ".venv", "Scripts", "python.exe")
_SVS_RUNNER = os.path.join(_SVS_ROOT, "svs_runner.py")


def svs_generate(midi_file, lyrics, timbre, timbre_content, chunk_sec, pitch_shift,
                 cfg, nfe, progress=gr.Progress()):
    """乐谱唱歌：MIDI + 歌词 + 音色音频 -> 歌声。yield (音频路径或None, 状态Markdown)。"""
    import json
    import queue
    import subprocess
    import threading

    if not midi_file or not timbre:
        yield None, "⚠️ **请上传「乐谱 MIDI」和「音色音频」**"
        return
    if not (lyrics or "").strip():
        yield None, "⚠️ **请填写要唱的歌词**"
        return
    if not (timbre_content or "").strip():
        yield None, (
            "⚠️ **请填写「音色片段所唱的词」**\n\n"
            "模型会把这段词与目标歌词拼成一个文本序列来对齐，缺失会导致咬字错误。"
        )
        return
    if not os.path.isfile(_SVS_PY) or not os.path.isfile(_SVS_RUNNER):
        yield None, (
            "❌ **未找到乐谱唱歌运行环境**\n\n"
            "缺少 `%s`\n\n请先完成 YingMusic-Singer 部署。" % _SVS_RUNNER
        )
        return

    ts = time.strftime("%Y%m%d_%H%M%S")
    job_path = os.path.join(OUT_DIR, "svs_job_%s.json" % ts)
    out_path = os.path.join(OUT_DIR, "svs_%s.wav" % ts)
    with open(job_path, "w", encoding="utf-8") as fh:
        json.dump({
            "midi": os.path.abspath(midi_file),
            "lyrics": lyrics,
            "timbre": os.path.abspath(timbre),
            "timbre_content": timbre_content,
            "out": out_path,
            "chunk_sec": float(chunk_sec),
            "pitch_shift": int(pitch_shift),
            "cfg": float(cfg),
            "nfe": int(nfe),
            "seed": 666,
        }, fh, ensure_ascii=False, indent=2)

    # 4GB 显存互斥：先把 Seed-VC 全部卸掉
    yield None, "🔄 **正在释放音色转换模型，将显存让给歌声合成…**"
    unload_seedvc()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = os.path.join(_SVS_ROOT, "src")
    env["TORCH_HOME"] = os.path.join(_SVS_ROOT, "ckpts", "torch_hub")
    env["HF_HOME"] = os.path.join(_SVS_ROOT, "ckpts", "hf")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"

    proc = subprocess.Popen(
        [_SVS_PY, _SVS_RUNNER, job_path],
        cwd=_SVS_ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )

    lines = queue.Queue()

    def _reader():
        try:
            for raw in proc.stdout:
                lines.put(raw.rstrip("\r\n"))
        finally:
            lines.put(None)

    threading.Thread(target=_reader, daemon=True).start()

    t0 = time.time()
    est_total = 0.0
    n_chunks = 0
    total_sec = 0.0
    total_notes = 0
    stage = "启动中"
    chunk_msg = ""
    done_path = None
    error = None
    stats = {}
    last_ui = 0.0           # 上次刷新界面的时刻（避免过密重绘导致闪烁）

    while True:
        try:
            line = lines.get(timeout=1.0)
        except queue.Empty:
            line = ""

        if line is None:
            break
        if line.startswith("PLAN|"):
            p = line.split("|")
            total_sec, n_chunks, total_notes = float(p[1]), int(p[2]), int(p[3])
            # 经验公式：模型加载约 22 秒 + 每段生成约 1.75×段时长
            est_total = 22.0 + total_sec * 1.75
        elif line.startswith("PROGRESS|"):
            p = line.split("|")
            stage = p[2] if len(p) > 2 else stage
            if len(p) > 3:
                chunk_msg = p[3]
        elif line.startswith("STATS|"):
            p = line.split("|")
            stats = {"total": p[1], "chunks": p[2], "dur": p[3], "notes": p[4]}
        elif line.startswith("DONE|"):
            done_path = line.split("|", 1)[1]
        elif line.startswith("ERROR|"):
            error = line.split("|", 1)[1]

        el = time.time() - t0
        # 关键：只有收到新事件、或距上次刷新超过 2 秒时才更新界面。
        # 之前每秒都 yield 一次 Markdown + 传 None 清空音频组件，长任务下会持续闪烁。
        got_event = line != ""
        if not got_event and (el - last_ui) < 2.0:
            continue
        last_ui = el

        if est_total > 0:
            progress(min(el / est_total, 0.97), desc=stage)
        else:
            progress(0.02, desc=stage)

        rows = ["⏳ **乐谱唱歌进行中**"]
        if total_sec:
            rows.append("乐谱 **%.1f 秒** / **%d** 段（%d 音符）· 当前：**%s**"
                        % (total_sec, n_chunks, total_notes, stage))
        else:
            rows.append("当前：**%s**" % stage)
        if chunk_msg:
            rows.append("分段进度：**%s**" % chunk_msg.replace("|", " "))
        if est_total > 0:
            rows.append("已用 **%.0f 秒** / 预计约 **%.0f 秒**（剩余约 %.0f 秒）"
                        % (el, est_total, max(0.0, est_total - el)))
        else:
            rows.append("已用 **%.0f 秒**" % el)
        # 音频输出保持不动：传 None 会反复清空播放器
        yield gr.update(), "\n\n".join(rows)

    proc.wait()
    el = time.time() - t0

    if error:
        yield None, ("❌ **乐谱唱歌失败**\n\n`%s`\n\n"
                     "常见原因：显存被其他程序占用。请关闭浏览器/游戏等后重试。" % error[:400])
        return
    if not done_path or not os.path.isfile(done_path):
        yield None, "❌ **未产出音频**（子进程退出码 %s）" % proc.returncode
        return

    dur = stats.get("dur", "?")
    progress(1.0, desc="完成")
    yield done_path, (
        "✅ **乐谱唱歌完成**\n\n"
        "| 指标 | 数值 |\n|---|---|\n"
        "| 输出时长 | %s 秒 |\n"
        "| 分段数 | %s 段 |\n"
        "| 音符总数 | %s |\n"
        "| 合成耗时 | %.1f 秒 |\n"
        "| 文件 | `%s` |"
        % (dur, stats.get("chunks", "?"), stats.get("notes", "?"), el, done_path)
    )


# ---------- 参考音频分离预览（只分离、不转换）----------
_sep_cache = {"ref": None, "vocals": None, "accomp": None}


def _clean_work(prefix, keep=None):
    """清理临时工作区里指定前缀的旧目录（keep 指定的那个保留）。

    分离产物默认只放临时区，同一会话内反复操作会累积，这里做滚动清理。
    """
    try:
        for name in os.listdir(WORK_DIR):
            p = os.path.join(WORK_DIR, name)
            if not name.startswith(prefix) or not os.path.isdir(p):
                continue
            if keep and os.path.abspath(p) == os.path.abspath(keep):
                continue
            shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass


def _trim_for_sep(path, work_dir, limit_sec=30.0):
    """分离前把过长音频裁到 limit_sec 秒，避免在长参考上白等。"""
    try:
        info = sf.info(path)
        if info.duration > limit_sec:
            data, sr = sf.read(path, always_2d=True,
                               frames=int(limit_sec * info.samplerate))
            os.makedirs(work_dir, exist_ok=True)
            out = os.path.join(work_dir, "trimmed_%ds.wav" % int(limit_sec))
            sf.write(out, data, sr)
            return out
    except Exception:
        pass
    return path


def sep_preview(checked, target):
    """勾选「参考音频也分离人声」后立即分离出参考音频的人声与伴奏，供试听。

    **只做分离，不做音色转换** —— 先让用户听分离效果，再决定是否点「开始转换」。
    结果会缓存：之后真正转换时若参考音频未变，直接复用，不重复分离。
    """
    if not checked:
        # 未勾选：什么都不做（不发提示，避免上传参考音频时覆盖状态文字）
        yield gr.update(), gr.update(), gr.update()
        return
    if not target:
        yield gr.update(), gr.update(), "⚠️ **请先上传参考音频**，然后再勾选此项。"
        return

    _clean_work("preview_")          # 清掉上一次预览的临时文件
    work = os.path.join(WORK_DIR, "preview_" + time.strftime("%Y%m%d_%H%M%S"))
    yield gr.update(), gr.update(), (
        "🔄 **正在分离参考音频的人声…**\n\n"
        "只做分离、不做转换（约 5~10 秒）。完成后可在右侧试听人声与伴奏。\n\n"
        "ℹ️ 分离结果是**临时文件**（应用启动时自动清空），"
        "需要留存请点「💾 保存分离结果」。"
    )
    try:
        src = _trim_for_sep(target, work)
        vocals, accomp = separate_vocals(src, work)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        yield gr.update(), gr.update(), (
            "❌ **参考音频分离失败**\n\n`%s: %s`" % (type(exc).__name__, str(exc)[:300])
        )
        return

    _sep_cache.update({"ref": os.path.abspath(target),
                       "vocals": vocals, "accomp": accomp})
    _last_stems.update({"vocals": vocals, "accomp": accomp, "source": "参考音频"})
    yield (gr.update(value=vocals, label="🎤 分离出的人声 · 参考音频（干声）", visible=True),
           gr.update(value=accomp, label="🎵 分离出的伴奏 · 参考音频", visible=True),
           "✅ **参考音频分离完成** —— 请试听右侧两轨。\n\n"
           "觉得可用再点「🚀 开始转换」（会复用这次分离结果，不重复分离）；"
           "觉得不如原声，取消勾选即可。\n\n"
           "ℹ️ 当前为**临时文件**，满意可点「💾 保存分离结果」留存。")


def save_stems():
    """把当前展示的分离结果另存到 out\\saved\\<时间戳>\\（是否留存由用户决定）。"""
    v = _last_stems.get("vocals")
    a = _last_stems.get("accomp")
    if not v or not os.path.isfile(v):
        return "⚠️ **当前没有可保存的分离结果**\n\n请先勾选「参考音频分离人声」，或跑一次完整歌曲转换。"
    ts = time.strftime("%Y%m%d_%H%M%S")
    dst = os.path.join(SAVE_DIR, ts)
    os.makedirs(dst, exist_ok=True)
    saved = []
    for src, name in ((v, "人声.wav"), (a, "伴奏.wav")):
        if src and os.path.isfile(src):
            shutil.copy(src, os.path.join(dst, name))
            saved.append("%s（%.1f MB）" % (name, os.path.getsize(os.path.join(dst, name)) / 1024**2))
    if not saved:
        return "❌ 保存失败：找不到分离文件（可能已被清理）。"
    return (
        "💾 **已保存分离结果**（来源：%s）\n\n"
        "| 文件 | 大小 |\n|---|---|\n%s\n\n"
        "目录：`%s`"
        % (_last_stems.get("source", "—"),
           "\n".join("| %s |" % s for s in saved), dst)
    )


def sep_preview_manual(target):
    """按钮触发的分离试听（与勾选后自动执行等价）。"""
    yield from sep_preview(True, target)


# ---------- 转换处理 ----------
def convert(source, target, is_song, sep_ref, steps, pitch, length_adjust, cfg, auto_f0,
            progress=gr.Progress()):
    """念山AI 转换主流程。yield (混音成品或None, 转换后人声或None, 状态Markdown)。

    输入处理：
      - is_song：源音频是完整歌曲 → 先 Demucs 分离人声/伴奏 → 转换 → 混音回伴奏。
      - sep_ref：参考音频带伴奏 → 先分离出人声再提取音色（否则乐器会污染
        CAM++ 说话人嵌入与 F0，导致音色不像、音高飘）。
    """
    import threading

    # 上一次跑的是「乐谱唱歌」（已卸掉 Seed-VC 腾显存），这里按需重新加载
    if not _seedvc_loaded:
        yield None, None, None, None, "🔄 **正在重新加载音色转换模型（约 20 秒）…**"
        load_seedvc()

    if not source or not target:
        yield None, None, None, None, "⚠️ **请先上传「源音频」和「参考音频」再开始转换**"
        return

    try:
        dur_in = float(sf.info(source).duration)
    except Exception:
        dur_in = 0.0

    t_start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 步骤总数：转换 1 步 + 源分离（歌曲模式）+ 参考分离（可选）+ 混音（歌曲模式）
    n_steps = 1 + (2 if is_song else 0) + (1 if sep_ref else 0)
    _clean_work("src_")              # 清掉上一次转换的临时分离文件
    _clean_work("ref_")
    step = 0
    vocal_source = source
    accomp_path = None
    sep_dt = 0.0
    # 展示给用户在线试听的分离产物（歌曲模式优先展示源音频的分离结果）
    show_vocal = None
    show_accomp = None

    # ===== 分离源音频人声（完整歌曲模式）=====
    if is_song:
        step += 1
        work_dir = os.path.join(WORK_DIR, "src_" + time.strftime("%Y%m%d_%H%M%S"))
        yield None, None, None, None, (
            "🔄 **步骤 %d/%d：分离源音频的人声与伴奏…**\n\n"
            "Demucs 分离中（GPU，约 0.3×音频时长），%.0f 秒音频预计 %.0f 秒。"
            % (step, n_steps, dur_in, dur_in * 0.3)
        )
        t_sep = time.time()
        try:
            vocal_source, accomp_path = separate_vocals(source, work_dir)
        except Exception as exc:
            yield None, None, None, None, (
                "❌ **源音频人声分离失败**\n\n`%s: %s`\n\n"
                "可尝试取消勾选「完整歌曲」选项，直接上传纯人声。"
                % (type(exc).__name__, str(exc)[:300])
            )
            return
        sep_dt = time.time() - t_sep
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 立刻把分离产物交给界面，方便边听边等后面的转换
        show_vocal, show_accomp = vocal_source, accomp_path
        _last_stems.update({"vocals": show_vocal, "accomp": show_accomp, "source": "源音频"})
        yield (gr.update(), gr.update(),
               gr.update(value=show_vocal, label="🎤 分离出的人声 · 源音频（干声）"),
               gr.update(value=show_accomp, label="🎵 分离出的伴奏 · 源音频（可直接当伴奏用）"),
               "✅ **源音频分离完成**（可在右侧试听人声与伴奏）\n\n"
               "⏳ 步骤 %d/%d：音色转换即将开始…" % (step + 1, n_steps))

    # ===== 分离参考音频人声（可选）=====
    if sep_ref:
        step += 1
        ref_work = os.path.join(WORK_DIR, "ref_" + time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(ref_work, exist_ok=True)

        # 若用户已用勾选/按钮预览过同一份参考音频，直接复用，不重复分离
        if (_sep_cache.get("ref") == os.path.abspath(target)
                and _sep_cache.get("vocals") and os.path.isfile(_sep_cache["vocals"])):
            ref_vocals = _sep_cache["vocals"]
            ref_accomp = _sep_cache.get("accomp")
            target = ref_vocals
            print("复用分离预览结果: %s" % ref_vocals)
            if show_vocal is None:
                show_vocal, show_accomp = ref_vocals, ref_accomp
                _last_stems.update({"vocals": show_vocal, "accomp": show_accomp,
                                    "source": "参考音频"})
                yield (gr.update(), gr.update(),
                       gr.update(value=show_vocal, label="🎤 分离出的人声 · 参考音频（干声）",
                                 visible=True),
                       gr.update(value=show_accomp, label="🎵 分离出的伴奏 · 参考音频",
                                 visible=True),
                       "♻️ **复用已预览的参考分离结果**（不重复分离）\n\n"
                       "⏳ 步骤 %d/%d：音色转换即将开始…" % (step + 1, n_steps))
        else:
            # 参考音频只需 25 秒左右，先裁剪以免在长参考上浪费分离时间
            ref_orig = os.path.abspath(target)      # 记录原始参考路径（下面 target 会被改写）
            ref_for_sep = _trim_for_sep(target, ref_work)

            yield None, None, gr.update(), gr.update(), (
                "🔄 **步骤 %d/%d：分离参考音频的人声…**\n\n"
                "带伴奏的参考会干扰音色提取与音高分析，先分离出干声再提取音色。"
                % (step, n_steps)
            )
            t_ref = time.time()
            try:
                ref_vocals, ref_accomp = separate_vocals(ref_for_sep, ref_work)
                target = ref_vocals
            except Exception as exc:
                yield None, None, gr.update(), gr.update(), (
                    "⚠️ **参考音频分离失败，将改用原始参考音频继续**\n\n`%s: %s`"
                    % (type(exc).__name__, str(exc)[:200])
                )
            else:
                print("参考音频分离完成: %.1f 秒 -> %s" % (time.time() - t_ref, ref_vocals))
                # 写入缓存，便于同一参考再次转换时复用
                _sep_cache.update({"ref": ref_orig,
                                   "vocals": ref_vocals, "accomp": ref_accomp})
                # 歌曲模式已展示源音频的分离结果；否则展示参考音频的
                if show_vocal is None:
                    show_vocal, show_accomp = ref_vocals, ref_accomp
                    _last_stems.update({"vocals": show_vocal, "accomp": show_accomp,
                                        "source": "参考音频"})
                    yield (gr.update(), gr.update(),
                           gr.update(value=show_vocal, label="🎤 分离出的人声 · 参考音频（干声）",
                                     visible=True),
                           gr.update(value=show_accomp, label="🎵 分离出的伴奏 · 参考音频",
                                     visible=True),
                           "✅ **参考音频分离完成**（可在右侧试听）\n\n"
                           "⏳ 步骤 %d/%d：音色转换即将开始…" % (step + 1, n_steps))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ===== 音色转换（带实时进度） =====
    step += 1
    conv_est = max(6.0, 0.05 * dur_in * steps + 3.0)
    total_chunks = max(1, int(np.ceil(max(dur_in - 30, 0) / 25.0)) + 1)
    if dur_in > 60:
        tip = "\n\n💡 源音频较长（%.0f 秒），属正常现象，请耐心等待。" % dur_in
    else:
        tip = ""

    holder = {}

    def _worker():
        try:
            for _mp3, full in app_svc.voice_conversion(
                vocal_source, target, steps, length_adjust, cfg, auto_f0, pitch
            ):
                holder["chunks"] = holder.get("chunks", 0) + 1
                if full is not None:
                    holder["full"] = full
        except Exception as exc:
            holder["error"] = exc

    t0 = time.time()
    th = threading.Thread(target=_worker, daemon=True)
    th.start()

    # 进度刷新间隔：过密会让长任务的界面持续重绘（表现为闪烁），2 秒足够顺滑
    HEARTBEAT = 2.0
    while th.is_alive():
        el = time.time() - t0
        frac = 0.15 + min(el / conv_est, 0.95) * 0.8 if is_song else min(el / conv_est, 0.97)
        progress(frac, desc="音色转换中…")
        seg = " · 分段 %d/%d" % (max(1, holder.get("chunks", 0)), total_chunks) if total_chunks > 1 else ""
        # 音频输出用 gr.update() 保持不动（传 None 会反复清空播放器 = 闪烁）
        yield gr.update(), gr.update(), gr.update(), gr.update(), (
            "⏳ **步骤 %d/%d：音色转换中…**%s\n\n"
            "已用时 **%.0f 秒** / 预计约 **%.0f 秒**（%.0f 秒音频 × %d 步）%s"
            % (step, n_steps, seg, el, conv_est, dur_in, steps, tip)
        )
        time.sleep(HEARTBEAT)
    th.join()

    if "error" in holder:
        exc = holder["error"]
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            yield None, None, None, None, (
                "❌ **显存不足（OOM）**\n\n"
                "请关闭其他占用显存的程序（浏览器、游戏、剪辑软件等）后重试；"
                "也可适当减少扩散步数（不影响显存，只提速）或缩短音频。"
            )
        else:
            yield None, None, None, None, "❌ **转换失败**\n\n`%s: %s`" % (type(exc).__name__, str(exc)[:300])
        return

    result = holder.get("full")
    if result is None:
        yield None, None, None, None, "❌ **未产出音频**，请检查输入文件是否为可解析的音频。"
        return

    sr_out, wave = result
    wave = np.asarray(wave, dtype=np.float32)
    ts = time.strftime("%Y%m%d_%H%M%S")
    vocal_path = os.path.join(OUT_DIR, "nianshan_vocal_%s.wav" % ts)
    sf.write(vocal_path, wave, sr_out)
    conv_dt = time.time() - t0

    # 分离产物的显式可见性：有则显示（可在线试听/下载），无则隐藏
    sep_vocal_out = gr.update(value=show_vocal, visible=bool(show_vocal))
    sep_accomp_out = gr.update(value=show_accomp, visible=bool(show_accomp))

    # ===== 阶段 3（仅歌曲模式）：混音 =====
    mix_path = None
    if is_song and accomp_path:
        yield gr.update(), vocal_path, sep_vocal_out, sep_accomp_out, (
            "🔄 **步骤 %d/%d：混音中…**\n\n"
            "人声转换已完成（可先试听纯人声），正在与原伴奏合并。"
            % (step + 1, n_steps)
        )
        try:
            mix_path = mix_tracks(wave, sr_out, accomp_path,
                                  os.path.join(OUT_DIR, "nianshan_song_%s.wav" % ts))
        except Exception as exc:
            yield None, vocal_path, sep_vocal_out, sep_accomp_out, (
                "⚠️ **混音失败**（纯人声已产出）\n\n`%s: %s`"
                % (type(exc).__name__, str(exc)[:300])
            )
            return

    dur = len(wave) / sr_out
    dt_total = time.time() - t_start
    peak = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0

    rows = [
        "| 总耗时 | %.1f 秒 |" % dt_total,
    ]
    if is_song:
        rows.append("| 其中 分离 / 转换 | %.1f / %.1f 秒 |" % (sep_dt, conv_dt))
    rows += [
        "| 转换速度 | RTF %.2f |" % (conv_dt / max(dur, 1e-6)),
        "| 输出时长 | %.1f 秒 · %d Hz |" % (dur, sr_out),
        "| 显存峰值 | %.0f MiB（PyTorch 实际分配）|" % peak,
    ]
    files = "| 人声文件 | `%s` |" % vocal_path
    if mix_path:
        files += "\n| 成品文件 | `%s` |" % mix_path
    rows.append(files)

    progress(1.0, desc="完成")
    yield mix_path, vocal_path, sep_vocal_out, sep_accomp_out, (
        "✅ **转换完成**\n\n| 指标 | 数值 |\n|---|---|\n" + "\n".join(rows)
    )


# ---------- 界面 ----------
CSS = """
.ns-header { text-align: center; padding: 1.4rem 0 0.2rem; }
.ns-header h1 {
    font-size: 2.6rem; font-weight: 800; margin: 0;
    background: linear-gradient(90deg, #0d9488, #059669 55%, #34d399);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.ns-header p { color: #64748b; margin: 0.35rem 0 0; font-size: 1.02rem; }
.ns-badge {
    display: inline-block; margin-top: .45rem; padding: .15rem .7rem;
    border: 1px solid #99f6e4; border-radius: 999px;
    color: #0f766e; font-size: .8rem; background: #f0fdfa;
}
.ns-footer { text-align: center; color: #94a3b8; font-size: .8rem; padding: 1rem 0 .4rem; line-height: 1.7; }
"""

_theme = gr.themes.Soft(primary_hue="teal", secondary_hue="emerald", neutral_hue="slate")

# 界面示例音频（可选）：放在 <workspace>/demo_audio/ 下即会自动出现
_TEST_SRC = os.path.join(DEMO_AUDIO_DIR, "test_source_short.wav")
_TEST_REF = os.path.join(DEMO_AUDIO_DIR, "test_ref.wav")
_TEST_SONG = os.path.join(DEMO_AUDIO_DIR, "song_demo.mp3")
_examples = []
if os.path.isfile(_TEST_SRC) and os.path.isfile(_TEST_REF):
    _examples.append([_TEST_SRC, _TEST_REF, False, False, 30, 0, 1.0, 0.7, False])
if os.path.isfile(_TEST_SONG) and os.path.isfile(_TEST_REF):
    _examples.append([_TEST_SONG, _TEST_REF, True, False, 30, 0, 1.0, 0.7, False])
for _s, _r in [("examples/source/yae_0.wav", "examples/reference/dingzhen_0.wav"),
               ("examples/source/jay_0.wav", "examples/reference/azuma_0.wav")]:
    if os.path.isfile(_s) and os.path.isfile(_r):
        _examples.append([os.path.abspath(_s), os.path.abspath(_r), False, False,
                          30, 0, 1.0, 0.7, False])

# 乐谱唱歌示例：直接用上游仓库自带的示例素材
_SVS_MIDI_DEMO = os.path.join(SVS_ROOT, "resources", "audios",
                              "female__Rnb_Funk__下等马_clip_001.mid")
_SVS_TIMBRE_DEMO = os.path.join(SVS_ROOT, "resources", "audios", "male.wav")
_SVS_CONTENT_DEMO = "在爱的回归线，又期待相见。"
_SVS_LYRICS_DEMO = (
    "头抬起来，你表情别太奇怪，无大碍。没伤到脑袋，如果我下手太重，私密马赛。"
    "习武十载，没下山没谈恋爱，吃光后山七八亩菜，练就这套拳脚，莫以貌取人哉。"
    "暮色压台，擂鼓未衰，下一个谁还要来？速来领拜，别耽误我热蒸铁揭盖。"
)

with gr.Blocks(theme=_theme, css=CSS, title="念山AI · 歌声工作台") as demo:
    gr.HTML(
        "<div class='ns-header'><h1>念山AI</h1>"
        "<p>AI 歌声工作台 · 本地运行 · 双引擎（音色转换 + 乐谱唱歌）</p>"
        "<span class='ns-badge'>无需训练 · 一段参考音频即可复刻音色</span></div>"
    )

    with gr.Tabs():
        # ================= 模式一：音色转换 =================
        with gr.Tab("🎤 音色转换（把一首歌变成你的音色）"):
            with gr.Row():
                with gr.Column(scale=5):
                    src = gr.Audio(type="filepath", sources=["upload", "microphone"],
                                   label="🎤 源音频（人声干声；完整歌曲请勾选下方选项）")
                    is_song = gr.Checkbox(
                        value=False,
                        label="🎵 源音频是完整歌曲（自动分离人声 → 转换 → 混音回伴奏，耗时约增加 30%）")
                    ref = gr.Audio(type="filepath", sources=["upload", "microphone"],
                                   label="🎯 参考音频（目标音色 · 1~30 秒；带伴奏请用下方分离试听）")
                    with gr.Row():
                        sep_ref = gr.Checkbox(
                            value=False, scale=4,
                            label="🎯 参考音频分离人声（勾选后立即分离并试听，不影响是否转换）")
                        sep_preview_btn = gr.Button("🔍 重新分离试听", size="sm", scale=1)
                    with gr.Row():
                        steps = gr.Slider(1, 200, value=30, step=1, label="扩散步数",
                                          info="歌声建议 30~50；越大质量越好、越慢（显存不变）")
                        pitch = gr.Slider(-24, 24, value=0, step=1, label="音调变换（半音）",
                                          info="用于匹配伴奏调性，正数升调、负数降调")
                    with gr.Accordion("⚙️ 高级参数", open=False):
                        length_adjust = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="长度调整",
                                                  info="<1 加速、>1 减速")
                        cfg = gr.Slider(0.0, 1.0, value=0.7, step=0.1, label="CFG 强度",
                                        info="对结果影响细微，一般保持默认")
                        auto_f0 = gr.Checkbox(value=False,
                                              label="自动 F0 调整（歌声转换请保持关闭！开启会导致歌声与伴奏调性不一致）")
                    with gr.Row():
                        btn = gr.Button("🚀 开始转换", variant="primary", scale=3)
                        clear = gr.ClearButton(components=[src, ref, is_song, sep_ref],
                                               value="🧹 清空", scale=1)

                    if _examples:
                        gr.Examples(examples=_examples,
                                    inputs=[src, ref, is_song, sep_ref, steps, pitch,
                                            length_adjust, cfg, auto_f0],
                                    label="📁 示例（点击一键填充）",
                                    cache_examples=False)

                with gr.Column(scale=6):
                    out_mix = gr.Audio(type="filepath",
                                       label="🎵 成品混音（歌曲模式产出：转换后人声 + 原伴奏）",
                                       interactive=False)
                    out = gr.Audio(type="filepath",
                                   label="🎧 转换后人声（纯人声，可播放 / 可下载）",
                                   interactive=False)
                    # 分离产物试听：只有勾选并跑过分离才会出现（动态显示）
                    out_sep_vocal = gr.Audio(type="filepath",
                                             label="🎤 分离出的人声（干声）",
                                             interactive=False, visible=False)
                    out_sep_accomp = gr.Audio(type="filepath",
                                              label="🎵 分离出的伴奏（可直接当伴奏用）",
                                              interactive=False, visible=False)
                    save_stems_btn = gr.Button("💾 保存分离结果（否则应用重启后自动清空）",
                                               size="sm", visible=True)
                    status = gr.Markdown("**状态**：就绪。上传两段音频后点击「开始转换」。")

            # 输出组件定义在右栏，创建晚于「清空」按钮，这里补充进清空列表
            clear.add([out, out_mix, out_sep_vocal, out_sep_accomp, status])

            btn.click(convert,
                      inputs=[src, ref, is_song, sep_ref, steps, pitch, length_adjust, cfg, auto_f0],
                      outputs=[out_mix, out, out_sep_vocal, out_sep_accomp, status],
                      api_name="convert")

            # 勾选「参考音频分离人声」→ 立即只做分离并试听（不转换）
            # 放在两栏都建好之后接线，否则引用不到右侧的输出组件
            sep_ref.change(sep_preview,
                           inputs=[sep_ref, ref],
                           outputs=[out_sep_vocal, out_sep_accomp, status])
            # 先勾选、后上传参考音频的情况：参考变了且已勾选时自动分离
            ref.change(sep_preview,
                       inputs=[sep_ref, ref],
                       outputs=[out_sep_vocal, out_sep_accomp, status])
            sep_preview_btn.click(sep_preview_manual,
                                  inputs=[ref],
                                  outputs=[out_sep_vocal, out_sep_accomp, status],
                                  api_name="sep_preview_run")
            # 是否留存分离结果由用户决定（默认只放临时目录）
            save_stems_btn.click(save_stems, inputs=None, outputs=[status],
                                 api_name="save_stems")

        # ================= 模式二：乐谱唱歌 =================
        with gr.Tab("🎼 乐谱唱歌（用你的音色唱一首新歌）"):
            with gr.Row():
                with gr.Column(scale=5):
                    svs_midi = gr.File(label="🎼 乐谱 MIDI 文件（.mid / .midi）",
                                       file_types=[".mid", ".midi"], type="filepath")
                    svs_lyrics = gr.Textbox(label="📝 要唱的歌词", lines=4,
                                            placeholder="把整首歌的歌词粘贴到这里（中文按字、英文按词自动分配到各段）")
                    svs_timbre = gr.Audio(type="filepath", sources=["upload", "microphone"],
                                          label="🎯 音色音频（5~7 秒清唱，无伴奏）")
                    svs_content = gr.Textbox(label="🗣 音色片段里唱的是什么词",
                                             placeholder="例如：在爱的回归线，又期待相见。",
                                             info="模型需要它来对齐文本，必须与实际演唱内容一致")
                    with gr.Row():
                        svs_chunk = gr.Slider(15, 40, value=30, step=1, label="分段长度（秒）",
                                              info="越大接缝越少但越吃显存；本机实测 40 秒可行、55 秒会爆")
                        svs_pitch = gr.Slider(-12, 12, value=-1, step=1, label="升降调（半音）",
                                              info="实测：-1 时音准最好（轮廓相关 0.89）；大幅移调会明显劣化音质，建议只在 -3~+2 内微调")
                    with gr.Accordion("⚙️ 高级参数", open=False):
                        svs_cfg = gr.Slider(1.0, 10.0, value=4.0, step=0.5, label="CFG 强度",
                                            info="越大越贴歌词，过大会失真")
                        svs_nfe = gr.Slider(10, 100, value=32, step=1, label="扩散步数",
                                            info="越大质量越好、越慢")
                    with gr.Row():
                        svs_btn = gr.Button("🎼 开始生成歌声", variant="primary", scale=3)
                        svs_clear = gr.ClearButton(
                            components=[svs_midi, svs_lyrics, svs_timbre, svs_content],
                            value="🧹 清空", scale=1)

                    if os.path.isfile(_SVS_MIDI_DEMO) and os.path.isfile(_SVS_TIMBRE_DEMO):
                        gr.Examples(
                            examples=[[_SVS_MIDI_DEMO, _SVS_LYRICS_DEMO, _SVS_TIMBRE_DEMO,
                                       _SVS_CONTENT_DEMO, 30, -1, 4.0, 32]],
                            inputs=[svs_midi, svs_lyrics, svs_timbre, svs_content,
                                    svs_chunk, svs_pitch, svs_cfg, svs_nfe],
                            label="📁 示例（官方示例乐谱 + 音色，点击一键填充）",
                            cache_examples=False)

                with gr.Column(scale=6):
                    svs_out = gr.Audio(type="filepath",
                                       label="🎧 生成的歌声（可播放 / 可下载）",
                                       interactive=False)
                    svs_status = gr.Markdown(
                        "**状态**：就绪。\n\n"
                        "上传 MIDI 乐谱 → 填歌词 → 给一段 5~7 秒清唱作为音色 → 点击「开始生成歌声」。\n\n"
                        "⚠️ 4GB 显存下与「音色转换」互斥：开始生成时会自动卸载转换模型腾出显存。"
                    )

            svs_btn.click(svs_generate,
                          inputs=[svs_midi, svs_lyrics, svs_timbre, svs_content,
                                  svs_chunk, svs_pitch, svs_cfg, svs_nfe],
                          outputs=[svs_out, svs_status],
                          api_name="svs")

    gr.HTML(
        "<div class='ns-footer'>念山AI · 本地推理，音频不上传任何服务器<br>"
        "音色转换基于 Seed-VC（零样本音色转换）；乐谱唱歌基于 YingMusic-Singer（零样本歌声合成）<br>"
        "合规提示：声音权益受《民法典》第 1023 条保护，请勿克隆未获授权的他人声音，"
        "建议使用本人声音、已授权素材或合成音色。</div>"
    )


if __name__ == "__main__":
    demo.launch(
        share=False,
        inbrowser=True,  # 启动后自动打开浏览器
        # WORK_DIR 是分离产物的临时目录，Gradio 需要它才能把音频喂给播放器
        allowed_paths=[OUT_DIR, WORK_DIR, DEMO_AUDIO_DIR,
                       os.path.join(_HERE, "examples")],
    )