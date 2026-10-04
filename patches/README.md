# 上游补丁说明

本目录存放**对上游项目的最小必要修改**，以补丁形式提供，**不内联上游代码**。

补丁基于以下**已验证的提交号**生成（`install.ps1` 会检出这些提交，确保补丁能干净应用）：

| 上游 | 提交号 | 日期 |
|---|---|---|
| Plachtaa/seed-vc | `51383efd921027683c89e5348211d93ff12ac2a8` | 2025-04-20 |
| GiantAILab/YingMusic-Singer | `6d7c453781cdaf286227ba7a51e79d57b19fad71` | 2026-04-28 |

---

## seed-vc-local.patch

只改 1 个文件 `app_svc.py`（9 行增 / 3 行删）：

| 修改 | 原因 |
|---|---|
| 扩散步数默认值 **10 → 30** | 官方建议歌声转换用 30~50 步。实测步数只影响音色与耗时，不影响显存 |
| 自动 F0 调整默认值 **True → False** | 歌声转换必须关闭。开启会把源音高整体移调，导致歌声与伴奏调性不一致 |

> 说明：本项目自己的界面（`app/app_nianshan.py`）直接通过参数传值，不依赖这两个默认值；
> 打补丁是为了**官方原始界面**（`app_launch.py`）也表现正确。

---

## yingmusic-local.patch

改 2 个文件：

### `src/singer/decoder/modules.py`（+26 行）★ 最关键

DiT 的自注意力**直接调用** `flash_attention()`，并在其中
`assert FLASH_ATTN_2_AVAILABLE` —— 而 flash-attn 在 Windows 上没有可用的预编译轮子，
官方 README 给出的还是 Linux 专用 wheel。

补丁在 `flash_attention()` 开头加入 PyTorch 原生 `scaled_dot_product_attention`
的等价实现分支：

- 保持 `[B, L, N, C]` 布局，不展平 varlen 序列
- 用布尔掩码屏蔽 `k_lens` 之外的补齐位置
- 未使用滑动窗口（`singer.yaml` 未配置 `window_size`，恒为默认 `(-1, -1)`），故可完全等价替换

### `src/singer/model.py`（+10 行 / -1 行）

`torchaudio` 2.9 的音频 I/O 已改为经 `torchcodec` 实现，需要 FFmpeg **共享库**
（`avcodec-*.dll` 等）；而本项目部署的是**静态** ffmpeg.exe，导致 `torchaudio.load()` 抛
`Could not load libtorchcodec`。

补丁把 `load_audio()` 改为优先用 `soundfile` 读取（其自带 libsndfile，支持 wav/mp3/flac/ogg），
失败再回退 `torchaudio`。

---

## 无法用补丁表达的两处修复

这两处修改的是**第三方包**（不在上游 git 仓库内），因此由 `install.ps1` 直接处理：

### 1. LangSegment 0.2.0 自身有 bug

`import LangSegment` 会直接失败：其 `__init__.py` 导入了 `LangSegment.py` 中**不存在**的
`setLangfilters` / `getLangfilters`。PyPI 上仅有 0.2.0 这一个版本，无法换版本规避。

**处理**：删除 `__init__.py` 中这两个名字（仓库只用到 `setfilters` / `getTexts`）。

### 2. phonemizer 找不到 espeak-ng

报 `RuntimeError: espeak not installed on your system`。Windows 上没有系统级 espeak-ng。

**处理**：安装 PyPI 包 `espeakng-loader`（自带 `espeak-ng.dll` 与 `espeak-ng-data`），
并在 venv 的 site-packages 放置 `sitecustomize.py` 自动设置：

```
PHONEMIZER_ESPEAK_LIBRARY   = .../espeakng_loader/espeak-ng.dll
PHONEMIZER_ESPEAK_DATA_PATH = .../espeakng_loader/espeak-ng-data
```

`sitecustomize.py` 由 Python 启动时自动导入，故对整个 venv 生效。
模板见 `tools/espeak_sitecustomize.py`。

> 附注：espeak 的普通话语言代码是 `cmn` 而非 `zh`。上游已正确映射，无需改动。

---

## 应用补丁

```powershell
cd <上游目录>
git apply <本仓库>/patches/xxx.patch
```

若因上游更新导致补丁冲突，可参考上表手工修改对应位置。
