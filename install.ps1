<#
.SYNOPSIS
    念山AI 一键部署脚本（Windows + NVIDIA 显卡）

.DESCRIPTION
    自动完成以下步骤：
      1. 检查依赖（git / uv / NVIDIA 显卡）
      2. 克隆上游项目并检出「已验证的提交号」
      3. 创建两个独立 Python 环境并安装依赖
      4. 应用本项目补丁（Windows 适配）
      5. 把本项目文件部署到正确位置
      6. 下载模型权重与 ffmpeg
      7. 修复第三方包（LangSegment / espeak-ng）

    默认把上游项目放在本仓库的**同级目录**。

.PARAMETER Workspace
    上游项目与工具的存放目录。默认 = 本仓库的上一级目录。

.PARAMETER Proxy
    可选 HTTP 代理（用于访问 HuggingFace）。默认 http://127.0.0.1:7897。
    代理不可用时会自动跳过。

.PARAMETER SkipWeights
    跳过权重下载（首次启动应用时会自动下载）。

.EXAMPLE
    .\install.ps1
    .\install.ps1 -Workspace D:\ai -SkipWeights
#>
[CmdletBinding()]
param(
    [string]$Workspace = (Split-Path $PSScriptRoot -Parent),
    [string]$Proxy = "http://127.0.0.1:7897",
    [switch]$SkipWeights
)

$ErrorActionPreference = 'Stop'
$RepoRoot = $PSScriptRoot

# ---- 上游仓库与「已验证提交号」（补丁基于这些提交生成）----
$SEEDVC_REPO   = "https://github.com/Plachtaa/seed-vc.git"
$SEEDVC_COMMIT = "51383efd921027683c89e5348211d93ff12ac2a8"   # 2025-04-20
$SVS_REPO      = "https://github.com/GiantAILab/YingMusic-Singer.git"
$SVS_COMMIT    = "6d7c453781cdaf286227ba7a51e79d57b19fad71"   # 2026-04-28

# ---- ModelScope 权重（YingMusic 侧，国内直连快）----
$MS_BASE = "https://www.modelscope.cn/api/v1/models/giantailab/YingMusic-Singer/repo?Revision=master&FilePath="
$MS_FILES = @('singer.v1.pt', 'autoencoder_music_dsp1920.ckpt', 'stable_audio_1920_vae.json',
              'rmvpe.pt', 'some.pt')

# ---- ffmpeg 静态构建（BtbN）----
$FFMPEG_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"

$SEEDVC_DIR = Join-Path $Workspace "seed-vc"
$SVS_DIR    = Join-Path $Workspace "YingMusic-Singer"
$FFMPEG_DIR = Join-Path $Workspace "tools\ffmpeg"

function Say($msg, $color = "Cyan") { Write-Host $msg -ForegroundColor $color }
function Step($n, $total, $msg) { Write-Host "`n[$n/$total] $msg" -ForegroundColor Cyan }
function Die($msg) { Write-Host "`n[错误] $msg" -ForegroundColor Red; exit 1 }

function Test-Proxy {
    try {
        return (Test-NetConnection -ComputerName 127.0.0.1 -Port ([uri]$Proxy).Port `
                -InformationLevel Quiet -WarningAction SilentlyContinue)
    } catch { return $false }
}

function Use-ProxyEnv {
    param([switch]$Enable)
    if ($Enable) {
        $env:HTTPS_PROXY = $Proxy; $env:HTTP_PROXY = $Proxy
        Say "  已启用代理: $Proxy" DarkGray
    } else {
        Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue
        Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue
    }
}

function Clone-Pinned($repo, $commit, $dest, $name) {
    if (Test-Path (Join-Path $dest ".git")) {
        Say "  $name 已存在，跳过克隆" DarkGray
    } else {
        if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
        Say "  克隆 $name ..."
        & git clone $repo $dest
        if ($LASTEXITCODE -ne 0) { Die "克隆 $name 失败（检查网络）" }
    }
    Push-Location $dest
    & git fetch --depth 1 origin $commit 2>$null | Out-Null
    & git checkout -q $commit 2>$null
    if ($LASTEXITCODE -ne 0) {
        Say "  警告：无法检出指定提交 $commit，将使用默认分支（补丁可能冲突）" Yellow
    } else {
        Say "  已检出 $($commit.Substring(0,8))" DarkGray
    }
    Pop-Location
}

# ============================================================================
Say "`n============================================" 
Say "  念山AI · 一键部署"
Say "============================================"
Say "  仓库位置 : $RepoRoot"
Say "  部署位置 : $Workspace"

# ---------------------------------------------------------------- 1. 依赖检查
Step 1 7 "检查依赖"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Die "未找到 git，请先安装 Git for Windows" }

$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) {
    Say "  未找到 uv，将使用 python -m venv（较慢）" Yellow
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        Die "既没有 uv 也没有 python。请安装其一：`n  uv: https://docs.astral.sh/uv/`n  或 Python 3.10+"
    }
} else {
    Say "  uv: $uv" DarkGray
}

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpu = (& nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) -join '; '
    Say "  显卡: $gpu" Green
    $vram = [int]((& nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | Select-Object -First 1))
    if ($vram -lt 4000) { Say "  警告：显存低于 4 GB，可能无法运行（实测下限为 4 GB）" Yellow }
    elseif ($vram -lt 8000) { Say "  提示：4~8 GB 显存下需依赖驱动内存回退，速度较慢" DarkGray }
} else {
    Say "  警告：未检测到 NVIDIA 显卡。CPU 推理约慢 20~50 倍，一首歌可能需数小时。" Yellow
}

$proxyOk = Test-Proxy
Say "  代理: $(if ($proxyOk) { "可用" } else { "不可用（访问 HuggingFace 可能失败）" })" $(if ($proxyOk) { "Green" } else { "Yellow" })

if (-not (Test-Path $Workspace)) { New-Item -ItemType Directory -Path $Workspace -Force | Out-Null }

# ------------------------------------------------------------ 2. 克隆上游项目
Step 2 7 "拉取上游项目"
Use-ProxyEnv -Enable:$false     # GitHub 通常可直连
Clone-Pinned $SEEDVC_REPO $SEEDVC_COMMIT $SEEDVC_DIR "Seed-VC"
Clone-Pinned $SVS_REPO    $SVS_COMMIT    $SVS_DIR    "YingMusic-Singer"

# ------------------------------------------------------------------ 3. 补丁
Step 3 7 "应用补丁"
Push-Location $SEEDVC_DIR
& git apply --reject (Join-Path $RepoRoot "patches\seed-vc-local.patch") 2>$null
Say "  Seed-VC 补丁已应用" DarkGray
Pop-Location
Push-Location $SVS_DIR
& git apply --reject (Join-Path $RepoRoot "patches\yingmusic-local.patch") 2>$null
Say "  YingMusic 补丁已应用" DarkGray
Pop-Location

# ------------------------------------------------------------ 4. 部署本项目文件
Step 4 7 "部署本项目文件"
Copy-Item (Join-Path $RepoRoot "app\app_nianshan.py")   $SEEDVC_DIR -Force
Copy-Item (Join-Path $RepoRoot "app\run_webapp.ps1")    $SEEDVC_DIR -Force
Copy-Item (Join-Path $RepoRoot "app\svc_launch.py")     $SEEDVC_DIR -Force
Copy-Item (Join-Path $RepoRoot "app\app_launch.py")     $SEEDVC_DIR -Force
Copy-Item (Join-Path $RepoRoot "app\requirements-infer.txt") $SEEDVC_DIR -Force
foreach ($f in 'long_song.py', 'svs_runner.py', 'verify_melody.py') {
    Copy-Item (Join-Path $RepoRoot "svs\$f") $SVS_DIR -Force
}
Copy-Item (Join-Path $RepoRoot "启动念山AI.bat") $Workspace -Force
Copy-Item (Join-Path $RepoRoot "停止念山AI.bat") $Workspace -Force
Say "  应用文件已就位（bat 启动器已放到 $Workspace）" DarkGray

# ------------------------------------------------------------ 5. 两个 Python 环境
Step 5 7 "创建 Python 环境并安装依赖（耗时较长，请耐心）"

function New-Venv($dir, $pyver) {
    Push-Location $dir
    if ($uv) {
        if ($pyver -eq "3.12") {
            # --python-preference only-managed：避免误用系统/其他软件自带的 Python
            & $uv venv --python $pyver --python-preference only-managed .venv
        } else {
            & $uv venv --python $pyver .venv
        }
    } else {
        & python -m venv .venv
    }
    Pop-Location
}

function Uv-Pip($dir, [string[]]$args_) {
    Push-Location $dir
    $py = Join-Path $dir ".venv\Scripts\python.exe"
    if ($uv) { & $uv pip install --python $py @args_ } else { & $py -m pip install @args_ }
    Pop-Location
}

# --- Seed-VC：Python 3.10 + torch 2.4.0/cu124 ---
Say "`n  >> Seed-VC 环境（Python 3.10 + torch 2.4.0+cu124）"
New-Venv $SEEDVC_DIR "3.10"
Uv-Pip $SEEDVC_DIR @('torch==2.4.0', 'torchaudio==2.4.0',
                     '--index-url', 'https://download.pytorch.org/whl/cu124',
                     '--index-strategy', 'unsafe-best-match')
Uv-Pip $SEEDVC_DIR @('-r', (Join-Path $SEEDVC_DIR 'requirements-infer.txt'))

# --- YingMusic：Python 3.12 + torch 2.9.1/cu126 ---
Say "`n  >> YingMusic 环境（Python 3.12 + torch 2.9.1+cu126）"
New-Venv $SVS_DIR "3.12"
Uv-Pip $SVS_DIR @('torch==2.9.1', 'torchaudio==2.9.1',
                  '--index-url', 'https://download.pytorch.org/whl/cu126',
                  '--index-strategy', 'unsafe-best-match')
Uv-Pip $SVS_DIR @('-r', (Join-Path $SVS_DIR 'requirements.txt'))
Uv-Pip $SVS_DIR @('espeakng-loader')

# ------------------------------------------------------------ 6. 修复第三方包
Step 6 7 "修复第三方包（LangSegment / espeak-ng）"
$svsPy = Join-Path $SVS_DIR ".venv\Scripts\python.exe"
$sitePackages = & $svsPy -c "import site; print(site.getsitepackages()[0])"

# 6.1 LangSegment 0.2.0 的 __init__ 导入了不存在的名字，导致 import 直接失败
$lsInit = Join-Path $sitePackages "LangSegment\__init__.py"
if (Test-Path $lsInit) {
    $content = Get-Content $lsInit -Raw -Encoding UTF8
    $fixed = $content -replace 'setLangfilters,getLangfilters,', ''
    if ($fixed -ne $content) {
        [System.IO.File]::WriteAllText($lsInit, $fixed, [System.Text.UTF8Encoding]::new($false))
        Say "  LangSegment 已修复" DarkGray
    } else { Say "  LangSegment 无需修复" DarkGray }
} else { Say "  警告：未找到 LangSegment，跳过" Yellow }

# 6.2 让 phonemizer 找到 espeakng-loader 自带的 espeak-ng
$scSrc = Join-Path $RepoRoot "tools\espeak_sitecustomize.py"
$scDst = Join-Path $sitePackages "sitecustomize.py"
if (Test-Path $scSrc) {
    Copy-Item $scSrc $scDst -Force
    Say "  sitecustomize.py 已放置（自动配置 espeak-ng 路径）" DarkGray
}

# 验证修复
& $svsPy -c "import LangSegment; from phonemizer.backend import EspeakBackend; EspeakBackend('cmn'); print('  OK: LangSegment + espeak-ng 均可用')"

# ------------------------------------------------------------ 7. 权重与 ffmpeg
Step 7 7 "下载模型权重与 ffmpeg"

# 7.1 ffmpeg
$ffmpegExe = Join-Path $FFMPEG_DIR "bin\ffmpeg.exe"
if (Test-Path $ffmpegExe) {
    Say "  ffmpeg 已存在" DarkGray
} else {
    Say "  下载 ffmpeg（约 100 MB）..."
    $zip = Join-Path $env:TEMP "ffmpeg-win64.zip"
    $tmp = Join-Path $env:TEMP "ffmpeg_extract"
    try {
        Invoke-WebRequest -Uri $FFMPEG_URL -OutFile $zip -UseBasicParsing
        if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
        Expand-Archive -Path $zip -DestinationPath $tmp -Force
        $inner = Get-ChildItem $tmp -Directory | Select-Object -First 1
        New-Item -ItemType Directory -Path $FFMPEG_DIR -Force | Out-Null
        Copy-Item (Join-Path $inner.FullName "bin") $FFMPEG_DIR -Recurse -Force
        Say "  ffmpeg 已安装到 $FFMPEG_DIR" DarkGray
    } catch {
        Say "  ffmpeg 下载失败：$_" Yellow
        Say "  可稍后手动下载并解压到 $FFMPEG_DIR（需包含 bin\ffmpeg.exe）" Yellow
    }
}

if (-not $SkipWeights) {
    # 7.2 YingMusic 权重（ModelScope 直连）
    $ckpts = Join-Path $SVS_DIR "ckpts"
    New-Item -ItemType Directory -Path $ckpts -Force | Out-Null
    foreach ($f in $MS_FILES) {
        $dest = Join-Path $ckpts $f
        if ((Test-Path $dest) -and ((Get-Item $dest).Length -gt 1KB)) {
            Say "  [跳过] $f" DarkGray; continue
        }
        Say "  下载 $f ..."
        & curl.exe -sS -L --ssl-no-revoke --max-time 1800 -o $dest "$MS_BASE$f"
        if (-not (Test-Path $dest) -or (Get-Item $dest).Length -lt 1KB) {
            Say "  $f 下载失败，请稍后重试" Yellow
        }
    }

    # 7.3 Seed-VC 权重（HuggingFace，可从 ModelScope 之外的渠道）
    Use-ProxyEnv -Enable:$proxyOk
    Say "`n  预取 Seed-VC 权重（HuggingFace，约 2.4 GB）..."
    Say "  若失败，首次启动应用时会自动重试下载。" DarkGray
    Push-Location $SEEDVC_DIR
    $py = Join-Path $SEEDVC_DIR ".venv\Scripts\python.exe"
    & $py -c @"
import os
os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
from huggingface_hub import hf_hub_download
for repo, fn in [('Plachta/Seed-VC','DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth'),
                 ('Plachta/Seed-VC','config_dit_mel_seed_uvit_whisper_base_f0_44k.yml')]:
    print('   ', fn)
    hf_hub_download(repo_id=repo, filename=fn, cache_dir='./checkpoints')
print('  Seed-VC 权重就绪')
"@
    Pop-Location
    Use-ProxyEnv -Enable:$false
}

# ============================================================================
Say "`n============================================"
Say "  部署完成 ✔" Green
Say "============================================"
Say "  启动：双击  $(Join-Path $Workspace '启动念山AI.bat')"
Say "  或运行：  cd '$SEEDVC_DIR'; .\run_webapp.ps1"
Say "  地址：    http://127.0.0.1:7860/"
Say ""
Say "  提示：首次启动会加载模型（约 20~60 秒），之后浏览器会自动打开。"
Say "        4 GB 显存下请勿同时运行其他占用显存的程序。"
Say ""
