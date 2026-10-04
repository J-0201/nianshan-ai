<#
.SYNOPSIS
    启动「念山AI · 歌声转换」网页应用（本地浏览器使用）。

.DESCRIPTION
    封装 app_nianshan.py，自动处理：
      - 修复 torchaudio fbank 在 CUDA 上的崩溃（否则转换必报错）
      - 把已部署的 ffmpeg/ffprobe 指给 pydub（否则 mp3 输出失败）
      - 缓存与输出目录重定向到 F 盘，避免写 C 盘
      - HF 流量走本地代理（权重已缓存，这里只是兜底）

    界面与后端在 app_nianshan.py（复用官方 app_svc.py 的模型代码）。
    官方原始界面仍可通过 app_launch.py 启动（对照用）。

.EXAMPLE
    .\run_webapp.ps1
    .\run_webapp.ps1 -Port 7861
#>
param(
    [int]$Port = 7860,
    [string]$BindHost = "127.0.0.1",
    [switch]$Share
)

$proj = $PSScriptRoot
$py = Join-Path $proj ".venv\Scripts\python.exe"
$launcher = Join-Path $proj "app_nianshan.py"

if (-not (Test-Path $py)) { throw "找不到虚拟环境 Python: $py" }
if (-not (Test-Path $launcher)) { throw "找不到启动器: $launcher" }

# 端口占用检查：避免误以为启动失败
$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($busy) { throw "端口 $Port 已被占用（PID $($busy.OwningProcess)），请用 -Port 换一个端口" }

# 缓存重定向到项目盘（本机 C 盘仅剩约 54GB）
$ckpt = Join-Path $proj "checkpoints"
foreach ($d in @($ckpt, (Join-Path $ckpt "hf_cache"), (Join-Path $ckpt "hf_home"),
                 (Join-Path $ckpt "hf_home\transformers"))) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
}
$env:HF_HOME = Join-Path $ckpt "hf_home"
$env:TRANSFORMERS_CACHE = Join-Path $ckpt "hf_home\transformers"

# Gradio 自身配置（用官方环境变量，无需改上游代码）
$env:GRADIO_SERVER_NAME = $BindHost
$env:GRADIO_SERVER_PORT = "$Port"
$env:GRADIO_ANALYTICS_ENABLED = "False"

# Python 输出统一 UTF-8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

# HF 下载通道（可选：本地代理在时走代理）
$clashPort = 7897
$proxyAlive = Test-NetConnection -ComputerName 127.0.0.1 -Port $clashPort `
                                 -InformationLevel Quiet -WarningAction SilentlyContinue
if ($proxyAlive) {
    $env:HTTPS_PROXY = "http://127.0.0.1:$clashPort"
    $env:HTTP_PROXY = "http://127.0.0.1:$clashPort"
}
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

# 权重缓存判断：
#   - 已缓存 -> 强制离线。hf_hub_download 即使命中缓存也会先发 HEAD 校验，
#     一旦代理/加速工具对 HF 做 TLS 拦截就会抛 SSLCertVerificationError 导致应用起不来。
#   - 未缓存 -> 保持联网，让上游代码在首次启动时自动下载权重。
$cachedDit = Get-ChildItem -Path $ckpt -Recurse -Filter 'DiT_seed_v2*.pth' `
                           -ErrorAction SilentlyContinue | Select-Object -First 1
if ($cachedDit) {
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
    Write-Host "  权重      : 已缓存（离线模式）" -ForegroundColor DarkGray
} else {
    Write-Host "  权重      : 未缓存，首次启动将自动下载约 2.4 GB，请耐心等待" -ForegroundColor Yellow
}

# Gradio 的输出临时目录也放到项目盘，便于找到产物、避免写 C 盘
$env:GRADIO_TEMP_DIR = Join-Path $proj "gradio_tmp"
if (-not (Test-Path $env:GRADIO_TEMP_DIR)) {
    New-Item -ItemType Directory -Force -Path $env:GRADIO_TEMP_DIR | Out-Null
}

# 必须切到项目目录再启动！
# app_svc.py 内部用的是相对路径（'./checkpoints/hf_cache'、'./checkpoints'、
# 界面示例 'examples/source/...'），若在别的目录启动，就会把 1.4GB 权重
# 重新下载到错误的盘位，示例音频也会加载不出来。
Set-Location $proj

Write-Host "念山AI · 歌声转换" -ForegroundColor Cyan
Write-Host "  地址     : http://${BindHost}:$Port/" -ForegroundColor Green
Write-Host "  首次访问 : 需等待模型加载完毕（约 20~40 秒）" -ForegroundColor DarkGray
Write-Host "  停止     : 在此窗口按 Ctrl+C`n" -ForegroundColor DarkGray

& $py $launcher