@echo off
chcp 936 >nul
title 念山AI
set "HERE=%~dp0"
echo ============================================
echo   念山AI · 歌声工作台
echo ============================================
powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"
if %errorlevel%==0 (
  echo.
  echo   念山AI 已在运行，正在打开浏览器...
  start http://127.0.0.1:7860/
  ping 127.0.0.1 -n 3 >nul
  exit /b
)
echo.
echo   正在启动（首次加载模型约需 20~60 秒）...
echo   启动成功后会自动打开浏览器。
echo.
echo   停止方法: 直接关闭本窗口，或双击「停止念山AI.bat」
echo ============================================
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%seed-vc\run_webapp.ps1"
pause
