@echo off
chcp 936 >nul
title 停止念山AI
powershell -NoProfile -ExecutionPolicy Bypass -Command "$c = Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue; if ($c) { $c | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }; Write-Host '念山AI 已停止。' } else { Write-Host '念山AI 当前未在运行。' }"
pause
