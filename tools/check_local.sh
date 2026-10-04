#!/usr/bin/env bash
# 本地预演 GitHub Actions 的检查逻辑（与 .github/workflows/check.yml 保持一致）
#
# 用法（需 bash，Windows 上可用 Git 自带的 D:\Git\bin\bash.exe）：
#     bash tools/check_local.sh
set -u
cd "$(dirname "$0")/.." || exit 1

fail=0

echo "== 1) 权重/音频文件检查 =="
if git ls-files | grep -Ei '\.(pth|pt|ckpt|bin|safetensors|index|wav|mp3|flac|m4a|ogg)$'; then
  echo "   FAIL：仓库中不应包含模型权重或音频文件"
  fail=1
else
  echo "   OK 无权重/音频文件"
fi

echo
echo "== 2) 绝对路径检查 =="
hits=$(git grep -n -I -E '(^|[^A-Za-z])[A-Za-z]:[\\/]' -- '*.py' '*.ps1' '*.bat' \
       | grep -viE 'https?://|127\.0\.0\.1' || true)
if [ -n "$hits" ]; then
  echo "   WARN：发现疑似绝对路径"
  echo "$hits"
else
  echo "   OK 未发现绝对路径"
fi

echo
echo "== 3) 大文件体积检查（> 5 MB） =="
big=""
while IFS= read -r f; do
  [ -f "$f" ] || continue
  s=$(stat -c%s "$f")
  if [ "$s" -gt 5242880 ]; then
    big="${big}${f} ($((s / 1024 / 1024)) MB)"$'\n'
  fi
done < <(git ls-files)
if [ -n "$big" ]; then
  echo "   FAIL：存在超过 5 MB 的文件"
  printf '%s' "$big"
  fail=1
else
  echo "   OK 无超大文件"
fi

echo
echo "== 4) 补丁文件必须是 git 可读的纯文本 =="
for p in patches/*.patch; do
  if grep -qP '\x00' "$p"; then
    echo "   FAIL $p 含 NUL 字节（疑似 UTF-16 编码，git apply 会失败）"
    fail=1
    continue
  fi
  if ! head -c 20 "$p" | grep -q '^diff --git'; then
    echo "   FAIL $p 未以 'diff --git' 开头"
    fail=1
    continue
  fi
  echo "   OK   $p ($(wc -c < "$p") 字节)"
done

echo
if [ "$fail" -ne 0 ]; then
  echo "存在失败项"
  exit 1
fi
echo "全部检查通过"
