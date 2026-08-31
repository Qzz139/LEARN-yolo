#!/bin/zsh

# Double-click this file on macOS, or run it from a terminal. The launcher uses
# the first local Python that already has OpenCV and NumPy available.

set -u

PREVIEW_DIR="$(cd "$(dirname "$0")" && pwd)"
PREVIEW_SCRIPT="$PREVIEW_DIR/main.py"
PREVIEW_PYTHON=""

for candidate in \
  "$PREVIEW_DIR/.venv/bin/python" \
  "/Users/zzq/micromamba/bin/python3" \
  "/opt/homebrew/bin/python3" \
  "/usr/local/bin/python3" \
  "/usr/bin/python3"
do
  if [ -x "$candidate" ] && "$candidate" -c 'import cv2, numpy' >/dev/null 2>&1; then
    PREVIEW_PYTHON="$candidate"
    break
  fi
done

if [ -z "$PREVIEW_PYTHON" ]; then
  echo "没有找到带 OpenCV 的 Python。请执行："
  echo "  cd \"$PREVIEW_DIR\""
  echo "  python3 -m venv .venv"
  echo "  .venv/bin/python -m pip install -r requirements.txt"
  echo
  read "?按回车键关闭..."
  exit 1
fi

echo "Python: $PREVIEW_PYTHON"
echo "启动本机 YOLO 预览；按 C/回车或点击画面拍照，按 Q/Esc 退出。"
echo

"$PREVIEW_PYTHON" "$PREVIEW_SCRIPT" "$@"
PREVIEW_STATUS=$?

if [ $PREVIEW_STATUS -ne 0 ]; then
  echo
  read "?程序运行失败；按回车键关闭..."
fi

exit $PREVIEW_STATUS
