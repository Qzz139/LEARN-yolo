#!/bin/zsh
# macOS 双击入口：委托公共启动脚本，失败时保留终端供查看错误。

# macOS Finder/Terminal entry point. Shared Python discovery lives in run.sh.

PREVIEW_DIR="$(cd "$(dirname "$0")" && pwd)"
"$PREVIEW_DIR/run.sh" "$@"
# 立即保存子进程退出码，避免后续提示命令覆盖失败状态。
PREVIEW_STATUS=$?

if [ $PREVIEW_STATUS -ne 0 ] && [ -t 0 ]; then
  echo
  read "?Program failed; press Return to close..."
fi

exit $PREVIEW_STATUS
