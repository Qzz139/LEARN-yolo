#!/bin/zsh
# macOS 兼容入口：共享 run.sh 的环境查找逻辑，并透传全部参数。

PREVIEW_DIR="$(cd "$(dirname "$0")" && pwd)"
"$PREVIEW_DIR/run.sh" "$@"
PREVIEW_STATUS=$?

# 仅在交互终端且运行失败时等待回车，自动化调用不会被提示阻塞。
if [ $PREVIEW_STATUS -ne 0 ] && [ -t 0 ]; then
  echo
  read "?Program failed; press Return to close..."
fi

exit $PREVIEW_STATUS
