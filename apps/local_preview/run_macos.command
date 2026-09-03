#!/bin/zsh

PREVIEW_DIR="$(cd "$(dirname "$0")" && pwd)"
"$PREVIEW_DIR/run.sh" "$@"
PREVIEW_STATUS=$?

if [ $PREVIEW_STATUS -ne 0 ] && [ -t 0 ]; then
  echo
  read "?Program failed; press Return to close..."
fi

exit $PREVIEW_STATUS
