#!/usr/bin/env sh

set -u

PREVIEW_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$(CDPATH= cd -- "$PREVIEW_DIR/../.." && pwd)
PREVIEW_PYTHON=""

for candidate in \
  "$PREVIEW_DIR/.venv/bin/python" \
  "$PROJECT_DIR/.venv/bin/python" \
  "$(command -v python3 2>/dev/null || true)" \
  "$(command -v python 2>/dev/null || true)"
do
  if [ -n "$candidate" ] && [ -x "$candidate" ] && \
     "$candidate" -c 'import cv2, numpy' >/dev/null 2>&1
  then
    PREVIEW_PYTHON="$candidate"
    break
  fi
done

if [ -z "$PREVIEW_PYTHON" ]; then
  echo "No Python with OpenCV and NumPy was found."
  echo "Create the application environment with:"
  echo "  cd \"$PREVIEW_DIR\""
  echo "  python3 -m venv .venv"
  echo "  .venv/bin/python -m pip install -r requirements.txt"
  exit 1
fi

exec "$PREVIEW_PYTHON" "$PREVIEW_DIR/launcher.py" "$@"
