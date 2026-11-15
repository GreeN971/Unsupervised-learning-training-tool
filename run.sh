#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY="$(command -v python3 || command -v python || true)"
if [ -z "$PY" ]; then
  echo "Python 3 not found on PATH. Please install Python and try again." >&2
  exit 1
fi

exec "$PY" "$SCRIPT_DIR/bootstrap.py" "$@"
