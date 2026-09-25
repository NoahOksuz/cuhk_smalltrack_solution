#!/usr/bin/env bash
# Compatibility wrapper for the installed Python CLI.
#
#   ./inference.sh INPUT [OUTPUT]
#
# Install first from this directory with: pip install -e .
set -euo pipefail

if ! command -v samhar >/dev/null 2>&1; then
  echo "error: samhar is not installed; run: pip install -e ." >&2
  exit 127
fi

exec samhar "$@"
