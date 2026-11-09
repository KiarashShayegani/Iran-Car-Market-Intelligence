#!/usr/bin/env bash
# ICMI development wrapper (Linux/macOS).
#
# Guarantees full tooling isolation: every ICMI command runs with caches and
# environments inside this repository directory. The globally exported
# UV_CACHE_DIR (shared with other projects, e.g. HOOSHYAR) is overridden for
# ICMI only — nothing outside this repo is read or written.
#
# Usage:  scripts/dev.sh <uv-args...>
#         scripts/dev.sh sync
#         scripts/dev.sh run pytest -q
#         scripts/dev.sh run ruff check .
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export UV_CACHE_DIR="$REPO_ROOT/.uv-cache"
export UV_PYTHON_PREFERENCE="only-system"

# Load repo-local secrets/env (.env is gitignored; never committed).
if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$REPO_ROOT/.env"
    set +a
fi

exec uv "$@"
