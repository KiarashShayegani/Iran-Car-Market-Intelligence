#!/bin/bash
# ICMI daily-use launcher: PRODUCTION frontend + backend.
# (npm run dev is for development only — it compiles per request and is heavy.)
set -a; source "$(dirname "$0")/../.env"; set +a
cd "$(dirname "$0")/.."
mkdir -p logs

echo "[1/3] building frontend (production)…"
(cd web && npm run build)

echo "[2/3] starting backend on :8000…"
setsid nohup uv run python scripts/run_api.py > logs/api.log 2>&1 < /dev/null &

echo "[3/3] starting production frontend on :3000…"
setsid nohup npm --prefix web run start > logs/frontend.log 2>&1 &

sleep 5
curl -s -o /dev/null -w "backend: %{http_code}\n" http://127.0.0.1:8000/health
curl -s -o /dev/null -w "frontend: %{http_code}\n" http://localhost:3000/
echo "ICMI ready — http://localhost:3000"
