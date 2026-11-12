# Scheduled Execution — Templates (ALL DISABLED)

> Owner rule: schedules are built but **disabled by default**; enabling one
> is an explicit owner action. All wrappers call the SAME unified pipeline
> (`scripts/pipeline.py`) used by manual runs — no separate implementation.

## Proposed production schedule (2026-11-13 — AWAITING OWNER APPROVAL)

**Cloud collection (GitHub Actions, `h-ops-pipeline.yml` cron — commented):**

| Source | Cadence | Budget/run | Evidence base |
|---|---|---|---|
| Sheypoor | nightly 02:30 UTC | 3 pages, ≤60 target, ≤5 requests | 3 cloud runs OK (60/60/30 rec) |
| Hamrah Mechanic | Mon/Wed/Fri 02:30 UTC | 3 pages, ≤60 target, ≤5 requests | 3 cloud runs OK (60/24 rec) |
| Bama | nightly 02:30 UTC | **2 presets × 2 pages**, ≤60 target, ≤8 requests | E1+E3 cloud runs OK; see burst note |

**Bama burst note (evidence-based):** the only cloud Bama failure
(2026-11-12, 12/12 pages failed) followed **4 collection bursts within 25
minutes** on the previous night (4 runner runs × up to 5 pages). Both
post-burst light-budget runs succeeded (E1: 1 page/21 s; E3: 1 preset in
the production pipeline). Recommendation: single nightly run, 2 rotating
presets × 2 pages (≈8 requests — well under the historical success rate),
preset rotation advancing daily through the 14-brand list. If a nightly
run ever fails, it is NOT retried the same night (job failure is
graceful — other sources unaffected; the manifest records it).

**Local retrieval (Linux cron / systemd timer / Windows Task Scheduler):**

| Step | When | Command |
|---|---|---|
| Retrieve + ingest | nightly **04:30 UTC** (2 h after collection; publish completes in ≈5 min) | `python scripts/sync_actions_artifact.py --pending` |
| Health report | after retrieval (same wrapper) | `python scripts/ops_status.py` |

A missed local window loses nothing: pending detection converges via the
import ledger, artifacts persist 30 days, imports are idempotent and
chronological; `--check` warns when a pending artifact is within 7 days of
expiry.

**Bama local fallback (only if cloud Bama regresses):** nightly 21:30 local
time via the existing pipeline wrapper with `--platforms bama`; rotation
already handled by the collection policy.

## Linux — cron (templates — UNCOMMENT to enable)

```cron
# Cloud collection — h-ops-pipeline.yml: keep cron line commented in the
# workflow file; activating it is the H-ops-5 owner gate (GitHub-side).

# Local retrieval + ingest, every night 04:30 UTC:
# 30 4 * * * cd /data/work/projects/Iran-Car-Market-Intelligence && set -a && . ./.env && set +a && .venv/bin/python scripts/sync_actions_artifact.py --pending >> logs/sync.log 2>&1

# Local Bama fallback (ONLY if cloud Bama is disabled again):
# 30 21 * * * cd /data/work/projects/Iran-Car-Market-Intelligence && .venv/bin/python scripts/pipeline.py collect all --platforms bama --mode scheduled --max-pages 2 --target-listings 60 >> logs/scheduled.log 2>&1
```

## Linux — systemd timer (alternative to cron)

```ini
# ~/.config/systemd/user/icmi-sync.service
[Unit]
Description=ICMI artifact retrieval + ingest

[Service]
Type=oneshot
WorkingDirectory=/data/work/projects/Iran-Car-Market-Intelligence
# GITHUB_TOKEN must be available to the service (EnvironmentFile=.env)
EnvironmentFile=/data/work/projects/Iran-Car-Market-Intelligence/.env
ExecStart=/data/work/projects/Iran-Car-Market-Intelligence/.venv/bin/python scripts/sync_actions_artifact.py --pending

# ~/.config/systemd/user/icmi-sync.timer
[Unit]
Description=Retrieve ICMI artifacts nightly

[Timer]
OnCalendar=*-*-* 04:30:00
Persistent=true

[Install]
WantedBy=timers.target
```

`Persistent=true` catches missed windows after downtime (systemd runs the
missed job at next boot — matching the never-lose-an-artifact rule).

## Windows 10 — Task Scheduler

```powershell
# Register (owner action):
schtasks /Create /TN "ICMI artifact sync" /SC DAILY /ST 04:30 /TR ^
  "cmd /c cd /d C:\path\to\Iran-Car-Market-Intelligence && .venv\Scripts\python.exe scripts\sync_actions_artifact.py --pending >> logs\sync.log 2>&1"
```

(Windows path + Task Scheduler untested — marked honest-unknown until
validated on the owner's Windows laptop.)

## Cloud lifecycle (complete)

```
GitHub Actions (02:30 UTC, dispatch-only today)
  → per-source collect (failure-isolated jobs, bounded budgets)
  → per-source artifacts (collect-*, 1-day retention)
  → validate-and-publish: merge → bundle-meta.json → SHA256SUMS
  → icmi-ingestion-<run_id> artifact (30-day retention)
Local (04:30 UTC nightly)
  → --pending: discover un-imported runs (ledger-based)
  → download → CRC + traversal → SHA256SUMS + bundle-meta cross-check
  → write-once install (identical=skip, differing=ABORT)
  → pre-import DB backup (SQLite backup API) → shrink-guard
  → standard ingest (run-level skip, record-level dedup)
  → idempotence probe (second pass must change nothing)
  → ledger entry (actions-run-<id>.json)
  → derived refresh when needed: pipeline.py export documents index
  → ops_status.py health report
```

## Safety model (already in the pipeline)

- `data/.pipeline.lock` prevents overlapping runs; stale locks (>6 h) are
  taken over automatically (crash recovery).
- Stages are idempotent: rerunning the same command is always safe.
- Failure behavior: per-stage ✓/✗ summary; the process exits non-zero; the
  authoritative DB only ever changes through idempotent ingestion.
- Importer guards: verification aborts before any local change; write-once
  install aborts on conflicting local files; DB shrinkage auto-restores the
  pre-import backup; interrupted imports are completed by rerunning
  (install skips identical files; ingest skips ingested runs).
- Logs: append to `logs/sync.log` / `logs/scheduled.log` (gitignored);
  per-run manifests land in `data/manifests/` + tracked `manifests/`.
- Manual override: run any command by hand at any time — the lock
  serializes; a manual run during a scheduled window simply waits or fails
  with a clear message.
