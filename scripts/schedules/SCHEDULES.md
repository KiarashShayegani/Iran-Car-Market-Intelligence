# Scheduled Execution — Templates (ALL DISABLED)

> Owner rule: schedules are built but **disabled by default**; enabling one
> is an explicit owner action. All wrappers call the SAME unified pipeline
> (`scripts/pipeline.py`) used by manual runs — no separate implementation.

## Linux — cron

Edit `crontab -e` and UNCOMMENT (times are examples; `--mode scheduled` is
recorded in manifests):

```cron
# ICMI collection + view refresh, every night 02:30
# 30 2 * * * cd /data/work/projects/Iran-Car-Market-Intelligence && scripts/dev.sh run python scripts/pipeline.py collect all --mode scheduled --max-pages 5 --target-listings 120 >> logs/scheduled.log 2>&1
```

Recommended cadence by platform behavior: Bama/Sheypoor nightly or every
other night; Hamrah Mechanic 2–3×/week (dealer inventory churns slower).

## Linux — systemd timer (alternative to cron)

```ini
# ~/.config/systemd/user/icmi-pipeline.service
[Unit]
Description=ICMI scheduled pipeline

[Service]
Type=oneshot
WorkingDirectory=/data/work/projects/Iran-Car-Market-Intelligence
ExecStart=/data/work/projects/Iran-Car-Market-Intelligence/scripts/dev.sh run python scripts/pipeline.py collect all --mode scheduled --max-pages 5 --target-listings 120

# ~/.config/systemd/user/icmi-pipeline.timer
[Unit]
Description=Run ICMI pipeline nightly

[Timer]
OnCalendar=*-*-* 02:30:00
Persistent=true

[Install]
WantedBy=timers.target
```

Enable with `systemctl --user enable --now icmi-pipeline.timer` (owner action).

## Windows 10 — Task Scheduler

```powershell
# Register (owner action; runs whether logged in or not only if chosen):
schtasks /Create /TN "ICMI pipeline" /SC DAILY /ST 02:30 /TR ^
  "cmd /c cd /d C:\path\to\Iran-Car-Market-Intelligence && .\scripts\dev.ps1 run python scripts\pipeline.py collect all --mode scheduled --max-pages 5 --target-listings 120 >> logs\scheduled.log 2>&1"
```

(Windows path + wrapper untested — marked honest-unknown until validated on
the owner's Windows laptop.)

## Safety model (already in the pipeline)

- `data/.pipeline.lock` prevents overlapping runs; stale locks (>6 h) are
  taken over automatically (crash recovery).
- Stages are idempotent: rerunning the same command is always safe.
- Failure behavior: per-stage ✓/✗ summary; the process exits non-zero; the
  authoritative DB only ever changes through idempotent ingestion.
- Logs: append to `logs/scheduled.log` (gitignored); per-run manifests land
  in `data/manifests/` + tracked `manifests/`.
- Manual override: run any pipeline command by hand at any time — the lock
  serializes; a manual run during a scheduled window simply waits or fails
  with a clear message.
