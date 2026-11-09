# ICMI — Iranian Car Market Intelligence

Multi-source intelligence platform for the Iranian automotive market, built on
publicly available car-advertisement data.

**Two pillars on one data foundation:**

1. **Market Intelligence** — price trends, market statistics, vehicle
   comparisons, and retrieval-based price estimation ("similar ads → market
   range"), all traceable to original listings.
2. **AI Knowledge System** — Persian-capable semantic search over collected
   advertisements, ranked results with original excerpts, direct source links,
   and LLM question answering grounded strictly in retrieved evidence.

## Status

v3.0 — ground-up rebuild in progress. See `docs/internal/MASTER_PLAN.md`
(local, not committed) for the current plan and decision log.

## Data sources

| Platform | Role | Status |
|---|---|---|
| Bama | Marketplace listings | Collector planned (F1) |
| Sheypoor | Marketplace listings | Collector planned (F1) |
| Hamrah Mechanic | Dealer exhibition inventory | Collector planned (F1) |
| Divar | Classifieds | Deferred (access investigation) |
| Karnameh | Certified used-car dealer | Deferred (listing identity) |

All endpoints are experimental findings, not guaranteed API contracts.

## Development

Use the repo-local wrapper — it keeps **all** caches and environments inside
the repository directory and never touches machine-global caches shared with
other projects:

```bash
scripts/dev.sh sync              # create .venv, install locked dependencies
scripts/dev.sh run pytest        # run tests
scripts/dev.sh run ruff check .
```

On Windows (PowerShell): `.\scripts\dev.ps1 sync`, etc.

Everything (virtualenv, package cache, datasets, model caches) lives inside
the repository directory, isolated from the rest of the system. Copying the
repository directory to another machine is sufficient; then run the sync
command and `uv run python scripts/sync_latest_data.py` to fetch the latest
data layer.

## License

MIT — see [LICENSE](LICENSE).
