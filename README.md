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

```bash
uv sync          # create .venv and install locked dependencies
uv run pytest    # run tests
uv run ruff check .
```

Everything (virtualenv, caches, datasets) lives inside the repository
directory, isolated from the rest of the system. Copying the repository
directory to another machine (including Windows) is sufficient; then run
`uv sync` and `uv run python scripts/sync_latest_data.py` to fetch the latest
data layer.

## License

MIT — see [LICENSE](LICENSE).
