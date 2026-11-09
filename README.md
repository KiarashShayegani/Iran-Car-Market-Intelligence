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

**Always use the repo-local wrappers** — `scripts/dev.sh` (Linux/macOS) or
`.\scripts\dev.ps1` (Windows PowerShell). They guarantee ICMI's full tooling
isolation:

- `UV_CACHE_DIR` is pointed at the repo's own `.uv-cache/`, **overriding any
  machine-global uv cache** — so ICMI never reads, writes, or conflicts with
  the caches of other projects on the machine.
- A gitignored repo-local `.env` is loaded into the environment (package
  index overrides, API keys, R2 credentials — see `.env.example`).

```bash
scripts/dev.sh sync              # create .venv, install locked dependencies
scripts/dev.sh run pytest        # run tests
scripts/dev.sh run ruff check .
```

```powershell
.\scripts\dev.ps1 sync
.\scripts\dev.ps1 run pytest
```

### Package index

Standard PyPI (`https://pypi.org/simple`) is the default — no network-specific
mirror is imposed by the repository. If pypi.org is unreachable from your
network (e.g. from Iran), copy `.env.example` to `.env` locally and uncomment
the `UV_DEFAULT_INDEX` mirror line. That choice stays on your machine and is
never committed.

Everything (virtualenv, package cache, datasets, model caches) lives inside
the repository directory — the data layer is **local-first**: raw JSON/JSONL,
canonical SQLite, analytical Parquet, manifests, and (later) retrieval indexes
are all repo-local files. Copying the repository directory to another machine
is sufficient. GitHub Actions runs the same pipeline experimentally (dispatch
only): it delivers results as downloadable artifacts, which
`python scripts/sync_latest_data.py --source <artifact.zip>` imports locally —
the local canonical DB stays authoritative and ingestion is idempotent.

## License

MIT — see [LICENSE](LICENSE).
