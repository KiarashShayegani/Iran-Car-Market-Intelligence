# Changelog

## [3.1.0] — 2026-11-10

### F5 — Price estimation, UI upgrades

- **Retrieval-based price estimation** (replaces ML, owner decision):
  `/estimate` + «تخمین قیمت» tab — observed p25/median/p75 from comparable
  ads, calendar-aware year matching, small-sample honesty, full audit trail
  of comparables with source links.
- **Store gains derived columns** (year_num, year_calendar, mileage_km,
  brand_key, model_key) — single derivation point at ingest, idempotent
  migration; index/search/API reuse them. Canonical DB rebuilt from raw.
- **UI upgrades**: quick-fact chips on result cards (formatted
  price/mileage/year from normalized metadata), skeleton loading states,
  demo-query chips, new «💰 تخمین قیمت» tab, app icon.
- **Divar probe** (time-boxed): API reachable from Iran (POST accepted),
  contract is protobuf-JSON (Any-typed) — feasible-unverified, documented.

### F6 — Operations and polish

- `docs/internal/OPERATIONS.md` — full manual start/stop/refresh guide.
- `scripts/rebuild_search_assets.py` — one-shot derived-view rebuild.
- README overhauled to match the shipped product.

[3.1.0]: https://github.com/KiarashShayegani/Iran-Car-Market-Intelligence/releases


All notable changes to ICMI are documented here. The project follows a
versioned, ground-truth-first approach: raw source data is immutable, and
every downstream artifact is traceable to it.

## [3.0.0] — 2026-11-09

### Ground-up rebuild kickoff

- Reset the repository to a clean scaffold. The complete v2 codebase
  (Bama-only price-prediction pipeline: brand-keyword scraper, CatBoost/XGBoost
  per-brand models, Gradio app, daily workflow) is preserved in the
  `archive/v2` branch and the `v2-final` tag, plus a full local backup
  outside the repository.
- New `src/icmi` package skeleton with the target layering:
  `core` (schema, paths, config), `collectors`, `pipeline`, `retrieval`,
  `analytics`.
- `uv`-managed, repo-local environment (`.venv/`, `.uv-cache/`) — fully
  isolated and portable across machines.
- Preserved the Deepnote multi-platform scraping research notebook
  (outputs stripped) at `docs/research/web_scraping_01_code.ipynb`; it is the
  seed for the F1 collectors.
- CI: ruff + pytest on every push/PR.
- Removed ML price prediction as a pillar (kept in history); replaced by
  retrieval-based price estimation on the analytics track.

[3.0.0]: https://github.com/KiarashShayegani/Iran-Car-Market-Intelligence/releases
