# Changelog

All notable changes to ICMI are documented here. Dates are approximate (development
sessions), not release dates.

## v2.0.2 (unreleased — this patch)

### Fixed
- **`trainer.py`**: restored the manual `KFold` cross-validation loop. A regression
  had reintroduced `sklearn.cross_val_score`, which calls `clone()` internally and
  silently fails CatBoost on every single call (CatBoost's `cat_features` constructor
  param breaks sklearn's clone contract). Confirmed via logs: CatBoost was failing on
  all 14 brands, meaning `random_forest`/`xgboost` were winning by default rather than
  genuine comparison.
- **`.gitignore`**: had unresolved git merge-conflict markers (`=======`,
  `>>>>>>> d9f72d...`) and blanket-ignored all of `data/` and `artifacts/`. Replaced
  with a clean, conflict-free version.
- **`scraper.py`**: restored `data/skipped_rows.csv` diagnostic logging (present in an
  earlier patch, lost in a merge). Also hardened `_parse_ad()` against explicit
  `"detail": null` / `"price": null` API responses — `.get(key, default)` only applies
  `default` when the key is *missing*, not when it's present with value `None`, which
  was the actual root cause of an intermittent `'NoneType' object has no attribute
  'get'` crash that killed an entire brand's scrape. Broadened the caught exception
  tuple to include `AttributeError` as a second line of defense.

### Added
- **`validator.py`**: `data/validation_dropped_rows.csv` — mirrors the scraper's
  skipped-rows diagnostics one stage downstream. When a schema check drops rows (e.g.
  "357 rows failed a fuel check"), the actual offending rows and which check they
  failed are now written out, instead of only a count in the logs.
- `docs/ARCHITECTURE.md`, `docs/DEPLOYMENT.md`, `CHANGELOG.md`, `ROADMAP.md`.

## v2.0.1

### Fixed
- **`scraper.py`**: Gregorian-calendar years (common on `manufacturer: imported`
  brands like toyota/kia/kmc) were being dropped as `invalid_year` instead of
  converted, losing 60-95% of some brands' data. Added `_normalize_year()` to detect
  and convert (`year - 621` approximation) instead of dropping.
- **`validator.py`**: `RAW_SCHEMA` was missing `name`/`trim` (and other) columns; with
  `strict="filter"`, this silently deleted them right after scraping, then
  `validate_processed()` crashed because it required those same columns.
- **`database.py`**: `upsert_listings()` did `INSERT ... SELECT *`, assuming the
  incoming DataFrame's columns exactly matched the `listings` table. Processed data
  carries ML-only columns (e.g. `age`) the table doesn't define, and can be missing
  metadata columns the table does define — this now builds an explicit column list
  from the table's actual schema (`PRAGMA table_info`) intersected with what's in the
  DataFrame.
- **`run_pipeline.py`**: fixed a deferred `import pandas` that only existed inside
  `if __name__ == "__main__"`, causing a `NameError` when `main()` used `pd` earlier.
  Bumped default `max_pages_per_brand` from 10 to 30 (then to config-driven).

### Added
- Multi-model training (`trainer.py`): CatBoost, XGBoost, RandomForest, MLP compared
  per brand via cross-validated R², with a global fallback model for brands with too
  little data. Previously a single fixed CatBoost model, no comparison.
- Cumulative dataset growth: `append_to_master()` merges every scrape into
  `data/raw/master_history.parquet` instead of overwriting a single day's snapshot.
- `skip_reasons` counters + `data/skipped_rows.csv` in the scraper, for visibility
  into what's being filtered and why (previously only a percentage with no reason).
- Non-fatal, row-level schema validation (`coerce=True` + drop-only-the-bad-rows)
  instead of the whole pipeline crashing on a handful of corrupted rows.

## v2.0.0

Initial rebuild from a single-notebook v1: scraper, pandera validation, cleaner,
SQLite database, single CatBoost trainer, Gradio app, and a daily GitHub Actions
pipeline scaffold.
