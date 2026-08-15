# Roadmap

Where ICMI is headed, roughly ordered by effort/impact. Nothing here is committed —
it's a working list to prioritize from.

## Near-term (data quality & correctness)

- **Widen the fuel category list.** `data/validation_dropped_rows.csv` will now show
  exactly which fuel values are getting dropped (357 rows in the last full run,
  almost certainly diesel/hybrid from imported brands). Once confirmed, add them to
  `validator.py`'s `isin()` check and `config/brands.yaml`'s `fuel_mapping`.
- **Persian text normalization for `name`/`trim`.** These are free-typed by sellers —
  typos, trailing spaces (beyond the existing `.strip()`), and partial trim names
  (`"111"` vs `"111 SE"`) currently all become distinct one-hot categories, which
  fragments the feature space for XGBoost/RandomForest/MLP especially (CatBoost's
  native categorical handling is more forgiving of this, which is likely part of why
  it tends to win once the clone() bug is fixed). A canonical lookup table or fuzzy
  matching (e.g. `rapidfuzz`) to consolidate near-duplicates would help all four
  models, not just CatBoost.
- **Real Jalali/Shamsi calendar support.** `current_year_shamsi` is a manually-updated
  config value and the Gregorian→Shamsi conversion is a year-only approximation.
  Swapping in the `jdatetime` library would make both exact and remove a manual
  yearly maintenance step.
- **Unit tests.** Three real bugs were found by hand-testing this session alone
  (schema column stripping, SQL column mismatch, CatBoost/clone incompatibility) —
  a `pytest` suite covering `scraper._parse_ad`, `validator`, and `cleaner` against
  fixture payloads would catch regressions automatically instead of relying on
  reading logs after the fact.

## Mid-term (product depth)

- **"Days on market" / price-drop tracking.** `master_history.parquet` already keeps
  `first_seen_at` per listing. Surfacing "this car has been listed 45 days, price
  dropped 5% since first seen" is a genuinely differentiated market-intelligence
  feature beyond a point price estimate — valuable for both buyers and sellers, and
  distinct from what a generic listings site shows.
- **Real prediction intervals, not a heuristic.** The current confidence interval is
  computed from the std-dev of comparable listings in the market, which is a
  reasonable fallback but not a real prediction interval. CatBoost supports quantile
  regression loss; training upper/lower quantile models per brand would give a
  statistically grounded interval instead.
- **Feature importance / explainability in the "Model Info" tab.** `feature_importances_`
  (tree models) or SHAP values, surfaced per brand, would make predictions more
  trustworthy and turn the Gradio app from a "black box price generator" into
  something closer to an actual analytical tool — which is the stated goal.
- **Finer-grained models once data volume supports it.** Currently one model per
  *brand*. Once `master_history.parquet` accumulates enough history, per-(brand,name)
  models (e.g. a dedicated Peugeot 206 model vs. a dedicated Peugeot Pars model)
  become viable and would likely outperform brand-level models — gate this on actual
  row counts, don't do it prematurely on the current data volume.

## Longer-term (system, not demo)

- **CI on every PR.** A lightweight GitHub Actions workflow running lint (`ruff`) +
  the pytest suite above, separate from the daily scrape/train pipeline.
- **Wire the daily GitHub Action to push straight to the HF Space.** Right now
  `daily_pipeline.yml`'s `git add data/ artifacts/` step is a no-op (those paths are
  gitignored on purpose, see `docs/ARCHITECTURE.md`). Rather than reversing that and
  bloating the main repo's git history with daily binary commits, add a step that
  pushes fresh `artifacts/` + `processed_latest.parquet` directly to the Space's git
  remote (needs an `HF_TOKEN` secret) — keeps the portfolio repo clean while the live
  demo actually stays current automatically.
- **A thin API layer.** A small FastAPI wrapper around the trained pipelines would let
  other tools query price estimates programmatically, decoupling "the model" from
  "the Gradio UI" — the difference between a demo and a system other things can build
  on.
- **Model performance history / drift tracking.** Log MAE/R² per brand per training
  run (a simple append-only file or a small SQLite table) so degradation over time —
  or a *genuine* improvement from more data — is visible and trackable, not just
  whatever the latest `training_summary.json` happens to say.
- **Scraping etiquette as the project scales.** Continue respecting reasonable
  request pacing (already has jitter + inter-brand cooldown + retry backoff). If
  scraping frequency or brand coverage grows, revisit bama.ir's robots.txt/ToS and
  consider caching unchanged listings rather than re-fetching them.
