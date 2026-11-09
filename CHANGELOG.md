# Changelog

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
