# Deploying ICMI to Hugging Face Spaces

This deploys the **Gradio app only** (inference on already-trained models) — Spaces'
free CPU tier is for serving, not for running the full daily scrape+train pipeline.
Keep training happening locally / via the GitHub Action, and push trained artifacts to
the Space.

## 1. Create the Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Pick an owner + name (e.g. `icmi-iran-car-market`).
3. **SDK: Gradio**. Hardware: CPU basic (free) is enough — inference on these models
   is fast, no GPU needed.
4. Visibility: Public (so it's actually usable as a portfolio piece).

This gives you a new git repo at `https://huggingface.co/spaces/<you>/<space-name>`.

## 2. What to upload (and what NOT to)

Spaces needs a working subset of the repo, not everything:

**Upload:**
```
app/gradio_app.py
src/                          # whole package (gradio_app.py imports src.trainer)
config/brands.yaml
requirements.txt
artifacts/models/*.joblib     # every brand model + global.joblib
artifacts/metadata/*.json     # per-brand metadata + training_summary.json
data/processed/processed_latest.parquet   # powers the dashboard + dropdowns
```

**Don't upload:**
```
data/raw/                     # scraping snapshots + master_history - not needed for serving
data/skipped_rows.csv, data/validation_dropped_rows.csv, data/icmi.db
logs/
.github/
```

`artifacts/` and most of `data/` are gitignored in the main repo on purpose (see
`docs/ARCHITECTURE.md`) — for the Space specifically, you need the trained model files
present, so either:
- `git add -f artifacts/models/*.joblib artifacts/metadata/*.json` when committing to
  the Space's repo specifically (force-add past the gitignore), or
- Just drag-and-drop the files in the Spaces web UI's "Files" tab, which doesn't care
  about your local `.gitignore` at all. This is the simplest path for a one-off deploy.

## 3. The Space needs its own `README.md` with a metadata header

Hugging Face Spaces reads YAML front-matter at the top of `README.md` to configure the
Space (this is separate from the project's main README — don't copy that one in, it'll
render as raw text with this frontmatter mixed in). Create a small one just for the
Space:

```markdown
---
title: ICMI - Iran Car Market Intelligence
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app/gradio_app.py
pinned: false
license: mit
---

# ICMI - Iran Car Market Intelligence

ML-powered car price estimation for Iran's automotive market. See the full project
(scraper, training pipeline, docs) at github.com/KiarashShayegani/ICMI.
```

`app_file: app/gradio_app.py` tells Spaces to run that nested file directly — no need
for a root-level `app.py`, since `gradio_app.py` already has
`if __name__ == "__main__": demo.launch()`, and Spaces executes `app_file` as the main
script.

Check `sdk_version` against whatever `gradio` version you actually have installed
(`pip show gradio`) so local behavior matches what Spaces runs.

## 4. Push it

```bash
git clone https://huggingface.co/spaces/<you>/<space-name>
cd <space-name>
# copy in the files listed in step 2, plus the Space README.md from step 3
git add -A
git commit -m "Initial deploy"
git push
```

Spaces will build automatically (pip installs `requirements.txt`) and the app comes up
at `https://huggingface.co/spaces/<you>/<space-name>`. First build takes a few minutes
(catboost/xgboost wheels aren't tiny).

## 5. Updating with fresh models later

Whenever you retrain locally and want the live Space to reflect it: copy the new
`artifacts/models/*.joblib`, `artifacts/metadata/*.json`, and
`data/processed/processed_latest.parquet` into your Space clone, commit, push. There's
no automatic sync from the main GitHub repo's daily pipeline run today — see
`ROADMAP.md` for wiring the GitHub Action to push straight to the Space as a future
CI/CD step (needs an `HF_TOKEN` repo secret and either `huggingface_hub`'s
`upload_folder()` or a second git remote in the Action).

## Notes / gotchas

- **File size limits**: GitHub/HF both require Git LFS for files over ~10MB without
  it (soft limits, hard blocks eventually). None of the current per-brand `.joblib`
  files should be near that with the current `iterations`/`n_estimators` settings, but
  if you increase them substantially, check file sizes before pushing.
- **Don't point the Space at the scraper.** Nothing here needs bama.ir reachable from
  Hugging Face's infrastructure, and there's no reason to run a scraper on someone
  else's serving infra.
- If the Space fails to build, check the "Logs" tab first — almost always a missing
  file (usually a forgotten `artifacts/metadata/*.json` — the Gradio app's "Model Info"
  tab reads `training_summary.json` specifically) or a `requirements.txt` version
  conflict.
