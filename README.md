# ICMI v2.0 — Iran Car Market Intelligence

> ML-powered car price estimation and market analytics for Iran's automotive market.
> Multi-brand support | Automated pipeline | HF Spaces deployment

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🚀 Quick Start

```bash
# 1. Clone and enter
git clone https://github.com/KiarashShayegani/ICMI.git
cd ICMI

# 2. Create virtual environment (Python 3.11+ required)
python -m venv venv

# macOS / Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline (scrape → clean → train)
python run_pipeline.py

# 5. Launch the Gradio app locally
python app/gradio_app.py
# Then open http://localhost:7860 in your browser
```

## 📁 Project Structure

```
ICMI/
├── .github/
│   └── workflows/
│       └── daily_pipeline.yml      # GitHub Actions automation
├── app/
│   └── gradio_app.py               # HF Spaces Gradio interface
├── artifacts/
│   ├── models/                     # Serialized CatBoost models
│   ├── scalers/                    # Scalers & encoders
│   └── metadata/                   # Model metadata & feature lists
├── config/
│   └── brands.yaml                 # All brand configurations
├── data/
│   ├── raw/                        # Raw scrapes (Parquet)
│   ├── processed/                  # Cleaned data (Parquet)
│   └── icmi.db                     # SQLite database
├── logs/                           # Pipeline logs
├── src/
│   ├── __init__.py
│   ├── utils.py                    # Logging & helpers
│   ├── scraper.py                  # Bama.ir scraper with retries
│   ├── validator.py                # Pandera data validation
│   ├── cleaner.py                  # Feature engineering
│   ├── database.py                 # SQLite + CSV export
│   └── trainer.py                  # CatBoost training
├── run_pipeline.py                 # Full orchestrator
├── pyproject.toml                  # Professional Python packaging
├── requirements.txt                # Dependencies (Python 3.11+)
└── README.md                       # This file
```
