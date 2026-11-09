"""Repo-relative path resolution for the Data Layer.

Every ICMI artifact lives inside the repository directory. This keeps the
project self-contained and portable: cloning the repo directory to another
machine (including Windows) brings the whole data layer along. The default
layout is:

    data/
        raw/         per-run raw payloads (JSONL), immutable
        canonical/   current canonical dataset (SQLite + exports)
        index/       RAG vector index artifacts
        manifests/   run manifests and quality reports
        models/      local model weights / embedding caches

ICMI_DATA_DIR overrides the location (e.g. for tests or split storage).
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_SUBDIRS = ("raw", "canonical", "index", "manifests", "models")


def repo_root() -> Path:
    """Return the repository root (the directory containing src/icmi)."""
    return Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    """Return the Data Layer root, creating it if needed."""
    override = os.environ.get("ICMI_DATA_DIR")
    path = Path(override).resolve() if override else repo_root() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _subdir(name: str) -> Path:
    path = data_dir() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_dir() -> Path:
    """Raw, immutable source payloads, partitioned per run."""
    return _subdir("raw")


def canonical_dir() -> Path:
    """Canonical dataset (current state + price history database)."""
    return _subdir("canonical")


def index_dir() -> Path:
    """RAG vector index artifacts."""
    return _subdir("index")


def manifests_dir() -> Path:
    """Run manifests and quality reports."""
    return _subdir("manifests")


def models_dir() -> Path:
    """Local model weights and embedding-model caches."""
    return _subdir("models")
