"""Data Layer persistence for collection runs.

Layout per run (under ``data/raw/<run_id>/<platform>/``):

    pages/page_NNNN.json   verbatim API page responses (immutable)
    records.jsonl          one line per ad: {source_listing_id, raw_payload}

Manifests go to ``data/manifests/<run_id>.json``. Raw data is write-once:
existing files are never overwritten, and a collection failure never damages
a previously published dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

from icmi.core.paths import manifests_dir, raw_dir


class RawWriter:
    """Writes raw artifacts for one (run_id, platform) collection run."""

    def __init__(self, run_id: str, platform: str) -> None:
        self.run_id = run_id
        self.platform = platform
        self._dir = raw_dir() / run_id / platform
        self._pages_dir = self._dir / "pages"
        self._pages_dir.mkdir(parents=True, exist_ok=True)
        self._records_path = self._dir / "records.jsonl"

    @property
    def raw_dir_path(self) -> Path:
        return self._dir

    def write_page(self, page_index: int, payload: dict) -> Path:
        """Persist one verbatim API page response."""
        path = self._pages_dir / f"page_{page_index:04d}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        return path

    def write_records(self, raw_entries: list[dict]) -> Path:
        """Persist raw ad payloads, one JSON object per line, keyed by listing id."""
        with open(self._records_path, "w", encoding="utf-8") as fh:
            for entry in raw_entries:
                fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        return self._records_path

    def write_manifest(self, manifest_payload: dict) -> Path:
        path = manifests_dir() / f"{self.run_id}.json"
        path.write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=1, default=str),
            encoding="utf-8",
        )
        return path
