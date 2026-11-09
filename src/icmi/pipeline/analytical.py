"""Analytical exports: Parquet snapshots of the canonical layer.

The canonical SQLite database remains the single authoritative store; these
Parquet files are derived, read-only snapshots for analytics and for the
app's fast reads. They carry the store's extraction state at export time and
are regenerated cheaply — never hand-edited, never authoritative.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from icmi.core.paths import analytical_dir
from icmi.pipeline.store import CanonicalStore


def export_analytics(store: CanonicalStore, out_dir: Path | None = None) -> dict[str, Path]:
    """Write listings + price-observation snapshots. Returns written paths."""
    out = out_dir or analytical_dir()
    out.mkdir(parents=True, exist_ok=True)

    listings = store.query_df("SELECT * FROM listings ORDER BY listing_key")
    listings_path = out / "listings_latest.parquet"
    listings.to_parquet(listings_path, index=False)

    observations = store.query_df(
        "SELECT * FROM price_observations ORDER BY listing_key, id"
    )
    observations_path = out / "price_observations_latest.parquet"
    observations.to_parquet(observations_path, index=False)

    return {
        "listings": listings_path,
        "price_observations": observations_path,
        "listings_rows": len(listings),  # type: ignore[dict-item]
        "observation_rows": len(observations),  # type: ignore[dict-item]
    }


def snapshot_stamp() -> str:
    """UTC stamp used for snapshot directory naming."""
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S")
