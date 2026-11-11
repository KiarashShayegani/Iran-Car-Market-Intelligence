"""Collection policy: per-source operational budgets from config/collection_policy.yaml.

The policy is the single source of truth for how much gets collected per run
and how. The pipeline runner validates and applies it; a disabled source
must never run. Schedule enablement is a separate owner gate (H-ops-5).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError

_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "collection_policy.yaml"


class SourcePolicy(BaseModel):
    enabled: bool = True
    cadence: str = "daily"
    max_listings_per_run: int = Field(default=60, gt=0)
    max_pages_per_run: int = Field(default=3, gt=0)
    max_requests_per_run: int = Field(default=10, gt=0)
    max_pages_per_preset: int | None = None
    presets_per_run: int | None = None
    rotation: str | None = None
    presets: list[str] = Field(default_factory=list)
    gated: str | None = None


class PolicyDefaults(BaseModel):
    request_delay_seconds: float = 2.0
    request_jitter_seconds: float = 1.0
    max_retries: int = 3
    backoff_multiplier: int = 2
    timeout_seconds: int = 30
    max_execution_minutes: int = 20
    stale_page_streak: int = 3


class CollectionPolicy(BaseModel):
    schema_version: int
    defaults: PolicyDefaults
    sources: dict[str, SourcePolicy]

    def enabled_sources(self) -> list[str]:
        return [name for name, src in self.sources.items() if src.enabled]

    def assert_can_collect(self, platform: str) -> SourcePolicy:
        policy = self.sources.get(platform)
        if policy is None:
            raise ValueError(f"unknown platform in policy: '{platform}'")
        if not policy.enabled:
            raise ValueError(f"platform '{platform}' is disabled in the collection policy")
        if policy.gated:
            raise ValueError(
                f"platform '{platform}' is gated: {policy.gated} — owner approval required"
            )
        return policy

    def rotation_for(self, platform: str, run_index: int) -> list[str]:
        """Deterministic preset rotation: returns the presets for this run.

        run_index should advance with each collection run (e.g. the count of
        previous runs for the platform, or days since start). Rotation is
        stable for the same run_index.
        """
        policy = self.sources.get(platform)
        if policy is None or not policy.presets:
            return []
        per_run = policy.presets_per_run or len(policy.presets)
        n = len(policy.presets)
        start = (run_index * per_run) % n
        ordered = policy.presets[start:] + policy.presets[:start]
        return ordered[:per_run]


@lru_cache(maxsize=1)
def load_policy(path: str | None = None) -> CollectionPolicy:
    p = Path(path) if path else _DEFAULT_PATH
    if not p.is_file():
        raise FileNotFoundError(f"collection policy not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        return CollectionPolicy.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"invalid collection policy at {p}:\n{exc}") from None
