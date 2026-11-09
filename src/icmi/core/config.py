"""Configuration loading for ICMI.

Sources are fully defined in config/sources.yaml. The loader validates the
document's shape (not every value — collectors own per-source validation) and
exposes typed access. Configuration changes are the only supported way to add
platforms or alter endpoints; there are no hardcoded sources in code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

SUPPORTED_SCHEMA_VERSION = 1


class PipelineConfig(BaseModel):
    extraction_version: str
    request_timeout_seconds: float = Field(gt=0)
    request_delay_seconds: float = Field(ge=0)
    request_jitter_seconds: float = Field(ge=0)
    max_retries: int = Field(ge=0)
    user_agent: str


class PaginationConfig(BaseModel):
    type: str
    page_param: str | None = None
    page_size: int | None = None
    page_size_param: str | None = None


class CollectionConfig(BaseModel):
    """A named parameter preset for one source (e.g. the unfiltered feed now,
    brand-targeted queries later). Adding breadth is a config entry."""

    description: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class SourceConfig(BaseModel):
    display_name: str
    display_name_fa: str
    enabled: bool
    status: str
    base_url: str
    pagination: PaginationConfig | None = None
    notes: str | None = None
    field_support: list[str] = Field(default_factory=list)
    collections: dict[str, CollectionConfig] = Field(default_factory=dict)

    def collection_params(self, name: str) -> dict[str, Any]:
        """Params for a named collection preset ('full' by default)."""
        if not self.collections:
            if name == "full":
                return {}
            raise KeyError(f"Source has no collection presets; '{name}' unknown")
        if name not in self.collections:
            available = ", ".join(sorted(self.collections))
            raise KeyError(f"Unknown collection '{name}' (available: {available})")
        return dict(self.collections[name].params)


class SourcesConfig(BaseModel):
    schema_version: int
    pipeline: PipelineConfig
    sources: dict[str, SourceConfig]

    def enabled_sources(self) -> dict[str, SourceConfig]:
        return {name: src for name, src in self.sources.items() if src.enabled}


def default_config_path() -> Path:
    """Path to sources.yaml relative to the repository root."""
    return Path(__file__).resolve().parents[3] / "config" / "sources.yaml"


def load_sources(path: Path | None = None) -> SourcesConfig:
    """Load and validate config/sources.yaml.

    ICMI_CONFIG may point at an alternative config file (used by tests and
    experimental runs). Raises a descriptive error if the file is missing,
    malformed, or written for a different schema version.
    """
    if path is None:
        import os

        env_path = os.environ.get("ICMI_CONFIG")
        path = Path(env_path) if env_path else default_config_path()
    if not path.is_file():
        raise FileNotFoundError(f"ICMI config not found: {path}")

    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"ICMI config must be a YAML mapping: {path}")

    try:
        config = SourcesConfig.model_validate(raw)
    except ValidationError as exc:
        problems = "\n".join(
            f"  - {'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
            for err in exc.errors(include_url=False)
        )
        raise ValueError(f"Invalid ICMI config at {path}:\n{problems}") from None

    if config.schema_version != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"Config schema_version {config.schema_version} is not supported "
            f"(expected {SUPPORTED_SCHEMA_VERSION}); update icmi or the config."
        )
    return config
