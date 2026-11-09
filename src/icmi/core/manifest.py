"""Run manifests and quality reports.

Every collection run produces one manifest — the operational record of what
happened, whether the run succeeded, where outputs live, and how well P0/P1
fields were covered. Manifests are the only collection artifacts that enter
git (datasets never do), giving the repository a visible collection history.

The quality report encodes the three missing-data states per field:

- ``not_supported_by_platform`` — field absent from the source's
  ``field_support`` declaration in config/sources.yaml (platform-level fact,
  reported once per run, not per record);
- ``missing_in_ad`` — supported but null on a record, with no extraction
  error;
- ``extraction_failed`` — extraction raised for that field on that record
  (``extraction_errors``), never silently treated as missing-in-ad.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from icmi.core.records import P0_FIELDS, P1_FIELDS, CanonicalRecord, has_value


class FieldQuality(BaseModel):
    """Per-field population statistics for one run."""

    field: str
    priority: str  # "P0" | "P1"
    supported_by_platform: bool
    populated: int
    missing_in_ad: int
    extraction_failed: int
    coverage_pct: float  # populated / total records (0 if unsupported)

    @property
    def missing_state(self) -> str:
        if not self.supported_by_platform:
            return "not_supported_by_platform"
        if self.extraction_failed:
            return "extraction_failed"
        return "missing_in_ad" if self.missing_in_ad else "populated"


class QualityReport(BaseModel):
    record_count: int
    p0_coverage_pct: float
    p1_coverage_pct: float
    fields: list[FieldQuality]

    @classmethod
    def build(
        cls,
        records: list[CanonicalRecord],
        field_support: list[str],
    ) -> QualityReport:
        """Compute per-field quality from extracted records.

        ``field_support`` lists the canonical fields the platform is known to
        be able to provide (from config/sources.yaml). Fields outside the list
        are reported as ``not_supported_by_platform`` even if incidentally
        populated.
        """
        total = len(records)
        fields: list[FieldQuality] = []
        for priority, names in (("P0", P0_FIELDS), ("P1", P1_FIELDS)):
            for name in names:
                supported = name in field_support
                populated = sum(1 for r in records if has_value(getattr(r, name)))
                failed = sum(1 for r in records if name in r.extraction_errors)
                missing_in_ad = (
                    max(total - populated - failed, 0) if supported else 0
                )
                fields.append(
                    FieldQuality(
                        field=name,
                        priority=priority,
                        supported_by_platform=supported,
                        populated=populated,
                        missing_in_ad=missing_in_ad,
                        extraction_failed=failed,
                        coverage_pct=round(100.0 * populated / total, 1) if total else 0.0,
                    )
                )
        p0 = [f for f in fields if f.priority == "P0"]
        p1 = [f for f in fields if f.priority == "P1"]
        return cls(
            record_count=total,
            p0_coverage_pct=round(
                sum(f.coverage_pct for f in p0) / len(p0), 1
            ) if p0 else 0.0,
            p1_coverage_pct=round(
                sum(f.coverage_pct for f in p1) / len(p1), 1
            ) if p1 else 0.0,
            fields=fields,
        )


class RunManifest(BaseModel):
    """Operational record of one collection run."""

    run_id: str
    platform: str
    execution_mode: str  # "local" | "ci"
    status: str  # "success" | "partial" | "failed"
    started_at: datetime
    completed_at: datetime
    extraction_version: str
    config_schema_version: int

    pages: dict[str, int] = Field(default_factory=dict)
    # e.g. {"requested": 3, "fetched": 3, "failed": 0, "empty": 0, "parse_failed": 0}

    records: dict[str, int] = Field(default_factory=dict)
    # e.g. {"received": 36, "extracted": 36, "duplicates_within_run": 0}

    skip_reasons: dict[str, int] = Field(default_factory=dict)
    failed_requests: list[str] = Field(default_factory=list)
    quality: QualityReport
    outputs: dict[str, str] = Field(default_factory=dict)  # artifact -> path
    notes: str | None = None

    def summary_lines(self) -> list[str]:
        """Human-readable summary for CLI output and reports."""
        lines = [
            f"run_id={self.run_id} platform={self.platform} mode={self.execution_mode}",
            f"status={self.status} pages={self.pages} records={self.records}",
            f"P0 coverage: {self.quality.p0_coverage_pct}%  "
            f"P1 coverage: {self.quality.p1_coverage_pct}%",
        ]
        problem_fields = [
            f"{f.field}[{f.missing_state}:{f.missing_in_ad or f.extraction_failed}]"
            for f in self.quality.fields
            if f.priority == "P0" and f.missing_state != "populated"
        ]
        if problem_fields:
            lines.append("P0 attention: " + ", ".join(problem_fields))
        if self.skip_reasons:
            lines.append(f"skip reasons: {self.skip_reasons}")
        if self.failed_requests:
            lines.append(f"failed requests: {len(self.failed_requests)}")
        return lines
