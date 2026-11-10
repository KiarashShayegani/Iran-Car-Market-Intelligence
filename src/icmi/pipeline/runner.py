"""Unified pipeline runner — ONE implementation, three execution modes.

Manual CLI, GitHub Actions workflows, and (future, disabled) scheduler
wrappers all invoke this runner; there is no separate collection/ingestion
implementation anywhere. Stages are idempotent, so interruption recovery is
"rerun the same command".

Stages:
    collect    per-platform collectors → raw runs + manifests (network)
    ingest     raw runs → canonical SQLite (cross-run dedup + price history)
    export     analytical Parquet snapshots
    documents  RAG documents
    index      vector index rebuild
    all        ingest → export → documents → index (no collection)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from icmi.core.config import load_sources
from icmi.core.locking import FileLock, LockHeldError
from icmi.core.paths import data_dir, repo_root
from icmi.pipeline.analytical import export_analytics
from icmi.pipeline.etl import ingest_run
from icmi.pipeline.store import CanonicalStore


@dataclass
class StageResult:
    name: str
    ok: bool
    detail: str = ""
    skipped: bool = False


@dataclass
class PipelineReport:
    mode: str
    started_at: str
    finished_at: str | None = None
    stages: list[StageResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.stages)

    def summary_lines(self) -> list[str]:
        lines = [
            f"pipeline mode={self.mode} started={self.started_at} ok={self.ok}"
        ]
        for s in self.stages:
            marker = "✓" if s.ok else "✗"
            skip = " (skipped)" if s.skipped else ""
            lines.append(f"  {marker} {s.name}{skip}: {s.detail}")
        return lines


class PipelineRunner:
    """The single pipeline implementation behind all execution modes."""

    def __init__(self, mode: str = "manual", config_path: Path | None = None) -> None:
        self.mode = mode
        self.sources = load_sources(config_path)
        self.report = PipelineReport(
            mode=mode,
            started_at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
        )
        self._lock_path = data_dir() / ".pipeline.lock"

    # --- stages ---------------------------------------------------------------

    def collect(
        self,
        platforms: list[str],
        max_pages: int,
        target_listings: int,
    ) -> StageResult:
        import subprocess
        import sys

        outcomes = []
        for platform in platforms:
            result = subprocess.run(
                [
                    sys.executable,
                    str(repo_root() / "scripts" / "collect.py"),
                    "--platform", platform,
                    "--mode", self.mode,
                    "--max-pages", str(max_pages),
                    "--target-listings", str(target_listings),
                ],
                cwd=repo_root(),
            )
            outcomes.append(f"{platform}=exit{result.returncode}")
        ok = all("=exit0" in o for o in outcomes)
        return StageResult("collect", ok, " ".join(outcomes))

    def ingest(self, platforms: list[str] | None = None) -> StageResult:
        store = CanonicalStore()
        try:
            raw_root = raw_dir_root()
            run_ids = sorted(
                d.name
                for d in raw_root.iterdir()
                if d.is_dir() and any(d.glob("*/records.jsonl"))
            ) if raw_root.is_dir() else []
            new_total = updated_total = obs_total = skipped = 0
            for run_id in run_ids:
                for platform_dir in sorted((raw_root / run_id).iterdir()):
                    if not platform_dir.is_dir():
                        continue
                    platform = platform_dir.name
                    if platform not in self.sources.sources:
                        continue
                    if platforms and platform not in platforms:
                        continue
                    stats = ingest_run(
                        run_id=run_id,
                        platform=platform,
                        store=store,
                        source=self.sources.sources[platform],
                        pipeline=self.sources.pipeline,
                    )
                    if stats.skipped_runs:
                        skipped += 1
                        continue
                    new_total += stats.new_listings
                    updated_total += stats.updated_listings
                    obs_total += stats.price_observations_appended
            return StageResult(
                "ingest",
                True,
                f"new={new_total} updated={updated_total} "
                f"observations+={obs_total} runs_skipped={skipped}",
            )
        finally:
            store.close()

    def export(self) -> StageResult:
        store = CanonicalStore()
        try:
            result = export_analytics(store)
        finally:
            store.close()
        return StageResult(
            "export",
            True,
            f"{result['listings_rows']} listings, {result['observation_rows']} observations",
        )

    def documents(self) -> StageResult:
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, str(repo_root() / "scripts" / "build_documents.py")],
            cwd=repo_root(),
            capture_output=True,
            text=True,
        )
        detail = (result.stdout or result.stderr).strip().splitlines()[-1] if (
            result.stdout or result.stderr
        ) else ""
        return StageResult("documents", result.returncode == 0, detail)

    def index(self) -> StageResult:
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, str(repo_root() / "scripts" / "build_index.py")],
            cwd=repo_root(),
            capture_output=True,
            text=True,
        )
        detail = (result.stdout or result.stderr).strip().splitlines()[-1] if (
            result.stdout or result.stderr
        ) else ""
        return StageResult("index", result.returncode == 0, detail)

    # --- orchestration ---------------------------------------------------------

    def run(
        self,
        stages: list[str],
        platforms: list[str] | None = None,
        max_pages: int = 2,
        target_listings: int = 48,
    ) -> PipelineReport:
        # fresh report per run() call — the runner instance is reusable
        self.report = PipelineReport(
            mode=self.mode,
            started_at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
        )
        lock = FileLock(self._lock_path, holder=f"pipeline:{self.mode}")
        try:
            lock.acquire()
        except LockHeldError as exc:
            self.report.stages.append(StageResult("lock", False, str(exc)))
            self.report.finished_at = datetime.now(tz=UTC).isoformat(timespec="seconds")
            return self.report

        try:
            for stage in stages:
                try:
                    if stage == "collect":
                        self.report.stages.append(
                            self.collect(
                                platforms or list(self.sources.enabled_sources()),
                                max_pages,
                                target_listings,
                            )
                        )
                    elif stage == "ingest":
                        self.report.stages.append(self.ingest(platforms))
                    elif stage == "export":
                        self.report.stages.append(self.export())
                    elif stage == "documents":
                        self.report.stages.append(self.documents())
                    elif stage == "index":
                        self.report.stages.append(self.index())
                    else:
                        self.report.stages.append(
                            StageResult(stage, False, f"unknown stage: {stage}")
                        )
                except Exception as exc:  # noqa: BLE001 — one stage failing must
                    # not corrupt the report; idempotence makes rerun safe.
                    self.report.stages.append(
                        StageResult(stage, False, f"{type(exc).__name__}: {exc}")
                    )
            self.report.finished_at = datetime.now(tz=UTC).isoformat(timespec="seconds")
            return self.report
        finally:
            lock.release()


def raw_dir_root() -> Path:
    from icmi.core.paths import raw_dir

    return raw_dir()
