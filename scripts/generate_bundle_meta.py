#!/usr/bin/env python3
"""Generate bundle-meta.json for a v2 ingestion bundle.

Runs in the publish job AFTER raw/ + manifests/ are staged into
data/g4-bundle and BEFORE SHA256SUMS is written, so the metadata itself is
checksum-covered. Describes the artifact sufficiently for the local
importer to validate it without guessing: contract identity, the Actions
run that produced it, per-platform collection run ids and record counts,
manifest references, and the extraction fingerprint. Manifests remain the
authoritative per-run record — this file only cross-references them.

Usage:
    python scripts/generate_bundle_meta.py [--bundle-dir data/g4-bundle] \\
        [--staging-dir data] [--actions-run-id N] [--workflow-name W] [--workflow-ref R]

CI (h-ops-pipeline publish step) calls it with no flags: --actions-run-id
and friends default to the GITHUB_* environment; staging dirs default to
the repository layout used by the publish step.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

CONTRACT = "icmi-ingestion-bundle"
CONTRACT_VERSION = 2


def _collect_platforms(manifest_dir: Path) -> tuple[dict, list[str], str | None]:
    """Map platform -> {status, run_ids, records_extracted} from manifests."""
    platforms: dict[str, dict] = {}
    refs: list[str] = []
    extraction: str | None = None
    for mf in sorted(manifest_dir.glob("*.json")):
        if mf.name == "summary.json" or mf.name.startswith("actions-run-"):
            continue
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
        except ValueError:
            print(f"WARNING: corrupt manifest skipped: {mf.name}", file=sys.stderr)
            continue
        platform = m.get("platform")
        run_id = m.get("run_id")
        if not platform or not run_id:
            continue
        entry = platforms.setdefault(
            platform, {"status": m.get("status"), "run_ids": [], "records_extracted": 0}
        )
        entry["run_ids"].append(run_id)
        entry["records_extracted"] += int(m.get("records", {}).get("extracted", 0) or 0)
        # worst status wins (failed < partial < success) so the meta is honest
        rank = {"failed": 0, "partial": 1, "success": 2}
        if rank.get(m.get("status"), 1) < rank.get(entry["status"], 1):
            entry["status"] = m.get("status")
        extraction = extraction or m.get("extraction_version")
        refs.append(f"metrics/{mf.name}")
    return platforms, refs, extraction


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle-dir", default="data/g4-bundle")
    parser.add_argument(
        "--staging-dir", default="data",
        help="workspace root holding raw/ and manifests/ to reference",
    )
    parser.add_argument("--actions-run-id", default=os.environ.get("GITHUB_RUN_ID"))
    parser.add_argument("--actions-run-attempt", default=os.environ.get("GITHUB_RUN_ATTEMPT", "1"))
    parser.add_argument("--workflow-name", default=os.environ.get("GITHUB_WORKFLOW"))
    parser.add_argument("--workflow-ref", default=os.environ.get("GITHUB_WORKFLOW_REF"))
    args = parser.parse_args(argv)

    staging = Path(args.staging_dir)
    bundle = Path(args.bundle_dir)
    bundle.mkdir(parents=True, exist_ok=True)

    # platforms come from the staged manifests (already copied into the bundle);
    # raw run dirs come from the staged raw tree — both describe THIS run only,
    # because the publish job workspace contains just the downloaded artifacts.
    manifest_src = bundle / "metrics"
    if not manifest_src.is_dir():
        manifest_src = staging / "manifests"
    platforms, manifest_refs, extraction = _collect_platforms(manifest_src)

    raw_src = bundle / "raw"
    if not raw_src.is_dir():
        raw_src = staging / "raw"
    raw_run_ids = (
        sorted(d.name for d in raw_src.iterdir() if d.is_dir())
        if raw_src.is_dir() else []
    )

    meta = {
        "contract": CONTRACT,
        "contract_version": CONTRACT_VERSION,
        "actions_run_id": int(args.actions_run_id) if args.actions_run_id else None,
        "actions_run_attempt": int(args.actions_run_attempt or 1),
        "workflow_name": args.workflow_name,
        "workflow_ref": args.workflow_ref,
        "created_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "extraction_version": extraction,
        "platforms": platforms,
        "raw_run_ids": raw_run_ids,
        "manifest_refs": manifest_refs,
        "sha256sums_file": "SHA256SUMS",
    }
    out = bundle / "bundle-meta.json"
    out.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"bundle-meta: {out} | platforms={sorted(platforms)} raw_runs={raw_run_ids}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
