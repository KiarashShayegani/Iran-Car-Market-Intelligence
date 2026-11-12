"""v2 ingestion-bundle validation: synthetic-bundle unit tests.

Covers the importer's rejection behavior for every contract violation the
verification protocol requires: missing/malformed bundle-meta, contract
mismatch, run-id mismatch, platform/manifest mismatch, record-count
mismatch, missing raw runs, checksum corruption, unclaimed raw runs,
unexpected files (warning), and extraction-version mismatch (warning-only
gate). Valid bundles must verify with zero errors.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

_LOCAL_VERSION = "icmi-v3-extraction-1"

_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "sync_actions_artifact.py"
)
_spec = importlib.util.spec_from_file_location("sync_actions_artifact", _SCRIPT)
sync_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync_mod)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _manifest(run_id: str, platform: str, extracted: int, version: str) -> dict:
    return {
        "platform": platform,
        "status": "success",
        "run_id": run_id,
        "extraction_version": version,
        "records": {"extracted": extracted, "duplicates_within_run": 0},
    }


def _rehash(bundle: Path) -> None:
    """Regenerate SHA256SUMS over everything currently in the bundle."""
    files = sorted(
        p for p in bundle.rglob("*")
        if p.is_file() and p.name != "SHA256SUMS"
    )
    lines = [
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(bundle)}"
        for p in files
    ]
    _write(bundle / "SHA256SUMS", "\n".join(lines) + "\n")


def _build_bundle(
    root: Path,
    platforms: dict[str, int],
    *,
    actions_run_id: int = 1000,
    extraction: str = _LOCAL_VERSION,
    with_meta: bool = True,
) -> Path:
    """Build a minimal but contract-complete synthetic ingestion bundle."""
    bundle = root / "bundle"
    bundle.mkdir(parents=True)
    meta_platforms: dict[str, dict] = {}
    refs: list[str] = []
    for platform, n in platforms.items():
        run_id = f"20261113T000000-{platform}-ci"
        recs = "\n".join(
            json.dumps({"source_listing_id": str(i), "raw_payload": {"p": i}})
            for i in range(n)
        )
        _write(
            bundle / "raw" / run_id / platform / "records.jsonl",
            (recs + "\n") if recs else "",
        )
        _write(
            bundle / "metrics" / f"{run_id}.json",
            json.dumps(_manifest(run_id, platform, n, extraction)),
        )
        meta_platforms[platform] = {
            "status": "success",
            "run_ids": [run_id],
            "records_extracted": n,
        }
        refs.append(f"metrics/{run_id}.json")
    if with_meta:
        meta = {
            "contract": "icmi-ingestion-bundle",
            "contract_version": 2,
            "actions_run_id": actions_run_id,
            "actions_run_attempt": 1,
            "created_at": "2026-11-13T00:00:00+00:00",
            "extraction_version": extraction,
            "platforms": meta_platforms,
            "raw_run_ids": sorted(
                d.name for d in (bundle / "raw").iterdir() if d.is_dir()
            ),
            "manifest_refs": refs,
            "sha256sums_file": "SHA256SUMS",
        }
        _write(bundle / "bundle-meta.json", json.dumps(meta))
    _rehash(bundle)
    return bundle


def test_valid_bundle_verifies_clean(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path, {"sheypoor": 3, "hamrah_mechanic": 2})
    report, mismatch = sync_mod._verify_ingestion(bundle, 1000)
    assert report["errors"] == []
    assert mismatch is False
    assert report["raw_runs"] == [
        "20261113T000000-hamrah_mechanic-ci", "20261113T000000-sheypoor-ci",
    ]
    assert report["bundle_meta"]["contract_version"] == 2


def test_missing_bundle_meta_rejected(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path, {"sheypoor": 2}, with_meta=False)
    report, _ = sync_mod._verify_ingestion(bundle, 1000)
    assert any("bundle-meta.json" in e for e in report["errors"])


def test_malformed_bundle_meta_rejected(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path, {"sheypoor": 2})
    _write(bundle / "bundle-meta.json", "{not json")
    _rehash(bundle)
    report, _ = sync_mod._verify_ingestion(bundle, 1000)
    assert any("malformed bundle-meta.json" in e for e in report["errors"])


def test_contract_identity_mismatch_rejected(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path, {"sheypoor": 2})
    meta = json.loads((bundle / "bundle-meta.json").read_text())
    meta["contract"] = "icmi-snapshot"
    _write(bundle / "bundle-meta.json", json.dumps(meta))
    _rehash(bundle)
    report, _ = sync_mod._verify_ingestion(bundle, 1000)
    assert any("contract mismatch" in e for e in report["errors"])


def test_actions_run_id_mismatch_rejected(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path, {"sheypoor": 2}, actions_run_id=1000)
    report, _ = sync_mod._verify_ingestion(bundle, 9999)
    assert any("actions_run_id" in e for e in report["errors"])


def test_platform_manifest_mismatch_rejected(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path, {"sheypoor": 2})
    run_id = "20261113T000000-sheypoor-ci"
    m = _manifest(run_id, "sheypoor", 2, _LOCAL_VERSION)
    m["platform"] = "hamrah_mechanic"
    _write(bundle / "metrics" / f"{run_id}.json", json.dumps(m))
    _rehash(bundle)
    report, _ = sync_mod._verify_ingestion(bundle, 1000)
    assert any(
        "manifest platform mismatch" in e for e in report["errors"]
    )


def test_unclaimed_raw_run_rejected(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path, {"sheypoor": 2})
    # add a raw run dir that bundle-meta does not claim
    _write(bundle / "raw" / "20261113T000000-bama-ci" / "bama" / "records.jsonl", "")
    _rehash(bundle)
    report, _ = sync_mod._verify_ingestion(bundle, 1000)
    assert any(
        "raw run 20261113T000000-bama-ci present but not claimed" in e
        for e in report["errors"]
    )


def test_claimed_run_with_missing_raw_rejected(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path, {"sheypoor": 2, "hamrah_mechanic": 2})
    # claim a run in meta whose raw dir does not exist
    meta = json.loads((bundle / "bundle-meta.json").read_text())
    meta["platforms"]["bama"] = {
        "status": "success",
        "run_ids": ["20261113T000000-bama-ci"],
        "records_extracted": 5,
    }
    _write(bundle / "bundle-meta.json", json.dumps(meta))
    _rehash(bundle)
    report, _ = sync_mod._verify_ingestion(bundle, 1000)
    assert any(
        "claims run 20261113T000000-bama-ci but raw/" in e
        for e in report["errors"]
    )


def test_record_count_mismatch_rejected(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path, {"sheypoor": 2})
    meta = json.loads((bundle / "bundle-meta.json").read_text())
    meta["platforms"]["sheypoor"]["records_extracted"] = 99
    _write(bundle / "bundle-meta.json", json.dumps(meta))
    _rehash(bundle)
    report, _ = sync_mod._verify_ingestion(bundle, 1000)
    assert any("record-count mismatch" in e for e in report["errors"])


def test_incorrect_checksums_rejected(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path, {"sheypoor": 2})
    # corrupt a payload file AFTER checksums were written
    rec = bundle / "raw" / "20261113T000000-sheypoor-ci" / "sheypoor" / "records.jsonl"
    rec.write_text(rec.read_text() + json.dumps(
        {"source_listing_id": "injected", "raw_payload": {}}) + "\n",
        encoding="utf-8",
    )
    report, _ = sync_mod._verify_ingestion(bundle, 1000)
    assert any(
        "checksum mismatch" in e for e in report["errors"]
    )


def test_unexpected_files_are_warnings_only(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path, {"sheypoor": 2})
    _write(bundle / "metrics" / "stray.json", "{}")
    report, _ = sync_mod._verify_ingestion(bundle, 1000)
    assert report["errors"] == []
    assert any("stray.json" in w for w in report["warnings"])


def test_extraction_version_mismatch_is_warning_gate(tmp_path: Path) -> None:
    bundle = _build_bundle(
        tmp_path, {"sheypoor": 2}, extraction="icmi-v99-extraction-1",
    )
    report, mismatch = sync_mod._verify_ingestion(bundle, 1000)
    # intra-bundle consistency holds; the LOCAL gate flags, raw still imports
    assert report["errors"] == []
    assert mismatch is True
    assert "icmi-v99-extraction-1" in report["extraction_version_mismatch"]


def test_meta_extraction_fingerprint_inconsistency_rejected(
    tmp_path: Path,
) -> None:
    bundle = _build_bundle(tmp_path, {"sheypoor": 2})
    meta = json.loads((bundle / "bundle-meta.json").read_text())
    meta["extraction_version"] = "icmi-v99-extraction-1"
    _write(bundle / "bundle-meta.json", json.dumps(meta))
    _rehash(bundle)
    report, _ = sync_mod._verify_ingestion(bundle, 1000)
    assert any(
        "bundle-meta extraction_version" in e for e in report["errors"]
    )


def test_missing_manifest_ref_rejected(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path, {"sheypoor": 2})
    meta = json.loads((bundle / "bundle-meta.json").read_text())
    meta["manifest_refs"].append("metrics/20261113T999999-ghost-ci.json")
    _write(bundle / "bundle-meta.json", json.dumps(meta))
    _rehash(bundle)
    report, _ = sync_mod._verify_ingestion(bundle, 1000)
    assert any("manifest_ref missing" in e for e in report["errors"])


def test_failed_platform_bundle_verifies_with_warning(tmp_path: Path) -> None:
    """A failed platform still ships its manifest; bundle must be importable."""
    bundle = _build_bundle(tmp_path, {"sheypoor": 2})
    run_id = "20261113T000000-sheypoor-ci"
    m = _manifest(run_id, "sheypoor", 2, _LOCAL_VERSION)
    m["status"] = "failed"
    m["records"] = {"extracted": 0, "duplicates_within_run": 0}
    _write(bundle / "metrics" / f"{run_id}.json", json.dumps(m))
    meta = json.loads((bundle / "bundle-meta.json").read_text())
    meta["platforms"]["sheypoor"]["status"] = "failed"
    meta["platforms"]["sheypoor"]["records_extracted"] = 0
    # failed platform ships pages but no records.jsonl
    rec = bundle / "raw" / run_id / "sheypoor" / "records.jsonl"
    rec.unlink()
    _rehash(bundle)
    report, _ = sync_mod._verify_ingestion(bundle, 1000)
    assert report["errors"] == []
    assert any("status=failed" in w for w in report["warnings"])


def test_missing_checksums_rejected(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path, {"sheypoor": 2})
    (bundle / "SHA256SUMS").unlink()
    report, _ = sync_mod._verify_ingestion(bundle, 1000)
    assert any("missing SHA256SUMS" in e for e in report["errors"])
    # the early return must not have produced other misleading errors
    assert len(report["errors"]) == 1


def test_all_platforms_failed_empty_bundle_warns(tmp_path: Path) -> None:
    """All platforms failed/skipped → empty raw tree must not be an error."""
    bundle = _build_bundle(tmp_path, {"sheypoor": 1})
    import shutil
    shutil.rmtree(bundle / "raw")
    meta = json.loads((bundle / "bundle-meta.json").read_text())
    meta["platforms"] = {}
    meta["raw_run_ids"] = []
    meta["manifest_refs"] = []
    _write(bundle / "bundle-meta.json", json.dumps(meta))
    _rehash(bundle)
    report, _ = sync_mod._verify_ingestion(bundle, 1000)
    assert report["errors"] == []
    assert any("no raw runs in bundle" in w for w in report["warnings"])


def test_raw_run_without_manifest_rejected(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path, {"sheypoor": 2})
    (bundle / "metrics" / "20261113T000000-sheypoor-ci.json").unlink()
    _rehash(bundle)
    report, _ = sync_mod._verify_ingestion(bundle, 1000)
    assert any(
        "raw run without manifest: 20261113T000000-sheypoor-ci" in e
        for e in report["errors"]
    )
