#!/usr/bin/env python3
"""One-command sync from GitHub Actions artifacts into the local data layer.

Supports both artifact contracts:

- v2 ingestion bundle (``icmi-ingestion-<run_id>``): raw/ + metrics/ +
  SHA256SUMS, no runner DB. Verified (zip CRC + traversal, SHA256SUMS,
  manifest-per-run, extraction-version fingerprint), installed write-once
  into data/raw + data/manifests, then ingested through the standard local
  path (run-level skip + record-level idempotence). A shrink-guard restores
  the pre-import backup if the authoritative DB ever loses rows, and an
  idempotence probe re-imports the same bundle expecting zero changes.
- v1 snapshot (``icmi-snapshot*``): legacy collect-etl runs; verified and
  imported via sync_latest_data.py as before.

Requires GITHUB_TOKEN (Actions: read) — already in the repo .env.

Examples:
    python scripts/sync_actions_artifact.py --latest
    python scripts/sync_actions_artifact.py --latest --dry-run
    python scripts/sync_actions_artifact.py --run-id 43128756244
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import requests

GITHUB_API = "https://api.github.com"
REPO = "KiarashShayegani/Iran-Car-Market-Intelligence"
EXPECTED_WORKFLOWS = ("h-ops-pipeline.yml", "collect-etl.yml")
INGESTION_PREFIX = "icmi-ingestion-"
SNAPSHOT_PREFIX = "icmi-snapshot"


def _is_expected_run(run: dict) -> bool:
    return any(run["path"].endswith(w) for w in EXPECTED_WORKFLOWS)


def _import_ledger_path(actions_run_id: int) -> Path:
    from icmi.core.paths import manifests_dir

    return manifests_dir() / f"actions-run-{actions_run_id}.json"


def _already_imported(actions_run_id: int) -> bool:
    return _import_ledger_path(actions_run_id).is_file()


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}


def _find_run(token: str, run_id: int | None) -> dict:
    if run_id:
        run = requests.get(
            f"{GITHUB_API}/repos/{REPO}/actions/runs/{run_id}",
            headers=_headers(token), timeout=30,
        ).json()
    else:
        runs = requests.get(
            f"{GITHUB_API}/repos/{REPO}/actions/runs",
            headers=_headers(token),
            params={"status": "success", "per_page": 20}, timeout=30,
        ).json()
        run = next(
            (r for r in runs.get("workflow_runs", []) if _is_expected_run(r)),
            None,
        )
    if not run or run.get("id") is None:
        raise SystemExit("no successful pipeline Actions run found")
    if run.get("conclusion") != "success":
        raise SystemExit(f"run {run['id']} conclusion={run.get('conclusion')} — refusing to import")
    return run


def _validate_members(zf: zipfile.ZipFile, root: Path) -> None:
    """Defense-in-depth over zipfile's built-in sanitization: reject absolute
    paths, '..' traversal, and members escaping the extraction root."""
    root_resolved = root.resolve()
    for name in zf.namelist():
        if name.startswith("/"):
            raise SystemExit(f"unsafe absolute zip member: {name}")
        parts = Path(name).parts
        if ".." in parts:
            raise SystemExit(f"zip traversal attempt: {name}")
        if not (root / name).resolve().is_relative_to(root_resolved):
            raise SystemExit(f"zip member escapes root: {name}")


def _pick_artifact(artifacts: dict) -> dict:
    """Prefer the v2 ingestion bundle; fall back to the legacy v1 snapshot."""
    arts = artifacts.get("artifacts", [])
    for a in arts:
        if a["name"].startswith(INGESTION_PREFIX):
            return a
    for a in arts:
        if a["name"].startswith(SNAPSHOT_PREFIX):
            return a
    raise SystemExit(
        "no icmi-ingestion (v2) or icmi-snapshot (v1) artifact on this run"
    )


def _download_artifact(token: str, run_id: int, workdir: Path) -> tuple[Path, str, dict]:
    """Download this run's data artifact. Returns (extract_root, kind, meta).

    kind is 'ingestion' (v2, flat zip) or 'snapshot' (v1, outer zip holding
    icmi_snapshot_*.zip).
    """
    arts = requests.get(
        f"{GITHUB_API}/repos/{REPO}/actions/runs/{run_id}/artifacts",
        headers=_headers(token), timeout=30,
    ).json()
    art = _pick_artifact(arts)
    t0 = time.time()
    r = requests.get(art["archive_download_url"], headers=_headers(token), timeout=300)
    if r.status_code != 200:
        raise SystemExit(
            f"artifact download failed: HTTP {r.status_code} for {art['name']} "
            f"(run {run_id}) — check token scope and artifact expiry"
        )
    duration = round(time.time() - t0, 1)
    artifact_zip = workdir / "artifact.zip"
    artifact_zip.write_bytes(r.content)
    with zipfile.ZipFile(artifact_zip) as zf:
        if zf.testzip() is not None:
            raise SystemExit("outer artifact zip is corrupt")
        _validate_members(zf, workdir)
        zf.extractall(workdir)

    meta = {
        "artifact_name": art["name"],
        "compressed_kb": len(r.content) // 1024,
        "download_seconds": duration,
    }
    if art["name"].startswith(INGESTION_PREFIX):
        return _bundle_root(workdir), "ingestion", meta
    inner = next(workdir.rglob("icmi_snapshot_*.zip"))
    snapshot_dir = workdir / inner.stem
    with zipfile.ZipFile(inner) as zf:
        if zf.testzip() is not None:
            raise SystemExit("inner snapshot zip is corrupt")
        _validate_members(zf, snapshot_dir)
        zf.extractall(snapshot_dir)
    meta["inner_zip"] = inner.name
    return snapshot_dir, "snapshot", meta


def _bundle_root(workdir: Path) -> Path:
    """The v2 bundle extracts flat (SHA256SUMS at top level) or under one dir."""
    if (workdir / "SHA256SUMS").is_file():
        return workdir
    candidates = [
        d for d in workdir.iterdir() if d.is_dir() and (d / "SHA256SUMS").is_file()
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise SystemExit("could not locate ingestion bundle root (SHA256SUMS missing)")


def _verify_ingestion(bundle_dir: Path, actions_run_id: int | None = None) -> tuple[dict, bool]:
    """Verify the v2 ingestion bundle. Returns (report, version_mismatch).

    Hard errors: missing/corrupt SHA256SUMS, any checksum mismatch, a raw run
    dir without its manifest, corrupt manifest/bundle-meta JSON, bundle-meta
    contract violations (missing meta, wrong contract/version, platform/raw
    mismatch, record-count mismatch, run-id mismatch). A missing DB is NOT an
    error — the v2 contract excludes the runner DB by design.
    """
    report: dict = {"errors": [], "warnings": [], "contract": "v2-ingestion"}
    version_mismatch = False

    sums = bundle_dir / "SHA256SUMS"
    if not sums.is_file():
        report["errors"].append("missing SHA256SUMS")
        return report, version_mismatch

    listed: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, sep, path = line.partition("  ")
        if not sep or len(digest) != 64:
            report["errors"].append(f"malformed SHA256SUMS line: {line!r}")
            continue
        listed[path.removeprefix("./")] = digest.lower()

    mismatches = []
    for rel, expected in sorted(listed.items()):
        f = bundle_dir / rel
        if not f.is_file():
            mismatches.append(f"{rel}: listed but missing")
            continue
        if hashlib.sha256(f.read_bytes()).hexdigest() != expected:
            mismatches.append(f"{rel}: checksum mismatch")
    if mismatches:
        report["errors"].extend(mismatches[:10])

    actual_files = {
        str(p.relative_to(bundle_dir))
        for p in bundle_dir.rglob("*") if p.is_file()
    }
    unlisted = sorted(actual_files - set(listed) - {"SHA256SUMS", "artifact.zip"})
    if unlisted:
        report["warnings"].append(f"files not in SHA256SUMS: {unlisted[:10]}")

    # --- bundle-meta: identity + cross-checks against actual contents -----
    meta_path = bundle_dir / "bundle-meta.json"
    if not meta_path.is_file():
        report["errors"].append(
            "missing bundle-meta.json — artifact predates finalized contract "
            "v2; re-publish from the updated workflow"
        )
        return report, version_mismatch
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        report["errors"].append(f"malformed bundle-meta.json: {exc}")
        return report, version_mismatch

    if meta.get("contract") != "icmi-ingestion-bundle" or meta.get("contract_version") != 2:
        report["errors"].append(
            "bundle-meta contract mismatch: expected "
            f"icmi-ingestion-bundle v2, got {meta.get('contract')!r} "
            f"v{meta.get('contract_version')!r}"
        )
    if (
        actions_run_id is not None
        and meta.get("actions_run_id") is not None
        and int(meta["actions_run_id"]) != int(actions_run_id)
    ):
        report["errors"].append(
            f"bundle-meta actions_run_id {meta['actions_run_id']} != downloaded "
            f"run id {actions_run_id}"
        )

    raw_root = bundle_dir / "raw"
    raw_runs = sorted(d.name for d in raw_root.iterdir() if d.is_dir()) if raw_root.is_dir() else []
    if not raw_runs:
        report["warnings"].append(
            "no raw runs in bundle — every platform failed or was skipped"
        )

    # every raw run dir must be claimed by exactly one platform in meta
    claimed: dict[str, str] = {}
    for platform, info in (meta.get("platforms") or {}).items():
        for run_id in info.get("run_ids", []) or []:
            if run_id in claimed:
                report["errors"].append(
                    f"bundle-meta: run {run_id} claimed by both "
                    f"{claimed[run_id]} and {platform}"
                )
            claimed[run_id] = platform
    for run_id in raw_runs:
        if run_id not in claimed:
            report["errors"].append(
                f"bundle-meta platform mismatch: raw run {run_id} present but "
                "not claimed by any platform"
            )
    for run_id, platform in claimed.items():
        if run_id not in raw_runs:
            report["errors"].append(
                f"bundle-meta platform mismatch: {platform} claims run "
                f"{run_id} but raw/{run_id}/ is missing"
            )

    fingerprint = None
    for run_id in raw_runs:
        mpath = bundle_dir / "metrics" / f"{run_id}.json"
        if not mpath.is_file():
            report["errors"].append(f"raw run without manifest: {run_id}")
            continue
        try:
            m = json.loads(mpath.read_text(encoding="utf-8"))
        except ValueError:
            report["errors"].append(f"corrupt manifest: {run_id}")
            continue
        fingerprint = fingerprint or m.get("extraction_version")
        claimed_platform = claimed.get(run_id)
        if claimed_platform and m.get("platform") != claimed_platform:
            report["errors"].append(
                f"manifest platform mismatch for {run_id}: manifest="
                f"{m.get('platform')!r} bundle-meta={claimed_platform!r}"
            )
        claimed_info = (meta.get("platforms") or {}).get(claimed_platform or "", {})
        records_file = (
            bundle_dir / "raw" / run_id / (claimed_platform or "") / "records.jsonl"
        )
        if claimed_platform and claimed_info and records_file.is_file():
            claimed_n = claimed_info.get("records_extracted")
            with records_file.open(encoding="utf-8") as fh:
                actual_n = sum(1 for line in fh if line.strip())
            if claimed_n is not None and int(claimed_n) != actual_n:
                report["errors"].append(
                    f"record-count mismatch for {run_id}: bundle-meta claims "
                    f"{claimed_n}, records.jsonl has {actual_n}"
                )
        if m.get("status") not in ("success", "partial"):
            report["warnings"].append(
                f"run {run_id}: status={m.get('status')} (ingest only consumes "
                "records.jsonl that was actually written)"
            )
    for ref in meta.get("manifest_refs", []) or []:
        if not (bundle_dir / ref).is_file():
            report["errors"].append(f"bundle-meta manifest_ref missing: {ref}")

    meta_extraction = meta.get("extraction_version")
    if (
        meta_extraction
        and fingerprint
        and meta_extraction != fingerprint
    ):
        report["errors"].append(
            f"bundle-meta extraction_version {meta_extraction} != manifest "
            f"fingerprint {fingerprint}"
        )

    report["raw_runs"] = raw_runs
    report["extraction_version"] = fingerprint
    report["bundle_meta"] = {
        k: meta.get(k)
        for k in ("contract", "contract_version", "actions_run_id",
                  "actions_run_attempt", "created_at")
    }

    current = _current_extraction_version()
    if fingerprint and current and fingerprint != current:
        version_mismatch = True
        report["extraction_version_mismatch"] = (
            f"bundle={fingerprint} local={current} — raw runs still import; "
            "refresh derived views locally after ingest"
        )
    return report, version_mismatch


def _extraction_fingerprint(snapshot_dir: Path) -> str | None:
    """Extraction version recorded in the snapshot's manifests — used to gate
    derived-data installation when the snapshot was built by incompatible
    extraction logic."""
    manifests = snapshot_dir / "manifests"
    if not manifests.is_dir():
        return None
    for f in sorted(manifests.glob("*.json")):
        try:
            m = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if m.get("extraction_version"):
            return m["extraction_version"]
    return None


def _current_extraction_version() -> str:
    from icmi.core.config import load_sources

    return load_sources().pipeline.extraction_version


def _verify(snapshot_dir: Path) -> tuple[dict, bool]:
    """Returns (report, version_mismatch). Hard errors land in
    report['errors']; a version mismatch is NOT a hard error — raw runs still
    import, derived data is skipped."""
    report: dict = {"errors": []}
    version_mismatch = False

    snap_db = snapshot_dir / "canonical" / "icmi.db"
    if not snap_db.is_file():
        report["errors"].append("no canonical/icmi.db in snapshot")
        return report, version_mismatch
    conn = sqlite3.connect(f"file:{snap_db}?mode=ro", uri=True)
    try:
        report["integrity"] = conn.execute("PRAGMA integrity_check").fetchone()[0]
        report["listings"] = conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
        report["observations"] = conn.execute(
            "SELECT COUNT(*) FROM price_observations").fetchone()[0]
        report["runs"] = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    finally:
        conn.close()
    if report["integrity"] != "ok":
        report["errors"].append(f"snapshot DB integrity: {report['integrity']}")

    manifests = snapshot_dir / "manifests"
    if manifests.is_dir():
        for f in manifests.glob("*.json"):
            try:
                json.loads(f.read_text(encoding="utf-8"))
            except ValueError:
                report["errors"].append(f"corrupt manifest: {f.name}")

    fingerprint = _extraction_fingerprint(snapshot_dir)
    report["extraction_version"] = fingerprint
    current = _current_extraction_version()
    if fingerprint and current and fingerprint != current:
        version_mismatch = True
        report["extraction_version_mismatch"] = (
            f"snapshot={fingerprint} local={current} — derived data will be "
            "skipped; raw runs still import"
        )
    return report, version_mismatch


def _import_snapshot(snapshot_dir: Path, skip_derived: bool = False) -> None:
    import subprocess

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "sync_latest_data.py"),
        "--source", str(snapshot_dir),
    ]
    if skip_derived:
        cmd.append("--skip-derived")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"import failed:\n{result.stdout}\n{result.stderr}")
    print(result.stdout.strip())


def _list_pending(token: str, repo: str) -> list[dict]:
    """List recent successful pipeline runs with pending/imported status.

    Import status comes from the local ledger (actions-run-<id>.json), which
    the import path writes on success — cloud run ids never collide with
    collection run_ids, so pending detection converges.
    """
    runs = requests.get(
        f"{GITHUB_API}/repos/{repo}/actions/runs",
        headers=_headers(token),
        params={"status": "success", "per_page": 20}, timeout=30,
    ).json()
    out = []
    for run in runs.get("workflow_runs", []):
        if not _is_expected_run(run):
            continue
        arts = requests.get(
            f"{GITHUB_API}/repos/{repo}/actions/runs/{run['id']}/artifacts",
            headers=_headers(token), timeout=30,
        ).json()
        names = [a["name"] for a in arts.get("artifacts", [])]
        out.append({
            "run_id": run["id"],
            "created_at": run["created_at"],
            "has_artifact": any(
                n.startswith(INGESTION_PREFIX) or n.startswith(SNAPSHOT_PREFIX)
                for n in names
            ),
            "imported": _already_imported(run["id"]),
        })
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--latest", action="store_true", help="import latest successful run")
    parser.add_argument("--run-id", type=int, default=None, help="specific Actions run id")
    parser.add_argument(
        "--pending", action="store_true",
        help="import all pending runs chronologically",
    )
    parser.add_argument("--list", action="store_true", help="list recent runs + import status")
    parser.add_argument("--check", action="store_true", help="compact status summary (no import)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not any([args.latest, args.run_id, args.pending, args.list, args.check]):
        parser.error("one of --latest, --run-id, --pending, --list, --check is required")

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN not set — add it to the repo .env", file=sys.stderr)
        return 2

    if args.check:
        _print_status(token)
        return 0
    if args.list:
        _list_runs(token)
        return 0

    repo = REPO
    if args.pending:
        # chronological import of all pending runs
        pending = _list_pending(token)
        not_yet = [r for r in pending if r["has_artifact"] and not r["imported"]]
        print(f"pending runs with artifacts: {len(not_yet)}")
        for run_info in sorted(not_yet, key=lambda r: r["created_at"]):
            print(f"\n--- importing run {run_info['run_id']} ({run_info['created_at']})")
            _do_import(token, repo, run_info["run_id"], args.dry_run)
        print("\npending import complete")
        return 0

    run_id = args.run_id
    if args.latest:
        run = _find_run(token, None)
        run_id = run["id"]
    _do_import(token, repo, run_id, args.dry_run)
    return 0


def _install_ingestion(bundle_dir: Path, actions_run_id: int, dry_run: bool) -> dict:
    """Write-once install of a v2 ingestion bundle, then standard ingest.

    Returns a report dict. Never overwrites existing local files: identical
    files are skipped, differing ones abort the import (local raw data is
    authoritative — a conflict means corruption or a hash collision).
    """
    from icmi.core.paths import manifests_dir, raw_dir

    raw_root = raw_dir()
    man_root = manifests_dir()

    pairs: list[tuple[Path, Path]] = []
    bundle_raw = bundle_dir / "raw"
    if bundle_raw.is_dir():
        for run_dir in sorted(bundle_raw.iterdir()):
            if not run_dir.is_dir():
                continue
            for f in sorted(run_dir.rglob("*")):
                if f.is_file():
                    pairs.append((f, raw_root / f.relative_to(bundle_raw)))
    bundle_metrics = bundle_dir / "metrics"
    if bundle_metrics.is_dir():
        for f in sorted(bundle_metrics.glob("*.json")):
            if f.name == "summary.json":
                continue
            if (bundle_raw / f.stem).is_dir():
                pairs.append((f, man_root / f.name))

    installed = unchanged = 0
    conflicts: list[str] = []
    for src, dst in pairs:
        if dst.is_file():
            if dst.read_bytes() == src.read_bytes():
                unchanged += 1
            else:
                conflicts.append(str(dst))
        else:
            installed += 1
    if conflicts:
        raise SystemExit(
            "ABORTING — install conflicts (local file differs from bundle, "
            f"never overwritten): {conflicts[:5]}"
        )
    if not dry_run:
        for src, dst in pairs:
            if not dst.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    print(f"install: {installed} new, {unchanged} identical (write-once)")

    from icmi.core.paths import canonical_dir
    from icmi.pipeline.runner import PipelineRunner

    db_path = canonical_dir() / "icmi.db"
    before = _counts(db_path)

    if dry_run:
        print("dry-run: no changes made")
        return {"installed_files": installed, "unchanged_files": unchanged,
                "before": before, "dry_run": True}

    backup = db_path.parent / f"pre-import-{actions_run_id}-{int(time.time())}.db"
    if db_path.is_file():
        src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        dst = sqlite3.connect(backup)
        src.backup(dst)
        dst.close()
        src.close()
        print(f"pre-import DB backup: {backup}")

    runner = PipelineRunner(mode="manual")
    report = runner.run(["ingest"])
    ingest_line = next((s.detail for s in report.stages if s.name == "ingest"), "")
    print(f"ingest: [{report.ok}] {ingest_line}")

    after = _counts(db_path)
    print(f"local DB: {before} -> {after}")
    if after["listings"] < before["listings"] or after["runs"] < before["runs"]:
        print(
            "FATAL: local DB shrank during import — restoring pre-import backup",
            file=sys.stderr,
        )
        shutil.copy2(backup, db_path)
        return {"shrink_guard": "restored-backup", "before": before, "after": after}

    # idempotence probe: re-run the same ingest — run-level skip must make it a no-op
    probe = runner.run(["ingest"])
    probe_line = next((s.detail for s in probe.stages if s.name == "ingest"), "")
    final = _counts(db_path)
    idempotent = final == after
    print(f"idempotence probe: {probe_line} -> "
          f"{'verified (zero changes)' if idempotent else 'VIOLATION'}")
    if not idempotent:
        raise SystemExit("FATAL: second import changed the DB — idempotence violated")

    out = {
        "before": before,
        "after": after,
        "installed_files": installed,
        "unchanged_files": unchanged,
        "ingest": ingest_line,
        "idempotent": idempotent,
    }
    ledger = _import_ledger_path(actions_run_id)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(json.dumps(
        {"actions_run_id": actions_run_id, "imported_at": time.time(), **out},
        indent=1,
    ), encoding="utf-8")
    return out


def _do_import(token: str, repo: str, run_id: int, dry_run: bool) -> None:
    if _already_imported(run_id):
        print(f"actions run {run_id}: already imported (ledger) — nothing to do")
        return
    run = requests.get(
        f"{GITHUB_API}/repos/{repo}/actions/runs/{run_id}",
        headers=_headers(token), timeout=30,
    ).json()
    print(f"actions run: {run_id} | conclusion={run.get('conclusion')} | {run['created_at']}")

    with tempfile.TemporaryDirectory(prefix="icmi-actions-sync-") as tmp:
        workdir = Path(tmp)
        extract_root, kind, dl_meta = _download_artifact(token, run_id, workdir)
        print(f"artifact: {dl_meta['artifact_name']} | {dl_meta['compressed_kb']} KB "
              f"| {dl_meta['download_seconds']}s | contract={kind}")

        if kind == "ingestion":
            verification, _version_mismatch = _verify_ingestion(extract_root, run_id)
            print(f"verification: {json.dumps(verification, default=str)}")
            if verification["errors"]:
                print("ABORTING — verification errors:", verification["errors"], file=sys.stderr)
                return
            report = _install_ingestion(extract_root, run_id, dry_run)
            report.update({
                "run_id": run_id,
                "download": dl_meta,
                "verification": verification,
            })
            print("report:", json.dumps(report, default=str))
            return

        # legacy v1 snapshot contract
        snapshot_dir = extract_root
        verification, version_mismatch = _verify(snapshot_dir)
        print(f"verification: {json.dumps(verification, default=str)}")
        if verification["errors"]:
            print("ABORTING — verification errors:", verification["errors"], file=sys.stderr)
            return
        if version_mismatch:
            print("extraction version mismatch — importing raw runs only")

        from icmi.core.paths import canonical_dir

        db_path = canonical_dir() / "icmi.db"
        before = _counts(db_path)
        already = _already_ingested_run_ids(db_path)
        snapshot_runs = _snapshot_run_ids(snapshot_dir)
        new_runs = [r for r in snapshot_runs if r not in already]
        print(f"snapshot runs: {len(snapshot_runs)} | new to import: {len(new_runs)}")

        if dry_run:
            print("dry-run: no changes made")
            return

        backup = db_path.parent / f"pre-import-{run_id}-{int(time.time())}.db"
        if db_path.is_file():
            src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            dst = sqlite3.connect(backup)
            src.backup(dst)
            dst.close()
            src.close()
            print(f"pre-import DB backup: {backup}")

        _import_snapshot(snapshot_dir, skip_derived=version_mismatch)

        after = _counts(db_path)
        print(f"local DB: {before} -> {after}")
        if after["listings"] < before["listings"] or after["runs"] < before["runs"]:
            print(
                "FATAL: local DB shrank during import — restoring pre-import backup",
                file=sys.stderr,
            )
            shutil.copy2(backup, db_path)
            return

        # idempotence probe
        _import_snapshot(snapshot_dir, skip_derived=version_mismatch)
        final = _counts(db_path)
        if final != after:
            print("FATAL: second import changed the DB — idempotence violated", file=sys.stderr)
            return
        print("idempotence verified: second import changed nothing")

        report = {
            "run_id": run_id,
            "snapshot_runs": len(snapshot_runs),
            "new_runs_imported": len(new_runs),
            "before": before, "after": after,
            "download": dl_meta,
            "verification": verification,
            "version_mismatch": version_mismatch,
        }
        print("report:", json.dumps(report, default=str))


def _print_status(token: str) -> None:
    from icmi.core.paths import canonical_dir

    db = canonical_dir() / "icmi.db"
    runs = requests.get(
        f"{GITHUB_API}/repos/{REPO}/actions/runs",
        headers=_headers(token),
        params={"status": "success", "per_page": 10}, timeout=30,
    ).json()
    cloud_runs = [r for r in runs.get("workflow_runs", []) if _is_expected_run(r)]
    pending = [r for r in cloud_runs if not _already_imported(r["id"])]
    latest = cloud_runs[0] if cloud_runs else None
    print(f"latest cloud run: {latest['id'] if latest else '—'} "
          f"({latest['created_at'] if latest else '—'})")
    print(f"pending cloud runs: {len(pending)}")
    for r in pending[:5]:
        print(f"  {r['id']} ({r['created_at']})")
    if db.is_file():
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        n_l = conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
        n_o = conn.execute("SELECT COUNT(*) FROM price_observations").fetchone()[0]
        print(f"local DB: {n_l} listings, {n_o} observations")
        conn.close()


def _list_runs(token: str) -> None:
    runs = requests.get(
        f"{GITHUB_API}/repos/{REPO}/actions/runs",
        headers=_headers(token),
        params={"status": "success", "per_page": 10}, timeout=30,
    ).json()
    for r in runs.get("workflow_runs", []):
        if not _is_expected_run(r):
            continue
        mark = "imported" if _already_imported(r["id"]) else "pending "
        print(f"  [{mark}]  {r['id']}  {r['created_at']}  {r['path']}")


def _counts(db_path: Path) -> dict:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return {
            "listings": conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0],
            "observations": conn.execute(
                "SELECT COUNT(*) FROM price_observations").fetchone()[0],
            "runs": conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
        }
    finally:
        conn.close()


def _already_ingested_run_ids(db_path: Path) -> set[str]:
    if not db_path.is_file():
        return set()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return {r[0] for r in conn.execute("SELECT run_id FROM runs")}
    finally:
        conn.close()


def _snapshot_run_ids(snapshot_dir: Path) -> list[str]:
    raw = snapshot_dir / "raw"
    return sorted(d.name for d in raw.iterdir() if d.is_dir()) if raw.is_dir() else []


if __name__ == "__main__":
    sys.exit(main())
