#!/usr/bin/env python3
"""Record a run manifest into the git-tracked manifests/ directory.

Manifests are the only collection evidence that enters git (datasets never
do). `scripts/collect.py` writes manifests to data/manifests/ (gitignored
Data Layer); this script copies one into manifests/ so collection history is
visible in the repository. GitHub Actions runs this automatically; locally
it is opt-in.

Example:
    python scripts/record_manifest.py --run-id 20261110T120000-bama-local
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from icmi.core.paths import manifests_dir, repo_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--all", action="store_true", help="record every manifest")
    args = parser.parse_args(argv)
    if not args.all and not args.run_id:
        parser.error("either --run-id or --all is required")

    src_dir = manifests_dir()
    dest_dir = repo_root() / "manifests"
    dest_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        files = sorted(src_dir.glob("*.json"))
    else:
        single = src_dir / f"{args.run_id}.json"
        files = [single] if single.is_file() else []
        if not files:
            print(f"Manifest not found: {single}", file=sys.stderr)
            return 2

    recorded = 0
    for f in files:
        _copy_sanitized(f, dest_dir / f.name)
        recorded += 1
    print(f"recorded {recorded} manifest(s) into {dest_dir}")
    return 0


def _copy_sanitized(src: Path, dest: Path) -> None:
    """Copy a manifest, rewriting any machine-specific absolute repo paths
    (legacy manifests carry them; new ones are repo-relative already)."""
    root = str(repo_root())

    def scrub(value):
        if isinstance(value, str):
            return value.replace(root + "/", "").replace(root, ".")
        if isinstance(value, list):
            return [scrub(v) for v in value]
        if isinstance(value, dict):
            return {k: scrub(v) for k, v in value.items()}
        return value

    payload = json.loads(src.read_text(encoding="utf-8"))
    dest.write_text(
        json.dumps(scrub(payload), ensure_ascii=False, indent=1, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    sys.exit(main())
