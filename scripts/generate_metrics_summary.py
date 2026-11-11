#!/usr/bin/env python3
"""Generate per-source metrics summary from manifests for CI artifacts."""
import json
from pathlib import Path

metrics = {}
for f in Path("data/manifests").glob("*.json"):
    m = json.loads(f.read_text())
    p = m.get("platform", "?")
    metrics[p] = {
        "status": m.get("status"),
        "records": m.get("records", {}).get("extracted", 0),
        "p0_coverage": m.get("quality", {}).get("p0_coverage_pct"),
    }
out = Path("data/g4-bundle/metrics/summary.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(metrics, indent=1))
print(f"metrics summary: {len(metrics)} platforms -> {out}")
