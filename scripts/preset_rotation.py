#!/usr/bin/env python3
"""Print today's Bama presets for scheduled (input-less) runs.

GitHub cron cannot pass dispatch inputs, so the bama job calls this on the
runner: rotation index = days since PILOT_START (deterministic, needs no
database), advancing through the policy's preset list presets_per_run at a
time — a full brand cycle every ceil(14/2)=7 days, matching the collection
policy's burst discipline (one preset set per night).

Manual dispatches may still pass explicit `presets` (or use
scripts/rotation_pick.py for coverage-driven choices).
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from icmi.pipeline.policy import load_policy

PILOT_START = date(2026, 11, 13)


def main() -> int:
    index = (datetime.now(tz=UTC).date() - PILOT_START).days
    presets = load_policy().rotation_for("bama", max(0, index))
    print(",".join(presets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
