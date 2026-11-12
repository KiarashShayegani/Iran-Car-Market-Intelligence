"""Operational journal + health checks: unit tests.

Journal: event schema, level validation, rotation, tail, best-effort on
write failure. Health: baseline gates — same-budget yield comparison,
corrupt manifests, platform totals, summary formatting.
"""

from __future__ import annotations

import json

from icmi.ops import journal as journal_mod
from icmi.ops.health import build_health_report, summarize_health


def test_journal_event_schema_and_level_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(journal_mod, "journal_dir", lambda: tmp_path)
    rec = journal_mod.journal_event("sync", "test_event", level="bogus", k=1)
    assert rec["level"] == "info"  # invalid level falls back
    assert rec["kind"] == "sync" and rec["event"] == "test_event" and rec["k"] == 1
    line = (tmp_path / "journal.jsonl").read_text().strip()
    assert json.loads(line)["event"] == "test_event"


def test_journal_tail_and_levels(tmp_path, monkeypatch):
    monkeypatch.setattr(journal_mod, "journal_dir", lambda: tmp_path)
    for i in range(5):
        journal_mod.journal_event("health", f"e{i}", level="critical" if i == 4 else "info")
    tail = journal_mod.journal_tail(3)
    assert [r["event"] for r in tail] == ["e2", "e3", "e4"]
    crit = journal_mod.journal_tail(5, level="critical")
    assert len(crit) == 1 and crit[0]["event"] == "e4"


def test_journal_rotation(tmp_path, monkeypatch):
    monkeypatch.setattr(journal_mod, "journal_dir", lambda: tmp_path)
    path = tmp_path / "journal.jsonl"
    path.write_bytes(b"x" * (journal_mod.MAX_BYTES + 1))
    journal_mod.journal_event("health", "after_rotation")
    assert path.with_suffix(".jsonl.1").stat().st_size > journal_mod.MAX_BYTES - 2
    assert json.loads(path.read_text())["event"] == "after_rotation"


def test_journal_write_failure_is_silent(tmp_path, monkeypatch):
    monkeypatch.setattr(journal_mod, "journal_dir", lambda: tmp_path)
    # make rotation fail (path is a directory where the file would go)
    (tmp_path / "journal.jsonl").mkdir()
    rec = journal_mod.journal_event("health", "must_not_raise")
    assert rec["event"] == "must_not_raise"  # no exception surfaced


def test_health_yield_check_requires_same_budget(tmp_path, monkeypatch):
    """Deliberately small experiment runs (1 page) must not alarm when the
    baseline is 3-page runs — regression for the live false alarm."""
    monkeypatch.setattr(
        "icmi.ops.health.manifests_dir", lambda: tmp_path
    )
    runs = []
    for i in range(4):
        m = {
            "run_id": f"2026110{i}T000000-sheypoor-ci",
            "platform": "sheypoor",
            "status": "success",
            "started_at": f"2026-11-{10 + i:02d}T00:00:00Z",
            "records": {"extracted": 50},
            "pages": {"requested": 3, "fetched": 3},
            "quality": {"p0_coverage_pct": 91.7},
        }
        p = tmp_path / f"{m['run_id']}.json"
        p.write_text(json.dumps(m))
        runs.append(m)
    # a small 1-page experiment run lands last
    small = {
        "run_id": "20261120T000000-sheypoor-ci",
        "platform": "sheypoor",
        "status": "success",
        "started_at": "2026-11-20T00:00:00Z",
        "records": {"extracted": 12},
        "pages": {"requested": 1, "fetched": 1},
        "quality": {"p0_coverage_pct": 91.7},
    }
    (tmp_path / f"{small['run_id']}.json").write_text(json.dumps(small))

    report = build_health_report()
    yield_checks = [c for c in report["checks"] if c["check"] == "yield_drop"]
    assert yield_checks == []  # budget mismatch → not comparable → no alarm


def test_health_zero_record_and_corrupt_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr("icmi.ops.health.manifests_dir", lambda: tmp_path)
    (tmp_path / "20261120T000000-bama-ci.json").write_text(
        json.dumps({
            "run_id": "20261120T000000-bama-ci", "platform": "bama",
            "status": "success", "started_at": "2026-11-20T00:00:00Z",
            "records": {"extracted": 0}, "pages": {},
            "quality": {"p0_coverage_pct": None},
        })
    )
    (tmp_path / "broken.json").write_text("{corrupt")
    report = build_health_report()
    checks = {c["check"] for c in report["checks"]}
    assert "zero_record_run" in checks
    assert "manifest_integrity" in checks
    assert report["critical_count"] >= 1


def test_health_summary_formatting():
    report = {
        "critical_count": 1, "warning_count": 2,
        "checks": [
            {"check": "a", "level": "critical", "detail": "bad"},
            {"check": "b", "level": "warning", "detail": "meh"},
        ],
    }
    s = summarize_health(report)
    assert "health=CRITICAL" in s and "bad" in s
