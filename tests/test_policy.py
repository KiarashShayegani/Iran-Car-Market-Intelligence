"""Tests for the collection policy (Phase 1)."""

import pytest

from icmi.pipeline.policy import load_policy


def test_load_policy_validates():
    policy = load_policy()
    assert policy.schema_version == 2
    assert policy.sources["bama"].enabled
    # measurement-derived budgets (2026-11-13): burst discipline
    assert policy.sources["bama"].max_listings_per_run == 60
    assert policy.sources["bama"].max_pages_per_preset == 2
    assert policy.sources["bama"].presets_per_run == 2
    assert policy.sources["sheypoor"].cadence == "2x_daily"
    assert len(policy.sources["bama"].presets) == 14


def test_disabled_sources_and_gates():
    policy = load_policy()
    assert policy.sources["divar"].enabled is False
    assert policy.sources["karnameh"].enabled is False
    # disabled takes precedence — a disabled source never runs regardless of gates
    with pytest.raises(ValueError, match="disabled"):
        policy.assert_can_collect("divar")
    with pytest.raises(ValueError, match="disabled"):
        policy.assert_can_collect("karnameh")
    # gated check: a hypothetical enabled-but-gated source
    from icmi.pipeline.policy import SourcePolicy

    policy.sources["test_gated"] = SourcePolicy(enabled=True, gated="browser-capture")
    with pytest.raises(ValueError, match="gated"):
        policy.assert_can_collect("test_gated")


def test_enabled_sources_pass():
    policy = load_policy()
    for platform in ("bama", "sheypoor", "hamrah_mechanic"):
        policy.assert_can_collect(platform)  # must not raise


def test_rotation_is_deterministic_and_covers_all_presets():
    policy = load_policy()
    presets = policy.sources["bama"].presets
    seen = []
    for run_index in range(7):  # 2 per run × 7 runs = 14 → full cycle
        seen.extend(policy.rotation_for("bama", run_index))
    # deterministic: same run_index → same slice
    assert policy.rotation_for("bama", 0) == policy.rotation_for("bama", 0)
    # 7 runs × 2 presets = 14 → every preset appears exactly once per cycle
    assert sorted(seen) == sorted(presets)


def test_unknown_platform_rejected():
    policy = load_policy()
    with pytest.raises(ValueError, match="unknown platform"):
        policy.assert_can_collect("nonexistent")


def test_policy_file_has_expected_sources():
    policy = load_policy()
    assert set(policy.sources) == {
        "bama", "sheypoor", "hamrah_mechanic", "divar", "karnameh",
        "test_gated",
    }
