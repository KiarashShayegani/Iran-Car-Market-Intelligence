"""Tests for configuration loading."""

import pytest

from icmi.core import config


def test_load_default_sources_yaml():
    sources = config.load_sources()
    enabled = sources.enabled_sources()
    assert set(enabled) == {"bama", "sheypoor", "hamrah_mechanic"}
    assert sources.sources["divar"].enabled is False
    assert sources.pipeline.extraction_version.startswith("icmi-v3")


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        config.load_sources(tmp_path / "nope.yaml")


def test_load_rejects_other_schema_version(tmp_path):
    bad = tmp_path / "sources.yaml"
    bad.write_text(
        "schema_version: 999\n"
        "pipeline:\n"
        "  extraction_version: test-1\n"
        "  request_timeout_seconds: 10\n"
        "  request_delay_seconds: 0\n"
        "  request_jitter_seconds: 0\n"
        "  max_retries: 1\n"
        "  user_agent: test-agent\n"
        "sources: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema_version"):
        config.load_sources(bad)


def test_load_rejects_non_mapping(tmp_path):
    bad = tmp_path / "sources.yaml"
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        config.load_sources(bad)


def test_icmi_config_env_override(tmp_path, monkeypatch):
    custom = tmp_path / "sources.yaml"
    custom.write_text(
        "schema_version: 1\n"
        "pipeline:\n"
        "  extraction_version: test-1\n"
        "  request_timeout_seconds: 10\n"
        "  request_delay_seconds: 0\n"
        "  request_jitter_seconds: 0\n"
        "  max_retries: 1\n"
        "  user_agent: test-agent\n"
        "sources: {}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ICMI_CONFIG", str(custom))
    sources = config.load_sources()
    assert sources.pipeline.extraction_version == "test-1"
    assert sources.enabled_sources() == {}
