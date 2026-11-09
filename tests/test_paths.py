"""Tests for Data Layer path resolution."""

from icmi.core import paths


def test_repo_root_contains_pyproject():
    root = paths.repo_root()
    assert (root / "pyproject.toml").is_file()


def test_data_dir_defaults_inside_repo():
    assert paths.data_dir() == paths.repo_root() / "data"


def test_data_subdirs_are_created():
    for get_dir in (
        paths.raw_dir,
        paths.canonical_dir,
        paths.index_dir,
        paths.manifests_dir,
        paths.models_dir,
    ):
        path = get_dir()
        assert path.is_dir()
        assert path.parent == paths.data_dir()


def test_data_dir_override(tmp_path, monkeypatch):
    monkeypatch.setenv("ICMI_DATA_DIR", str(tmp_path / "elsewhere"))
    assert paths.data_dir() == tmp_path / "elsewhere"
    assert (tmp_path / "elsewhere").is_dir()
    assert paths.raw_dir() == tmp_path / "elsewhere" / "raw"
