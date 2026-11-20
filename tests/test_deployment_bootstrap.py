"""Tests for the runtime data bootstrap used by the hosted dashboard."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from credit_card_fraud.deployment import bootstrap

SETTINGS = (
    "DATA_DOWNLOAD_URL",
    "DATA_RELEASE_REPO",
    "DATA_RELEASE_TAG",
    "DATA_RELEASE_ASSET",
    "DATA_DOWNLOAD_TOKEN",
)


@pytest.fixture(autouse=True)
def _clear_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in SETTINGS:
        monkeypatch.delenv(name, raising=False)


class _FakeResponse(io.BytesIO):
    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def test_download_url_uses_release_defaults() -> None:
    assert bootstrap.download_url() == (
        "https://github.com/MoisesHsilva1/projeto-integrador-dash/"
        "releases/download/data-v1/transactions.parquet"
    )


def test_download_url_uses_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATA_RELEASE_REPO", "Owner/Repo")
    monkeypatch.setenv("DATA_RELEASE_TAG", "v2")
    monkeypatch.setenv("DATA_RELEASE_ASSET", "custom.parquet")

    assert bootstrap.download_url() == (
        "https://github.com/Owner/Repo/releases/download/v2/custom.parquet"
    )


def test_download_url_explicit_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DOWNLOAD_URL", "https://example.org/base.parquet")

    assert bootstrap.download_url() == "https://example.org/base.parquet"


def test_ensure_prepared_data_keeps_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = tmp_path / "transactions.parquet"
    existing.write_bytes(b"PAR1 local")

    def fail_download(url: str, destination: Path) -> None:
        raise AssertionError("download should not run")

    monkeypatch.setattr(bootstrap, "download_prepared", fail_download)

    assert bootstrap.ensure_prepared_data(existing) == existing


def test_download_prepared_writes_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_urlopen(request: object, timeout: float) -> _FakeResponse:
        assert timeout == bootstrap.DOWNLOAD_TIMEOUT
        return _FakeResponse(b"PAR1" + b"x" * 32)

    monkeypatch.setattr(bootstrap, "urlopen", fake_urlopen)
    destination = tmp_path / "nested" / "transactions.parquet"

    bootstrap.download_prepared("https://example.org/base.parquet", destination)

    assert destination.read_bytes().startswith(b"PAR1")
    assert not destination.with_name(destination.name + ".part").exists()


def test_download_prepared_rejects_invalid_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_urlopen(request: object, timeout: float) -> _FakeResponse:
        return _FakeResponse(b"not a parquet file")

    monkeypatch.setattr(bootstrap, "urlopen", fake_urlopen)

    with pytest.raises(FileNotFoundError, match="Could not obtain"):
        bootstrap.download_prepared("https://example.org/x", tmp_path / "x.parquet")
