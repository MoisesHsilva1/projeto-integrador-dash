
from __future__ import annotations

import os
from pathlib import Path
from urllib.request import Request, urlopen

from credit_card_fraud.config import PROCESSED_PATH

DEFAULT_REPOSITORY = "MoisesHsilva1/projeto-integrador-dash"
DEFAULT_RELEASE_TAG = "data-v1"
DEFAULT_ASSET_NAME = "transactions.parquet"
DOWNLOAD_TIMEOUT = 300
CHUNK_SIZE = 1024 * 1024


def setting(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st

        secret = st.secrets.get(name)
        return str(secret) if secret else None
    except Exception:
        return None


def download_url() -> str:
    explicit = setting("DATA_DOWNLOAD_URL")
    if explicit:
        return explicit
    repository = setting("DATA_RELEASE_REPO") or DEFAULT_REPOSITORY
    tag = setting("DATA_RELEASE_TAG") or DEFAULT_RELEASE_TAG
    asset = setting("DATA_RELEASE_ASSET") or DEFAULT_ASSET_NAME
    return f"https://github.com/{repository}/releases/download/{tag}/{asset}"


def _headers() -> dict[str, str]:
    token = setting("DATA_DOWNLOAD_TOKEN")
    headers = {"Accept": "application/octet-stream"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def download_prepared(url: str, destination: Path) -> None:
    """Download the release asset into ``destination`` using a temporary file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    request = Request(url, headers=_headers())
    try:
        with (
            urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response,
            temporary.open("wb") as output,
        ):
            while chunk := response.read(CHUNK_SIZE):
                output.write(chunk)
        with temporary.open("rb") as downloaded:
            if downloaded.read(4) != b"PAR1":
                raise ValueError("downloaded asset is not a Parquet file")
    except (OSError, ValueError) as error:
        temporary.unlink(missing_ok=True)
        raise FileNotFoundError(
            f"Could not obtain the prepared base from {url}: {error}. "
            "See docs/DEPLOYMENT.md."
        ) from error
    temporary.replace(destination)


def ensure_prepared_data(path: Path | None = None) -> Path:
    destination = Path(path) if path is not None else PROCESSED_PATH
    if destination.exists():
        return destination
    download_prepared(download_url(), destination)
    return destination
