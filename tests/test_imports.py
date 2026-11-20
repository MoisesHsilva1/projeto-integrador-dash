"""Verify that all modules in the package import cleanly."""

from __future__ import annotations

import importlib
import pkgutil

import credit_card_fraud


def test_all_modules_importable() -> None:
    failed: list[tuple[str, str]] = []
    for module_info in pkgutil.walk_packages(
        credit_card_fraud.__path__, credit_card_fraud.__name__ + "."
    ):
        try:
            importlib.import_module(module_info.name)
        except Exception as error:
            failed.append((module_info.name, str(error)))

    assert not failed, f"Failed importing modules: {failed}"
