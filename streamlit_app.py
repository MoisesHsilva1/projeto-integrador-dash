"""Entrypoint used by Streamlit Community Cloud.

The cloud runner executes ``streamlit run`` from the repository root and does
not install this project as a package, so ``src`` is added to ``sys.path``
before importing the application modules. Local execution is unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parent / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))


def run() -> None:
    """Import and render the dashboard after the path is configured."""
    from credit_card_fraud.presentation.app import main

    main()


if __name__ == "__main__":
    run()
