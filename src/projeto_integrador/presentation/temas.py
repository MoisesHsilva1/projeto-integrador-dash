"""Tema visual e tokens semânticos do dashboard (pt-BR, estilo financeiro)."""

from __future__ import annotations

from typing import Any

# Cores semânticas consistentes em todo o dashboard.
COR_LEGITIMA = "#2e9e6b"
COR_FRAUDE = "#d64545"
COR_NEUTRA = "#6c7a89"
COR_ALERTA = "#d9a441"
COR_FUNDO = "#0e1117"

CORES_SEMANTICAS = {
    "legitima": COR_LEGITIMA,
    "fraude": COR_FRAUDE,
    "neutra": COR_NEUTRA,
    "alerta": COR_ALERTA,
}

LAYOUT_PADRAO: dict[str, Any] = {
    "template": "plotly_dark",
    "font": {"family": "Inter, sans-serif", "size": 12},
    "margin": {"l": 40, "r": 20, "t": 50, "b": 40},
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "hoverlabel": {"namelength": -1},
}


def aplicar_tema(figura: Any) -> Any:
    """Aplica o layout padrão e os tokens visuais a uma figura Plotly."""
    figura.update_layout(**LAYOUT_PADRAO)
    return figura
