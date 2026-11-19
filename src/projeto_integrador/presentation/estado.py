"""Coordenação de estado de sessão do dashboard (Application Layer)."""

from __future__ import annotations

import streamlit as st

from projeto_integrador.application.filtros import FiltrosDashboard

CHAVE_FILTROS = "filtros_dashboard"


def inicializar_filtros(limites: dict[str, float]) -> FiltrosDashboard:
    """Garante que exista um estado de filtros inicial válido."""
    if CHAVE_FILTROS not in st.session_state:
        st.session_state[CHAVE_FILTROS] = FiltrosDashboard(
            valor_min=float(limites["valor_min"]),
            valor_max=float(limites["valor_max"]),
            hora_inicio=0.0,
            hora_fim=float(limites["hora_max"]),
        )
    return st.session_state[CHAVE_FILTROS]


def atualizar_filtros(filtros: FiltrosDashboard) -> None:
    """Substitui o estado atual de filtros."""
    st.session_state[CHAVE_FILTROS] = filtros


def limpar_filtros(limites: dict[str, float]) -> FiltrosDashboard:
    """Restaura os filtros para o padrão e retorna o novo estado."""
    novo = FiltrosDashboard(
        valor_min=float(limites["valor_min"]),
        valor_max=float(limites["valor_max"]),
        hora_inicio=0.0,
        hora_fim=float(limites["hora_max"]),
    )
    st.session_state[CHAVE_FILTROS] = novo
    return novo


def obter_filtros_ativos(
    filtros: FiltrosDashboard, limites: dict[str, float]
) -> list[str]:
    """Lista textual dos filtros atualmente aplicados (para exibição)."""
    ativos: list[str] = []
    if (
        filtros.valor_min > limites["valor_min"]
        or filtros.valor_max < limites["valor_max"]
    ):
        ativos.append(
            f"Valor: {filtros.valor_min:,.2f} – {filtros.valor_max:,.2f}".replace(
                ",", "§"
            )
            .replace(".", ",")
            .replace("§", ".")
        )
    if filtros.hora_inicio > 0 or filtros.hora_fim < limites["hora_max"]:
        ativos.append(f"Tempo: {filtros.hora_inicio:.0f} – {filtros.hora_fim:.0f} h")
    if filtros.status.value != "todas":
        rotulos = {"fraude": "Somente fraudes", "legitima": "Somente legítimas"}
        ativos.append(rotulos.get(filtros.status.value, filtros.status.value))
    return ativos
