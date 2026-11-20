"""Dashboard session state coordination (Application Layer)."""

from __future__ import annotations

import streamlit as st

from credit_card_fraud.application.filters import DashboardFilters
from credit_card_fraud.formatting import format_amount

FILTERS_KEY = "dashboard_filters"


def initialize_filters(limits: dict[str, float]) -> DashboardFilters:
    """Ensure a valid initial filter state exists."""
    if FILTERS_KEY not in st.session_state:
        st.session_state[FILTERS_KEY] = DashboardFilters(
            min_amount=float(limits["min_amount"]),
            max_amount=float(limits["max_amount"]),
            start_hour=0.0,
            end_hour=float(limits["max_hour"]),
        )
    return st.session_state[FILTERS_KEY]


def update_filters(filters: DashboardFilters) -> None:
    """Replace the current filter state."""
    st.session_state[FILTERS_KEY] = filters


def clear_filters(limits: dict[str, float]) -> DashboardFilters:
    """Restore filters to defaults and return the new state."""
    new = DashboardFilters(
        min_amount=float(limits["min_amount"]),
        max_amount=float(limits["max_amount"]),
        start_hour=0.0,
        end_hour=float(limits["max_hour"]),
    )
    st.session_state[FILTERS_KEY] = new
    return new


def get_active_filters(
    filters: DashboardFilters, limits: dict[str, float]
) -> list[str]:
    """Textual list of currently applied filters (for display)."""
    active: list[str] = []
    if (
        filters.min_amount > limits["min_amount"]
        or filters.max_amount < limits["max_amount"]
    ):
        active.append(
            f"Amount: {format_amount(filters.min_amount)} – "
            f"{format_amount(filters.max_amount)}"
        )
    if filters.start_hour > 0 or filters.end_hour < limits["max_hour"]:
        active.append(
            f"Time: {format_amount(filters.start_hour, 0)} – "
            f"{format_amount(filters.end_hour, 0)} h"
        )
    if filters.status.value != "all":
        labels = {"fraud": "Somente fraudes", "legitimate": "Somente legítimas"}
        active.append(labels.get(filters.status.value, filters.status.value))
    return active
