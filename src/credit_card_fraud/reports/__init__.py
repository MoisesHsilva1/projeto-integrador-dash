"""Report and figure generation (deliverables of steps E3, E4 and E7)."""

from credit_card_fraud.reports.eda import generate_eda_report, generate_figures
from credit_card_fraud.reports.model import generate_model_report
from credit_card_fraud.reports.quality import generate_quality_report

__all__ = [
    "generate_figures",
    "generate_eda_report",
    "generate_model_report",
    "generate_quality_report",
]
