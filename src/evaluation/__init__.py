"""
Evaluation and visualization modules.
"""

from .metrics import (
    calculate_classification_metrics,
    calculate_roc_curve,
    calculate_precision_recall_curve,
    calculate_metrics_at_thresholds
)
from .visualizations import (
    plot_roc_curve,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_threshold_analysis,
    plot_score_distribution,
    plot_model_comparison
)

__all__ = [
    'calculate_classification_metrics',
    'calculate_roc_curve',
    'calculate_precision_recall_curve',
    'calculate_metrics_at_thresholds',
    'plot_roc_curve',
    'plot_confusion_matrix',
    'plot_feature_importance',
    'plot_threshold_analysis',
    'plot_score_distribution',
    'plot_model_comparison'
]
