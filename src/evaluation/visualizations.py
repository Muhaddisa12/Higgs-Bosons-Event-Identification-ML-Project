"""
Visualization utilities for model evaluation and results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from pathlib import Path
import seaborn as sns

# Set style
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    plt.style.use('seaborn')
sns.set_palette("husl")


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float,
                   label: str = 'Model', savepath: Optional[str] = None) -> None:
    """
    Plot ROC curve.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc: AUC-ROC score
        label: Model label
        savepath: Path to save figure
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{label} (AUC = {auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          model_name: str = 'Model',
                          savepath: Optional[str] = None) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        savepath: Path to save figure
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Background', 'Signal'],
                yticklabels=['Background', 'Signal'])
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance(importances: np.ndarray, feature_names: List[str],
                            model_name: str = 'Model',
                            savepath: Optional[str] = None) -> None:
    """
    Plot feature importance.
    
    Args:
        importances: Feature importance scores
        feature_names: Names of features
        model_name: Name of the model
        savepath: Path to save figure
    """
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    sorted_importances = importances[indices]
    sorted_names = [feature_names[i] for i in indices]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_names)), sorted_importances)
    plt.yticks(range(len(sorted_names)), sorted_names)
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_threshold_analysis(thresholds: np.ndarray, signal_eff: np.ndarray,
                           bg_reject: np.ndarray, sb_ratio: np.ndarray,
                           model_name: str = 'Model',
                           savepath: Optional[str] = None) -> None:
    """
    Plot threshold analysis showing signal efficiency, background rejection, and S/B ratio.
    
    Args:
        thresholds: Threshold values
        signal_eff: Signal efficiency at each threshold
        bg_reject: Background rejection at each threshold
        sb_ratio: S/B ratio at each threshold
        model_name: Name of the model
        savepath: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Signal efficiency
    axes[0].plot(thresholds, signal_eff, 'b-', linewidth=2, label='Signal Efficiency')
    axes[0].set_xlabel('Threshold', fontsize=11)
    axes[0].set_ylabel('Signal Efficiency', fontsize=11)
    axes[0].set_title('Signal Efficiency vs Threshold', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Background rejection
    axes[1].plot(thresholds, bg_reject, 'r-', linewidth=2, label='Background Rejection')
    axes[1].set_xlabel('Threshold', fontsize=11)
    axes[1].set_ylabel('Background Rejection', fontsize=11)
    axes[1].set_title('Background Rejection vs Threshold', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # S/B ratio
    axes[2].plot(thresholds, sb_ratio, 'g-', linewidth=2, label='S/B Ratio')
    axes[2].set_xlabel('Threshold', fontsize=11)
    axes[2].set_ylabel('S/B Ratio', fontsize=11)
    axes[2].set_title('S/B Ratio vs Threshold', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_yscale('log')
    
    plt.suptitle(f'Threshold Analysis - {model_name}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_score_distribution(y_true: np.ndarray, y_pred_proba: np.ndarray,
                           model_name: str = 'Model',
                           savepath: Optional[str] = None) -> None:
    """
    Plot distribution of prediction scores for signal and background.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        model_name: Name of the model
        savepath: Path to save figure
    """
    signal_scores = y_pred_proba[y_true == 1]
    background_scores = y_pred_proba[y_true == 0]
    
    plt.figure(figsize=(10, 6))
    plt.hist(background_scores, bins=50, alpha=0.7, label='Background', density=True)
    plt.hist(signal_scores, bins=50, alpha=0.7, label='Signal', density=True)
    plt.xlabel('Prediction Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'Score Distribution - {model_name}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_comparison(metrics_dict: Dict[str, Dict[str, float]],
                         savepath: Optional[str] = None) -> None:
    """
    Plot comparison of multiple models.
    
    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        savepath: Path to save figure
    """
    models = list(metrics_dict.keys())
    metrics_to_plot = ['auc_roc', 'accuracy', 'precision', 'recall', 'signal_to_background_ratio']
    
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(20, 5))
    
    for i, metric in enumerate(metrics_to_plot):
        values = [metrics_dict[model].get(metric, 0) for model in models]
        axes[i].bar(models, values)
        axes[i].set_title(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        axes[i].set_ylabel('Value', fontsize=10)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Model Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close()
