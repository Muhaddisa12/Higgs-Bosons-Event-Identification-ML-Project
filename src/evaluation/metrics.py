"""
Evaluation metrics for Higgs boson discrimination models.
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, precision_recall_curve
)


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                     y_pred_proba: Optional[np.ndarray] = None,
                                     threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels (or probabilities if threshold is used)
        y_pred_proba: Predicted probabilities (optional)
        threshold: Classification threshold (if y_pred_proba provided)
        
    Returns:
        Dictionary of metrics
    """
    # Convert probabilities to predictions if needed
    if y_pred_proba is not None:
        y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Derived metrics
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Signal efficiency
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Background rejection
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    # Signal-to-background ratio
    sb_ratio = tp / fp if fp > 0 else np.inf
    
    # AUC-ROC if probabilities provided
    auc = None
    if y_pred_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_pred_proba)
        except Exception:
            auc = None
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positive_rate': tpr,  # Signal efficiency
        'true_negative_rate': tnr,   # Background rejection
        'false_positive_rate': fpr,
        'false_negative_rate': fnr,
        'signal_to_background_ratio': sb_ratio,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'events_passing_selection': int(tp + fp)
    }
    
    if auc is not None:
        metrics['auc_roc'] = auc
    
    return metrics


def calculate_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calculate ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        Dictionary with 'fpr', 'tpr', 'thresholds'
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    }


def calculate_precision_recall_curve(y_true: np.ndarray, 
                                     y_pred_proba: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calculate precision-recall curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        Dictionary with 'precision', 'recall', 'thresholds'
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    return {
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds
    }


def calculate_metrics_at_thresholds(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   thresholds: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calculate metrics at multiple thresholds.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        thresholds: Array of threshold values
        
    Returns:
        Dictionary with arrays of metrics for each threshold
    """
    results = {
        'thresholds': thresholds,
        'signal_efficiency': [],
        'background_rejection': [],
        'sb_ratio': [],
        'precision': [],
        'events_passing': []
    }
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        metrics = calculate_classification_metrics(y_true, y_pred)
        
        results['signal_efficiency'].append(metrics['true_positive_rate'])
        results['background_rejection'].append(metrics['true_negative_rate'])
        results['sb_ratio'].append(metrics['signal_to_background_ratio'])
        results['precision'].append(metrics['precision'])
        results['events_passing'].append(metrics['events_passing_selection'])
    
    # Convert to numpy arrays
    for key in results:
        if key != 'thresholds':
            results[key] = np.array(results[key])
    
    return results
