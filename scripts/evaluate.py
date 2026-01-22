#!/usr/bin/env python3
"""
Model evaluation script.

Evaluates trained models on test set and generates comprehensive metrics.
"""

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.data.loader import load_processed_data
from src.models import (
    RandomForestClassifierModel,
    GradientBoostingClassifierModel,
    XGBoostClassifierModel
)
from src.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_roc_curve,
    calculate_metrics_at_thresholds
)
from src.utils.helpers import load_config, load_metadata


def evaluate_model(model_class, model_name, config_path, data_dir):
    """Evaluate a trained model."""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")
    
    # Load config
    config = load_config(config_path)
    
    # Load test data
    print("\n[1/3] Loading test data...")
    X_test, y_test = load_processed_data(data_dir, 'test')
    print(f"    Test set: {len(X_test):,} events")
    
    # Load model
    print("\n[2/3] Loading trained model...")
    model = model_class(config['model'], random_state=config['model']['random_state'])
    model_path = Path(__file__).parent.parent / config['output']['model_path']
    model.load(str(model_path))
    print(f"    Model loaded from: {model_path}")
    
    # Evaluate
    print("\n[3/3] Evaluating model...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    threshold = config['evaluation']['threshold']
    metrics = calculate_classification_metrics(y_test, y_pred, y_pred_proba, threshold)
    
    # Calculate ROC curve
    if config['evaluation']['calculate_roc']:
        roc_data = calculate_roc_curve(y_test, y_pred_proba)
        metrics['roc_curve'] = roc_data
    
    # Calculate metrics at different thresholds
    thresholds = np.array(config['evaluation']['threshold_range'])
    threshold_metrics = calculate_metrics_at_thresholds(y_test, y_pred_proba, thresholds)
    metrics['threshold_analysis'] = threshold_metrics
    
    # Print results
    print(f"\n{'='*60}")
    print("Test Set Results")
    print(f"{'='*60}")
    print(f"\nOverall Metrics:")
    print(f"  AUC-ROC:              {metrics.get('auc_roc', 'N/A'):.4f}")
    print(f"  Accuracy:             {metrics['accuracy']:.4f}")
    print(f"  Precision:            {metrics['precision']:.4f}")
    print(f"  Recall (Signal Eff.): {metrics['recall']:.4f}")
    print(f"  F1-Score:            {metrics['f1_score']:.4f}")
    print(f"  S/B Ratio:            {metrics['signal_to_background_ratio']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {metrics['true_positives']:,}")
    print(f"  True Negatives:  {metrics['true_negatives']:,}")
    print(f"  False Positives: {metrics['false_positives']:,}")
    print(f"  False Negatives: {metrics['false_negatives']:,}")
    
    print(f"\nThreshold Analysis (S/B Ratio):")
    for i, thresh in enumerate(thresholds):
        sb = threshold_metrics['sb_ratio'][i]
        eff = threshold_metrics['signal_efficiency'][i]
        print(f"  Threshold {thresh:.1f}: S/B = {sb:.4f}, Signal Eff. = {eff:.4f}")
    
    # Feature importance
    feature_importance = model.get_feature_importance()
    if feature_importance is not None:
        feature_names = ['m_bb_paper', 'delta_m', 'm_missing', 'bjet_1_btag',
                        'bjet_2_btag', 'gap_forward', 'gap_backward', 'n_extra_activity']
        print(f"\nFeature Importance:")
        indices = np.argsort(feature_importance)[::-1]
        for idx in indices[:5]:  # Top 5
            print(f"  {feature_names[idx]}: {feature_importance[idx]:.4f}")
    
    return metrics


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--model', type=str, choices=['rf', 'gb', 'xgb', 'all'],
                       default='all', help='Which model to evaluate')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Directory containing processed data')
    
    args = parser.parse_args()
    
    data_dir = str(Path(__file__).parent.parent / args.data_dir)
    
    models_to_eval = []
    if args.model == 'all':
        models_to_eval = ['rf', 'gb', 'xgb']
    else:
        models_to_eval = [args.model]
    
    results = {}
    
    for model_code in models_to_eval:
        if model_code == 'rf':
            config_path = Path(__file__).parent.parent / "configs" / "random_forest_config.yml"
            metrics = evaluate_model(RandomForestClassifierModel, 'Random Forest',
                                     str(config_path), data_dir)
            results['random_forest'] = metrics
        
        elif model_code == 'gb':
            config_path = Path(__file__).parent.parent / "configs" / "gradient_boosting_config.yml"
            metrics = evaluate_model(GradientBoostingClassifierModel, 'Gradient Boosting',
                                    str(config_path), data_dir)
            results['gradient_boosting'] = metrics
        
        elif model_code == 'xgb':
            config_path = Path(__file__).parent.parent / "configs" / "xgboost_config.yml"
            metrics = evaluate_model(XGBoostClassifierModel, 'XGBoost',
                                     str(config_path), data_dir)
            results['xgboost'] = metrics
    
    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
