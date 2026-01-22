#!/usr/bin/env python3
"""
Results visualization script.

Generates plots for model evaluation including ROC curves, confusion matrices,
feature importance, and threshold analysis.
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
from src.evaluation.visualizations import (
    plot_roc_curve,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_threshold_analysis,
    plot_score_distribution
)
from src.evaluation.metrics import calculate_roc_curve
from src.utils.helpers import load_config, ensure_dir


def plot_model_results(model_class, model_name, config_path, data_dir, output_dir):
    """Generate all plots for a model."""
    print(f"\n{'='*60}")
    print(f"Generating plots for {model_name}")
    print(f"{'='*60}")
    
    # Load config
    config = load_config(config_path)
    
    # Load test data
    print("\n[1/3] Loading test data...")
    X_test, y_test = load_processed_data(data_dir, 'test')
    
    # Load model
    print("\n[2/3] Loading trained model...")
    model = model_class(config['model'], random_state=config['model']['random_state'])
    model_path = Path(__file__).parent.parent / config['output']['model_path']
    model.load(str(model_path))
    
    # Get predictions
    print("\n[3/3] Generating predictions...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    threshold = config['evaluation']['threshold']
    y_pred_thresh = (y_pred_proba >= threshold).astype(int)
    
    # Create output directory
    model_output_dir = Path(output_dir) / model_name.lower().replace(' ', '_')
    ensure_dir(str(model_output_dir))
    
    # Generate plots
    print(f"\nGenerating plots...")
    
    # ROC curve
    roc_data = calculate_roc_curve(y_test, y_pred_proba)
    auc = roc_data['tpr'].sum() / len(roc_data['tpr'])  # Approximate AUC
    plot_roc_curve(roc_data['fpr'], roc_data['tpr'], auc, model_name,
                   str(model_output_dir / 'roc_curve.png'))
    print(f"  ✓ ROC curve saved")
    
    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred_thresh, model_name,
                         str(model_output_dir / 'confusion_matrix.png'))
    print(f"  ✓ Confusion matrix saved")
    
    # Feature importance
    feature_importance = model.get_feature_importance()
    if feature_importance is not None:
        feature_names = ['m_bb_paper', 'delta_m', 'm_missing', 'bjet_1_btag',
                        'bjet_2_btag', 'gap_forward', 'gap_backward', 'n_extra_activity']
        plot_feature_importance(feature_importance, feature_names, model_name,
                               str(model_output_dir / 'feature_importance.png'))
        print(f"  ✓ Feature importance saved")
    
    # Threshold analysis
    thresholds = np.array(config['evaluation']['threshold_range'])
    from src.evaluation.metrics import calculate_metrics_at_thresholds
    threshold_metrics = calculate_metrics_at_thresholds(y_test, y_pred_proba, thresholds)
    plot_threshold_analysis(thresholds, threshold_metrics['signal_efficiency'],
                           threshold_metrics['background_rejection'],
                           threshold_metrics['sb_ratio'], model_name,
                           str(model_output_dir / 'threshold_analysis.png'))
    print(f"  ✓ Threshold analysis saved")
    
    # Score distribution
    plot_score_distribution(y_test, y_pred_proba, model_name,
                           str(model_output_dir / 'score_distribution.png'))
    print(f"  ✓ Score distribution saved")
    
    print(f"\nAll plots saved to: {model_output_dir}")


def main():
    """Main plotting pipeline."""
    parser = argparse.ArgumentParser(description='Generate evaluation plots')
    parser.add_argument('--model', type=str, choices=['rf', 'gb', 'xgb', 'all'],
                       default='all', help='Which model to plot')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Directory containing processed data')
    parser.add_argument('--output-dir', type=str, default='outputs/figures',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    data_dir = str(Path(__file__).parent.parent / args.data_dir)
    output_dir = str(Path(__file__).parent.parent / args.output_dir)
    ensure_dir(output_dir)
    
    models_to_plot = []
    if args.model == 'all':
        models_to_plot = ['rf', 'gb', 'xgb']
    else:
        models_to_plot = [args.model]
    
    for model_code in models_to_plot:
        if model_code == 'rf':
            config_path = Path(__file__).parent.parent / "configs" / "random_forest_config.yml"
            plot_model_results(RandomForestClassifierModel, 'Random Forest',
                              str(config_path), data_dir, output_dir)
        
        elif model_code == 'gb':
            config_path = Path(__file__).parent.parent / "configs" / "gradient_boosting_config.yml"
            plot_model_results(GradientBoostingClassifierModel, 'Gradient Boosting',
                              str(config_path), data_dir, output_dir)
        
        elif model_code == 'xgb':
            config_path = Path(__file__).parent.parent / "configs" / "xgboost_config.yml"
            plot_model_results(XGBoostClassifierModel, 'XGBoost',
                              str(config_path), data_dir, output_dir)
    
    print(f"\n{'='*60}")
    print("Plotting Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
