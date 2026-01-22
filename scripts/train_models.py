#!/usr/bin/env python3
"""
Model training script.

Trains Random Forest, Gradient Boosting, and XGBoost models on the HIGGS dataset.
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.data.loader import load_processed_data
from src.models import (
    RandomForestClassifierModel,
    GradientBoostingClassifierModel,
    XGBoostClassifierModel
)
from src.evaluation.metrics import calculate_classification_metrics
from src.utils.helpers import load_config, save_metadata, ensure_dir


def train_model(model_class, model_name, config_path, data_dir):
    """Train a single model."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Load config
    config = load_config(config_path)
    
    # Load data
    print("\n[1/3] Loading data...")
    X_train, y_train = load_processed_data(data_dir, 'train')
    X_val, y_val = load_processed_data(data_dir, 'validation')
    
    print(f"    Training:   {len(X_train):,} events")
    print(f"    Validation: {len(X_val):,} events")
    
    # Initialize model
    print("\n[2/3] Initializing model...")
    model = model_class(config['model'], random_state=config['model']['random_state'])
    
    # Train model
    print("\n[3/3] Training model...")
    train_metrics = model.train(X_train, y_train, X_val, y_val)
    
    print(f"\nTraining completed in {train_metrics['training_time']:.2f} seconds")
    print(f"Training accuracy: {train_metrics['train_accuracy']:.4f}")
    if 'val_accuracy' in train_metrics:
        print(f"Validation accuracy: {train_metrics['val_accuracy']:.4f}")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)
    
    threshold = config['evaluation']['threshold']
    metrics = calculate_classification_metrics(y_val, y_pred, y_pred_proba, threshold)
    
    print(f"\nValidation Metrics:")
    print(f"  AUC-ROC:              {metrics.get('auc_roc', 'N/A'):.4f}")
    print(f"  Accuracy:             {metrics['accuracy']:.4f}")
    print(f"  Precision:            {metrics['precision']:.4f}")
    print(f"  Recall (Signal Eff.): {metrics['recall']:.4f}")
    print(f"  S/B Ratio:            {metrics['signal_to_background_ratio']:.4f}")
    
    # Save model
    model_path = Path(__file__).parent.parent / config['output']['model_path']
    ensure_dir(str(model_path.parent))
    model.save(str(model_path))
    print(f"\nModel saved to: {model_path}")
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'config': config,
        'training_metrics': train_metrics,
        'validation_metrics': metrics,
        'feature_names': ['m_bb_paper', 'delta_m', 'm_missing', 'bjet_1_btag',
                         'bjet_2_btag', 'gap_forward', 'gap_backward', 'n_extra_activity'],
        'training_timestamp': datetime.now().isoformat(),
        'n_features': X_train.shape[1],
        'n_train_samples': len(X_train),
        'n_val_samples': len(X_val)
    }
    
    metadata_path = Path(__file__).parent.parent / config['output']['metadata_path']
    save_metadata(metadata, str(metadata_path))
    
    return model, metrics


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train ML models for Higgs discrimination')
    parser.add_argument('--model', type=str, choices=['rf', 'gb', 'xgb', 'all'],
                       default='all', help='Which model to train')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Directory containing processed data')
    
    args = parser.parse_args()
    
    data_dir = str(Path(__file__).parent.parent / args.data_dir)
    
    models_to_train = []
    if args.model == 'all':
        models_to_train = ['rf', 'gb', 'xgb']
    else:
        models_to_train = [args.model]
    
    results = {}
    
    for model_code in models_to_train:
        if model_code == 'rf':
            config_path = Path(__file__).parent.parent / "configs" / "random_forest_config.yml"
            model, metrics = train_model(RandomForestClassifierModel, 'Random Forest',
                                        str(config_path), data_dir)
            results['random_forest'] = metrics
        
        elif model_code == 'gb':
            config_path = Path(__file__).parent.parent / "configs" / "gradient_boosting_config.yml"
            model, metrics = train_model(GradientBoostingClassifierModel, 'Gradient Boosting',
                                        str(config_path), data_dir)
            results['gradient_boosting'] = metrics
        
        elif model_code == 'xgb':
            config_path = Path(__file__).parent.parent / "configs" / "xgboost_config.yml"
            model, metrics = train_model(XGBoostClassifierModel, 'XGBoost',
                                        str(config_path), data_dir)
            results['xgboost'] = metrics
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print("\nSummary:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  AUC-ROC: {metrics.get('auc_roc', 'N/A'):.4f}")
        print(f"  S/B Ratio: {metrics['signal_to_background_ratio']:.4f}")


if __name__ == "__main__":
    main()
