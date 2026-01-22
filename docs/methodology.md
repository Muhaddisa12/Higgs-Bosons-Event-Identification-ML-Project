# Methodology

## Overview

This document describes the methodology used for Higgs boson signal discrimination using machine learning.

## Data Preprocessing

1. **Data Loading**: Load HIGGS dataset (11M events, 28 features)
2. **Feature Selection**: Use first 8 features (key physics variables)
3. **Data Splitting**: 64% train, 16% validation, 20% test (stratified)
4. **Scaling**: Not applied (tree-based models don't require it)

## Model Architecture

### Random Forest
- 100 trees, max depth 20
- Balanced class weights
- Parallel training (n_jobs=-1)

### Gradient Boosting
- 100 boosting stages, learning rate 0.1
- Max depth 3, early stopping enabled
- Sequential training

### XGBoost
- 100 boosting rounds, learning rate 0.1
- Max depth 6, L2 regularization (λ=1)
- Histogram-based tree method
- Early stopping with 10 rounds patience

## Training Procedure

1. Train on training set (320k events)
2. Monitor on validation set (80k events)
3. Early stopping if applicable
4. Evaluate on test set (100k events)

## Evaluation Metrics

- **AUC-ROC**: Overall discrimination ability
- **Accuracy**: Overall classification accuracy
- **Precision**: Signal purity
- **Recall**: Signal efficiency
- **S/B Ratio**: Signal-to-background ratio
- **F1-Score**: Harmonic mean of precision and recall

## Threshold Optimization

Models output probabilities. Classification threshold is optimized for:
- Maximum S/B ratio (precision measurements)
- Maximum signal efficiency (discovery searches)
- Balanced performance (general analysis)

## Cross-Validation

5-fold cross-validation used for:
- Model selection
- Hyperparameter tuning
- Performance estimation

## Reproducibility

- Fixed random seeds (42)
- Deterministic data splits
- Saved model checkpoints
- Complete metadata logging
