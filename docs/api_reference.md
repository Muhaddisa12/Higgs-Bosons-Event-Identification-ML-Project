# API Reference

## Data Loading

### `load_higgs_data(filepath, n_samples=None, random_state=42)`

Load HIGGS dataset from CSV file.

**Parameters:**
- `filepath` (str): Path to HIGGS.csv or HIGGS.csv.gz
- `n_samples` (int, optional): Number of samples to load
- `random_state` (int): Random seed

**Returns:**
- `pd.DataFrame`: DataFrame with label and features

### `load_processed_data(data_dir, split='train')`

Load processed data splits.

**Parameters:**
- `data_dir` (str): Directory containing processed data
- `split` (str): 'train', 'validation', or 'test'

**Returns:**
- `Tuple[np.ndarray, np.ndarray]`: (X, y) arrays

## Models

### `RandomForestClassifierModel(config, random_state=42)`

Random Forest classifier.

**Methods:**
- `train(X_train, y_train, X_val=None, y_val=None)`: Train model
- `predict(X)`: Predict class labels
- `predict_proba(X)`: Predict probabilities
- `save(filepath)`: Save model
- `load(filepath)`: Load model
- `get_feature_importance()`: Get feature importances

### `GradientBoostingClassifierModel(config, random_state=42)`

Gradient Boosting classifier.

### `XGBoostClassifierModel(config, random_state=42)`

XGBoost classifier.

## Evaluation

### `calculate_classification_metrics(y_true, y_pred, y_pred_proba=None, threshold=0.5)`

Calculate comprehensive classification metrics.

**Returns:**
- `Dict[str, float]`: Dictionary of metrics

### `plot_roc_curve(fpr, tpr, auc, label, savepath=None)`

Plot ROC curve.

### `plot_confusion_matrix(y_true, y_pred, model_name, savepath=None)`

Plot confusion matrix.

## Utilities

### `load_config(config_path)`

Load YAML configuration file.

### `save_metadata(metadata, filepath)`

Save model metadata to JSON.
