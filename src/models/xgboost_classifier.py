"""
XGBoost classifier implementation for Higgs boson discrimination.
"""

import numpy as np
from typing import Dict, Any, Optional
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

import joblib
from pathlib import Path

# Handle both relative and absolute imports
try:
    from .base_classifier import BaseClassifier
except ImportError:
    import sys
    # Add project root to path (go up from src/models/ to project root)
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.models.base_classifier import BaseClassifier


class XGBoostClassifierModel(BaseClassifier):
    """
    XGBoost classifier for binary classification.
    
    Implements an XGBoost model using the xgboost library with
    configurable hyperparameters and early stopping.
    """
    
    def __init__(self, config: Dict[str, Any], random_state: int = 42):
        """
        Initialize XGBoost classifier.
        
        Args:
            config: Dictionary containing hyperparameters:
                - n_estimators: Number of boosting rounds
                - learning_rate: Step size shrinkage
                - max_depth: Maximum tree depth
                - min_child_weight: Minimum sum of instance weight
                - subsample: Subsample ratio of training instances
                - colsample_bytree: Subsample ratio of columns
                - gamma: Minimum loss reduction for split
                - reg_alpha: L1 regularization
                - reg_lambda: L2 regularization
                - scale_pos_weight: Balancing of positive weights
                - early_stopping_rounds: Early stopping patience
            random_state: Random seed
        """
        super().__init__(config, random_state)
        
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is not installed. Install it with: pip install xgboost"
            )
        
        self.model = xgb.XGBClassifier(
            n_estimators=config.get('n_estimators', 100),
            learning_rate=config.get('learning_rate', 0.1),
            max_depth=config.get('max_depth', 6),
            min_child_weight=config.get('min_child_weight', 1),
            subsample=config.get('subsample', 0.8),
            colsample_bytree=config.get('colsample_bytree', 0.8),
            gamma=config.get('gamma', 0),
            reg_alpha=config.get('reg_alpha', 0),
            reg_lambda=config.get('reg_lambda', 1),
            scale_pos_weight=config.get('scale_pos_weight', 1),
            objective='binary:logistic',
            eval_metric='auc',
            random_state=random_state,
            n_jobs=config.get('n_jobs', -1),
            tree_method=config.get('tree_method', 'hist'),
            verbose=config.get('verbose', 0)
        )
        
        self.early_stopping_rounds = config.get('early_stopping_rounds', 10)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features for early stopping
            y_val: Optional validation labels for early stopping
            
        Returns:
            Dictionary with training metrics
        """
        import time
        start_time = time.time()
        
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        # Handle early stopping for different XGBoost versions
        # In XGBoost 2.0+, early_stopping_rounds is set as model parameter, not passed to fit()
        # In older versions, it's passed to fit()
        if eval_set and self.early_stopping_rounds:
            # Try new API first (XGBoost 2.0+): set as model parameter
            try:
                self.model.set_params(early_stopping_rounds=self.early_stopping_rounds)
                self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
            except (TypeError, ValueError):
                # Fall back to old API (XGBoost < 2.0): pass to fit()
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose=False
                )
        else:
            # No early stopping
            if eval_set:
                self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
            else:
                self.model.fit(X_train, y_train, verbose=False)
        
        training_time = time.time() - start_time
        self.is_trained = True
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        train_proba = self.model.predict_proba(X_train)[:, 1]
        train_acc = np.mean(train_pred == y_train)
        
        metrics = {
            'training_time': training_time,
            'train_accuracy': train_acc,
            'n_estimators': self.model.n_estimators,
            'n_features': X_train.shape[1]
        }
        
        # Get evaluation history if available
        if hasattr(self.model, 'evals_result_') and self.model.evals_result_:
            evals_result = self.model.evals_result_
            if 'validation_0' in evals_result:
                metrics['train_auc_history'] = evals_result['validation_0']['auc']
            if 'validation_1' in evals_result:
                metrics['val_auc_history'] = evals_result['validation_1']['auc']
        
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_proba = self.model.predict_proba(X_val)[:, 1]
            val_acc = np.mean(val_pred == y_val)
            metrics['val_accuracy'] = val_acc
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)
    
    def save(self, filepath: str) -> None:
        """Save model to disk."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        # Save in both formats for compatibility
        joblib.dump(self.model, filepath)
        # Also save in XGBoost native format
        json_path = str(filepath).replace('.pkl', '.json')
        self.model.save_model(json_path)
    
    def load(self, filepath: str) -> None:
        """Load model from disk."""
        # Try loading from pickle first, then JSON
        try:
            self.model = joblib.load(filepath)
        except:
            json_path = str(filepath).replace('.pkl', '.json')
            self.model = xgb.XGBClassifier()
            self.model.load_model(json_path)
        self.is_trained = True
