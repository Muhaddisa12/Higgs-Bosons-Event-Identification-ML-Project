"""
Gradient Boosting classifier implementation for Higgs boson discrimination.
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.ensemble import GradientBoostingClassifier
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


class GradientBoostingClassifierModel(BaseClassifier):
    """
    Gradient Boosting classifier for binary classification.
    
    Implements a Gradient Boosting model using scikit-learn's
    GradientBoostingClassifier with configurable hyperparameters.
    """
    
    def __init__(self, config: Dict[str, Any], random_state: int = 42):
        """
        Initialize Gradient Boosting classifier.
        
        Args:
            config: Dictionary containing hyperparameters:
                - n_estimators: Number of boosting stages
                - learning_rate: Shrinking factor
                - max_depth: Maximum tree depth
                - min_samples_split: Minimum samples to split
                - min_samples_leaf: Minimum samples at leaf
                - subsample: Fraction of samples for fitting
                - max_features: Number of features for split
                - validation_fraction: Fraction for early stopping
                - n_iter_no_change: Early stopping patience
            random_state: Random seed
        """
        super().__init__(config, random_state)
        
        self.model = GradientBoostingClassifier(
            n_estimators=config.get('n_estimators', 100),
            learning_rate=config.get('learning_rate', 0.1),
            max_depth=config.get('max_depth', 3),
            min_samples_split=config.get('min_samples_split', 2),
            min_samples_leaf=config.get('min_samples_leaf', 1),
            subsample=config.get('subsample', 1.0),
            max_features=config.get('max_features', None),
            validation_fraction=config.get('validation_fraction', 0.2),
            n_iter_no_change=config.get('n_iter_no_change', 10),
            random_state=random_state,
            verbose=config.get('verbose', 0)
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the Gradient Boosting model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features (not used, uses internal split)
            y_val: Optional validation labels (not used, uses internal split)
            
        Returns:
            Dictionary with training metrics
        """
        import time
        start_time = time.time()
        
        self.model.fit(X_train, y_train)
        
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
        
        # Get training history if available
        if hasattr(self.model, 'train_score_'):
            metrics['train_scores'] = self.model.train_score_.tolist()
        
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
        joblib.dump(self.model, filepath)
    
    def load(self, filepath: str) -> None:
        """Load model from disk."""
        self.model = joblib.load(filepath)
        self.is_trained = True


if __name__ == "__main__":
    # This module is not meant to be run directly
    # Use it via: from src.models import GradientBoostingClassifierModel
    print("This module is part of the higgs-ml-discrimination package.")
    print("Import it using: from src.models import GradientBoostingClassifierModel")
    print("\nExample usage:")
    print("  from src.models import GradientBoostingClassifierModel")
    print("  from src.utils.helpers import load_config")
    print("  config = load_config('configs/gradient_boosting_config.yml')")
    print("  model = GradientBoostingClassifierModel(config['model'])")
