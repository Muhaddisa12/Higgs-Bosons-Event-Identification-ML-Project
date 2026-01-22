"""
Random Forest classifier implementation for Higgs boson discrimination.
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier
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


class RandomForestClassifierModel(BaseClassifier):
    """
    Random Forest classifier for binary classification.
    
    Implements a Random Forest model using scikit-learn's RandomForestClassifier
    with configurable hyperparameters.
    """
    
    def __init__(self, config: Dict[str, Any], random_state: int = 42):
        """
        Initialize Random Forest classifier.
        
        Args:
            config: Dictionary containing hyperparameters:
                - n_estimators: Number of trees
                - max_depth: Maximum tree depth
                - min_samples_split: Minimum samples to split
                - min_samples_leaf: Minimum samples at leaf
                - max_features: Number of features for split
                - bootstrap: Whether to use bootstrap sampling
                - class_weight: Class weight strategy
            random_state: Random seed
        """
        super().__init__(config, random_state)
        
        self.model = RandomForestClassifier(
            n_estimators=config.get('n_estimators', 100),
            max_depth=config.get('max_depth', 20),
            min_samples_split=config.get('min_samples_split', 2),
            min_samples_leaf=config.get('min_samples_leaf', 1),
            max_features=config.get('max_features', 'sqrt'),
            bootstrap=config.get('bootstrap', True),
            class_weight=config.get('class_weight', 'balanced'),
            random_state=random_state,
            n_jobs=config.get('n_jobs', -1),
            verbose=config.get('verbose', 0)
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features (not used for RF)
            y_val: Optional validation labels (not used for RF)
            
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
