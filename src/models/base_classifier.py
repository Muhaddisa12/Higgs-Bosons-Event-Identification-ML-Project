"""
Base classifier abstract class for all ML models.

This module defines the abstract base class that all model implementations
must inherit from, ensuring consistent interface across different algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
from pathlib import Path


class BaseClassifier(ABC):
    """
    Abstract base class for all classifier models.
    
    All model implementations must inherit from this class and implement
    the required methods: train, predict, predict_proba, save, and load.
    """
    
    def __init__(self, config: Dict[str, Any], random_state: int = 42):
        """
        Initialize the base classifier.
        
        Args:
            config: Dictionary containing model hyperparameters
            random_state: Random seed for reproducibility
        """
        self.config = config
        self.random_state = random_state
        self.model = None
        self.is_trained = False
        self.feature_names = None
        
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,)
            X_val: Optional validation features
            y_val: Optional validation labels
            
        Returns:
            Dictionary containing training metrics and history
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input data.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Predicted class labels (n_samples,)
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input data.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Predicted probabilities (n_samples, n_classes)
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path where model should be saved
        """
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to saved model file
        """
        pass
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores if available.
        
        Returns:
            Array of feature importances or None if not available
        """
        if self.model is None or not self.is_trained:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model metadata
        """
        return {
            'model_type': self.__class__.__name__,
            'is_trained': self.is_trained,
            'config': self.config,
            'random_state': self.random_state,
            'has_feature_importance': self.get_feature_importance() is not None
        }
