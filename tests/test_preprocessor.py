"""
Tests for data preprocessing module.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessor import DataPreprocessor


def test_data_preprocessor_split():
    """Test data splitting."""
    config = {
        'train_ratio': 0.6,
        'val_ratio': 0.2,
        'test_ratio': 0.2,
        'scale_features': False
    }
    
    preprocessor = DataPreprocessor(config, random_state=42)
    
    # Create dummy data
    X = np.random.rand(1000, 8)
    y = np.random.randint(0, 2, 1000)
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.split_data(X, y)
    
    assert len(X_train) + len(X_val) + len(X_test) == 1000
    assert len(y_train) + len(y_val) + len(y_test) == 1000
    assert abs(len(X_train) / 1000 - 0.6) < 0.05  # Allow small tolerance


def test_feature_selection():
    """Test feature selection."""
    config = {
        'feature_indices': [0, 1, 2],
        'scale_features': False
    }
    
    preprocessor = DataPreprocessor(config, random_state=42)
    
    X = np.random.rand(100, 8)
    X_selected = preprocessor.select_features(X)
    
    assert X_selected.shape[1] == 3


def test_scaling():
    """Test feature scaling."""
    config = {
        'scale_features': True,
        'scaler_type': 'standard'
    }
    
    preprocessor = DataPreprocessor(config, random_state=42)
    
    X_train = np.random.rand(100, 8) * 100
    X_test = np.random.rand(50, 8) * 100
    
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    # Check that scaling was applied (mean should be ~0 for standard scaler)
    assert np.abs(X_train_scaled.mean()) < 0.1
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
