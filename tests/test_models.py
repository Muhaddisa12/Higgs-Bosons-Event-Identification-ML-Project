"""
Tests for model implementations.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    RandomForestClassifierModel,
    GradientBoostingClassifierModel
)


def test_random_forest_training():
    """Test Random Forest model training."""
    config = {
        'n_estimators': 10,
        'max_depth': 5,
        'random_state': 42
    }
    
    model = RandomForestClassifierModel(config, random_state=42)
    
    # Create dummy data
    X_train = np.random.rand(100, 8)
    y_train = np.random.randint(0, 2, 100)
    
    # Train
    metrics = model.train(X_train, y_train)
    
    assert model.is_trained
    assert 'training_time' in metrics
    assert metrics['train_accuracy'] > 0


def test_random_forest_prediction():
    """Test Random Forest prediction."""
    config = {
        'n_estimators': 10,
        'max_depth': 5,
        'random_state': 42
    }
    
    model = RandomForestClassifierModel(config, random_state=42)
    
    X_train = np.random.rand(100, 8)
    y_train = np.random.randint(0, 2, 100)
    
    model.train(X_train, y_train)
    
    # Predict
    X_test = np.random.rand(50, 8)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    assert len(y_pred) == 50
    assert y_proba.shape == (50, 2)
    assert np.all((y_pred == 0) | (y_pred == 1))


def test_gradient_boosting_training():
    """Test Gradient Boosting model training."""
    config = {
        'n_estimators': 10,
        'learning_rate': 0.1,
        'max_depth': 3,
        'random_state': 42
    }
    
    model = GradientBoostingClassifierModel(config, random_state=42)
    
    X_train = np.random.rand(100, 8)
    y_train = np.random.randint(0, 2, 100)
    
    metrics = model.train(X_train, y_train)
    
    assert model.is_trained
    assert 'training_time' in metrics


def test_model_save_load():
    """Test model saving and loading."""
    config = {
        'n_estimators': 10,
        'max_depth': 5,
        'random_state': 42
    }
    
    model = RandomForestClassifierModel(config, random_state=42)
    
    X_train = np.random.rand(100, 8)
    y_train = np.random.randint(0, 2, 100)
    
    model.train(X_train, y_train)
    
    # Save
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        temp_path = f.name
    
    try:
        model.save(temp_path)
        assert os.path.exists(temp_path)
        
        # Load
        model2 = RandomForestClassifierModel(config, random_state=42)
        model2.load(temp_path)
        
        assert model2.is_trained
        
        # Test prediction consistency
        X_test = np.random.rand(10, 8)
        pred1 = model.predict(X_test)
        pred2 = model2.predict(X_test)
        
        assert np.array_equal(pred1, pred2)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
