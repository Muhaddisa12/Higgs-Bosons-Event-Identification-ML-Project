"""
Tests for data loading module.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_higgs_data, get_feature_names


def test_get_feature_names():
    """Test feature name retrieval."""
    names = get_feature_names()
    assert len(names) == 8
    assert 'm_bb_paper' in names
    assert 'bjet_1_btag' in names


def test_load_higgs_data(tmp_path):
    """Test loading HIGGS data."""
    # Create a dummy CSV file for testing
    dummy_data = np.random.rand(100, 29)
    dummy_data[:, 0] = np.random.randint(0, 2, 100)  # Labels
    
    df = pd.DataFrame(dummy_data)
    df.columns = ['label'] + [f'feature_{i}' for i in range(1, 29)]
    
    test_file = tmp_path / "test_higgs.csv"
    df.to_csv(test_file, index=False, header=False)
    
    # Test loading
    loaded_df = load_higgs_data(str(test_file), n_samples=50, random_state=42)
    
    assert len(loaded_df) == 50
    assert 'label' in loaded_df.columns
    assert len(loaded_df.columns) == 29
