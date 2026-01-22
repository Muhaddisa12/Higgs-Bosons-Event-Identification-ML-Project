#!/usr/bin/env python3
"""
Data preparation script.

Loads raw HIGGS data, applies preprocessing, and saves train/validation/test splits.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from src.data.loader import load_higgs_data
from src.data.preprocessor import DataPreprocessor
from src.data.feature_engineering import map_features_to_physics_names
from src.utils.helpers import load_config, ensure_dir


def main():
    """Main data preparation pipeline."""
    # Load preprocessing config
    config_path = Path(__file__).parent.parent / "configs" / "preprocessing_config.yml"
    config = load_config(str(config_path))
    
    print("=" * 60)
    print("HIGGS Data Preparation Pipeline")
    print("=" * 60)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config['preprocessing'], 
                                    random_state=config['data']['random_state'])
    
    # Load raw data
    print(f"\n[1/4] Loading raw data from {config['data']['raw_data_path']}...")
    raw_data_path = Path(__file__).parent.parent / config['data']['raw_data_path']
    
    n_samples = config['data'].get('n_samples', None)
    df = load_higgs_data(str(raw_data_path), n_samples=n_samples,
                        random_state=config['data']['random_state'])
    
    print(f"    Loaded {len(df):,} events with {len(df.columns)-1} features")
    
    # Extract features and labels
    print("\n[2/4] Extracting features and labels...")
    y = df['label'].values
    X = df.drop('label', axis=1).values
    
    # Select first 8 features (key physics variables)
    # Based on HIGGS dataset structure
    X = X[:, :8]
    print(f"    Using {X.shape[1]} features")
    
    # Split data
    print("\n[3/4] Splitting data into train/validation/test...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.split_data(X, y)
    
    print(f"    Training set:   {len(X_train):,} events")
    print(f"    Validation set: {len(X_val):,} events")
    print(f"    Test set:       {len(X_test):,} events")
    
    # Apply preprocessing (feature selection, scaling if needed)
    print("\n[4/4] Applying preprocessing...")
    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)
    
    # Save processed data
    output_dir = Path(__file__).parent.parent / config['data']['processed_data_dir']
    ensure_dir(str(output_dir))
    
    print(f"\nSaving processed data to {output_dir}...")
    preprocessor.save_processed_data(X_train, y_train, str(output_dir), 'train')
    preprocessor.save_processed_data(X_val, y_val, str(output_dir), 'validation')
    preprocessor.save_processed_data(X_test, y_test, str(output_dir), 'test')
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print(f"\nProcessed data saved to: {output_dir}")
    print(f"  - train.csv")
    print(f"  - validation.csv")
    print(f"  - test.csv")


if __name__ == "__main__":
    main()
