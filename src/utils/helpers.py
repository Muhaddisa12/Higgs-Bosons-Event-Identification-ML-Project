"""
Helper utilities for the project.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_metadata(metadata: Dict[str, Any], filepath: str) -> None:
    """
    Save model metadata to JSON file.
    
    Args:
        metadata: Dictionary containing metadata
        filepath: Path to save metadata
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_metadata(filepath: str) -> Dict[str, Any]:
    """
    Load model metadata from JSON file.
    
    Args:
        filepath: Path to metadata file
        
    Returns:
        Dictionary containing metadata
    """
    with open(filepath, 'r') as f:
        metadata = json.load(f)
    return metadata


def ensure_dir(dirpath: str) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        dirpath: Directory path
        
    Returns:
        Path object
    """
    path = Path(dirpath)
    path.mkdir(parents=True, exist_ok=True)
    return path
