"""
Setup script for Higgs ML Discrimination project.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="higgs-ml-discrimination",
    version="1.0.0",
    author="CERN Research Team",
    description="Machine learning pipeline for Higgs boson signal discrimination",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/higgs-ml-discrimination",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pyyaml>=5.4.0",
        "joblib>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
        ],
    },
)
