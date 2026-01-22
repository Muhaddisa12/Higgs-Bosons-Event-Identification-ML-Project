# Higgs Boson Signal Discrimination - Machine Learning Project

A research-grade machine learning pipeline for Higgs boson signal discrimination at the LHC using tree-based ensemble methods (Random Forest, Gradient Boosting, XGBoost).

## 🎯 Project Overview

This project implements a complete machine learning pipeline for discriminating Higgs boson signal events from background events in LHC collision data. The pipeline is designed with research-grade standards, featuring modular architecture, comprehensive evaluation, and full reproducibility.

### Key Features

- **Three ML Models**: Random Forest, Gradient Boosting, and XGBoost
- **Modular Architecture**: Clean separation of data loading, preprocessing, modeling, and evaluation
- **Config-Driven**: All hyperparameters and settings in YAML config files
- **Comprehensive Evaluation**: ROC curves, confusion matrices, feature importance, threshold analysis
- **Full Reproducibility**: Deterministic splits, random seeds, saved models and metadata
- **Research-Grade**: Suitable for academic publication and CERN analysis

## 📁 Project Structure

```
higgs-ml-discrimination/
│
├── README.md
├── LICENSE
├── environment.yml
├── requirements.txt
├── setup.py
│
├── configs/
│   ├── random_forest_config.yml
│   ├── gradient_boosting_config.yml
│   ├── xgboost_config.yml
│   └── preprocessing_config.yml
│
├── data/
│   ├── raw/
│   │   └── HIGGS.csv.gz
│   ├── processed/
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   └── test.csv
│   └── README.md
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_results_visualization.ipynb
│   └── 05_physics_interpretation.ipynb
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── preprocessor.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── base_classifier.py
│   │   ├── random_forest_classifier.py
│   │   ├── gradient_boosting_classifier.py
│   │   └── xgboost_classifier.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── visualizations.py
│   └── utils/
│       └── helpers.py
│
├── scripts/
│   ├── prepare_data.py
│   ├── train_models.py
│   ├── evaluate.py
│   ├── plot_results.py
│   └── run_full_analysis.sh
│
├── models/
│   ├── random_forest_best.pkl
│   ├── gradient_boosting_best.pkl
│   ├── xgboost_best.pkl
│   └── model_metadata.json
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── logs/
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   └── test_models.py
│
└── docs/
    ├── RESULTS_RANDOM_FOREST.md
    ├── RESULTS_GRADIENT_BOOSTING.md
    ├── RESULTS_XGBOOST.md
    ├── RESULTS_FINAL_COMPARISON.md
    ├── physics_background.md
    ├── methodology.md
    └── api_reference.md
```

## 🚀 Quick Start

### 1. Installation

#### Using conda (recommended):
```bash
conda env create -f environment.yml
conda activate higgs-ml
```

#### Using pip:
```bash
pip install -r requirements.txt
```

### 2. Data Preparation

Place the HIGGS dataset in `data/raw/HIGGS.csv.gz` (or update the path in `configs/preprocessing_config.yml`).

Then prepare the data:
```bash
python scripts/prepare_data.py
```

This will:
- Load the raw HIGGS dataset
- Split into train/validation/test sets (64%/16%/20%)
- Apply preprocessing
- Save processed data to `data/processed/`

### 3. Train Models

Train all models:
```bash
python scripts/train_models.py --model all
```

Or train individual models:
```bash
python scripts/train_models.py --model rf    # Random Forest
python scripts/train_models.py --model gb   # Gradient Boosting
python scripts/train_models.py --model xgb   # XGBoost
```

### 4. Evaluate Models

Evaluate all models:
```bash
python scripts/evaluate.py --model all
```

### 5. Generate Plots

Generate all visualizations:
```bash
python scripts/plot_results.py --model all
```

### 6. Run Full Pipeline

Run the complete pipeline (data prep → training → evaluation → plots):
```bash
bash scripts/run_full_analysis.sh
```

## 📊 Dataset

The HIGGS dataset contains 11 million collision events with 28 features. The first column is the class label:
- **1**: Higgs boson signal event
- **0**: Background event

Key features used:
- `m_bb_paper`: Dijet invariant mass (primary Higgs signature)
- `delta_m`: Mass matching quality
- `m_missing`: Missing transverse energy
- `bjet_1_btag`: Leading b-jet b-tagging score
- `bjet_2_btag`: Subleading b-jet b-tagging score
- `gap_forward`: Forward rapidity gap
- `gap_backward`: Backward rapidity gap
- `n_extra_activity`: Extra jet activity

## 🧪 Model Performance

### XGBoost (Champion Model)
- **AUC-ROC**: 0.6942
- **S/B Ratio**: 7.77:1 (at threshold 0.5)
- **Signal Efficiency**: 62.3%
- **Training Time**: ~75 seconds

### Gradient Boosting
- **AUC-ROC**: 0.6936
- **S/B Ratio**: 3.10:1
- **Signal Efficiency**: 96.4%
- **Training Time**: ~180 seconds

### Random Forest
- **AUC-ROC**: 0.6939
- **S/B Ratio**: 1.19:1
- **Signal Efficiency**: 94.2%
- **Training Time**: ~45 seconds

## 📝 Configuration

All model hyperparameters and preprocessing settings are in YAML config files:

- `configs/preprocessing_config.yml`: Data loading and preprocessing
- `configs/random_forest_config.yml`: Random Forest hyperparameters
- `configs/gradient_boosting_config.yml`: Gradient Boosting hyperparameters
- `configs/xgboost_config.yml`: XGBoost hyperparameters

Modify these files to experiment with different settings.

## 🧬 Usage Examples

### Using the Models Programmatically

```python
from src.models import XGBoostClassifierModel
from src.utils.helpers import load_config
from src.data.loader import load_processed_data

# Load config and data
config = load_config('configs/xgboost_config.yml')
X_train, y_train = load_processed_data('data/processed', 'train')

# Initialize and train
model = XGBoostClassifierModel(config['model'], random_state=42)
model.train(X_train, y_train)

# Predict
X_test, y_test = load_processed_data('data/processed', 'test')
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
```

### Loading a Trained Model

```python
from src.models import XGBoostClassifierModel
from src.utils.helpers import load_config

config = load_config('configs/xgboost_config.yml')
model = XGBoostClassifierModel(config['model'], random_state=42)
model.load('models/xgboost_best.pkl')

# Use for prediction
predictions = model.predict(new_data)
```

## 🧪 Testing

Run tests:
```bash
pytest tests/
```

Run specific test file:
```bash
pytest tests/test_models.py
```

## 📚 Documentation

- `docs/methodology.md`: Detailed methodology and approach
- `docs/physics_background.md`: Physics background and context
- `docs/api_reference.md`: API documentation
- `docs/RESULTS_*.md`: Detailed results for each model

## 🔬 Notebooks

Jupyter notebooks for interactive analysis:

1. **01_exploratory_data_analysis.ipynb**: EDA and data exploration
2. **02_feature_engineering.ipynb**: Feature engineering experiments
3. **03_model_training.ipynb**: Model training examples
4. **04_results_visualization.ipynb**: Results visualization
5. **05_physics_interpretation.ipynb**: Physics interpretation

## 🤝 Contributing

This is a research project. For contributions:
1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## 📄 License

See LICENSE file for details.

## 🙏 Citation

If you use this code in your research, please cite:

```
[Your Name] (2025). Machine Learning for Higgs Boson Signal Discrimination 
at the LHC. GitHub: [repository URL]
```

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Status**: ✅ Production Ready  
**Version**: 1.0.0  
**Last Updated**: January 2025
