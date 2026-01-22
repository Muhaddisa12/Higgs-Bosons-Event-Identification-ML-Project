# Installation Notes

## ⚠️ Python Version Issue

You are currently using **Python 3.15.0a3** (alpha version), which does not have pre-built wheels for many scientific packages like numpy, pandas, scikit-learn, etc.

## Recommended Solutions

### Option 1: Use Stable Python Version (Recommended)

Install Python 3.8, 3.9, 3.10, 3.11, or 3.12 (stable releases) and use that instead:

1. Download Python from https://www.python.org/downloads/
2. Install Python 3.11 or 3.12 (recommended)
3. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Use Conda (Recommended for Scientific Computing)

Conda can handle binary dependencies better:

```bash
conda create -n higgs-ml python=3.11
conda activate higgs-ml
conda install -c conda-forge numpy pandas scikit-learn matplotlib seaborn pyyaml joblib pytest jupyter notebook
conda install -c conda-forge xgboost
```

Or use the provided environment.yml:
```bash
conda env create -f environment.yml
conda activate higgs-ml
```

### Option 3: Build from Source (Advanced)

If you must use Python 3.15 alpha, you'll need to:
1. Install a newer GCC compiler (>= 9.3) or use Visual Studio Build Tools
2. Build numpy and other packages from source

This is not recommended for most users.

## Currently Installed Packages

The following packages are already installed:
- PyYAML (6.0.3) ✅
- setuptools, wheel, packaging ✅
- pytest, jupyter, notebook (may install) ✅

## Missing Packages (Require Stable Python)

- numpy (requires compilation or stable Python)
- pandas (depends on numpy)
- scikit-learn (depends on numpy)
- matplotlib (depends on numpy)
- seaborn (depends on numpy, matplotlib)
- xgboost (may work, depends on numpy)

## Quick Test

To check if packages are available for your Python version:
```bash
python -c "import sys; print(f'Python {sys.version}')"
python -m pip index versions numpy
```

## Next Steps

1. **Switch to Python 3.11 or 3.12** (easiest solution)
2. **Or use conda** (best for scientific computing)
3. **Then run**: `pip install -r requirements.txt`
