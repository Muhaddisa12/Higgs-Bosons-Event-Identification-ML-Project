#!/bin/bash
# Full analysis pipeline script
# Runs complete data preparation, training, evaluation, and visualization

set -e  # Exit on error

echo "=========================================="
echo "HIGGS ML Discrimination - Full Pipeline"
echo "=========================================="

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_DIR"

# Step 1: Prepare data
echo ""
echo "Step 1: Preparing data..."
python scripts/prepare_data.py

# Step 2: Train all models
echo ""
echo "Step 2: Training models..."
python scripts/train_models.py --model all

# Step 3: Evaluate models
echo ""
echo "Step 3: Evaluating models..."
python scripts/evaluate.py --model all

# Step 4: Generate plots
echo ""
echo "Step 4: Generating plots..."
python scripts/plot_results.py --model all

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Models: models/"
echo "  - Figures: outputs/figures/"
echo "  - Metadata: models/model_metadata.json"
