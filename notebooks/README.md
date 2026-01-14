# Notebooks Guide

## Overview

This directory contains Jupyter notebooks for testing, training, evaluation, and analysis of the reasoning distillation project.

## Notebooks

### Core Testing & Training

1. **01_data_exploration.ipynb**

- Explore e-SNLI dataset
- Analyze data distribution and quality
- Visualize explanation patterns

2. **02_preprocessing_and_datasets.ipynb**

   - Test preprocessing pipeline
   - Create and validate datasets
   - Verify tokenization and formatting

3. **03_model_testing.ipynb**

   - Test FLAN-T5 student models
   - Verify forward pass and generation
   - Test DatasetTeacher
   - Compare model sizes (small, base)

4. **04_training_loop_testing.ipynb**
   - Test distillation loss computation
   - Validate training loop
   - Test checkpointing and resume
   - Analyze optimizer state

### Evaluation & Analysis

5. **05_evaluation_testing.ipynb** ⭐ **ENHANCED**

   - Test individual metrics (Accuracy, ROUGE, BERTScore, Faithfulness)
   - Full evaluation pipeline
   - Error analysis with confusion matrix
   - Per-label evaluation
   - **Batch evaluation** (compare multiple models)
   - **Model Size Degradation Analysis** (NEW)
     - Compare small, base, large FLAN-T5
     - Performance vs efficiency trade-offs
     - Compression ratio analysis
     - Speedup vs accuracy retention

6. **06_ablation_studies.ipynb** ⭐ **NEW**
   - **Ablation Study 1**: Label Smoothing (0.0, 0.1, 0.2)
   - **Ablation Study 2**: Training Data Size (10%, 50%, 100%)
   - **Ablation Study 3**: Generation Temperature (0.5, 0.7, 1.0, 1.2)
   - Systematic hyperparameter analysis
   - Optimal configuration recommendations

## Execution Order

### For Initial Setup & Testing

```
01 → 02 → 03 → 04 → 05
```

### For Research & Analysis

```
06 (Ablation Studies) → Train with optimal params → 05 (Degradation Analysis)
```

## Key Features

### Notebook 05 - Degradation Analysis (Section 10)

Provides systematic analysis of model size impact:

- **Metrics Tracked**:

  - Label accuracy degradation
  - ROUGE score degradation
  - Faithfulness retention
  - Inference speed changes
  - Memory footprint

- **Visualizations**:

  - 6 comprehensive plots showing degradation curves
  - Trade-off scatter plots (compression vs performance)
  - Efficiency analysis (accuracy per parameter)

- **Outputs**:
  - `model_size_comparison.csv` - Full metrics for all model sizes
  - `tradeoff_analysis.csv` - Compression/speedup trade-offs

### Notebook 06 - Ablation Studies

Systematic hyperparameter testing:

- **Study 1: Label Smoothing**
  - Tests: 0.0, 0.1, 0.2
  - Impact on accuracy, ROUGE, faithfulness
- **Study 2: Training Data Size**

  - Tests: 10%, 50%, 100% of data
  - Data efficiency analysis
  - Performance scaling curves

- **Study 3: Generation Temperature**

  - Tests: 0.5, 0.7, 1.0, 1.2
  - Impact on generation quality
  - Sensitivity analysis

- **Outputs**:
  - `label_smoothing_results.csv`
  - `data_size_results.csv`
  - `temperature_results.csv`

## Usage Examples

### Run Degradation Analysis

```python
# In notebook 05, section 10
# Automatically compares small, base, large
# Generates comprehensive trade-off visualizations
```

### Run Ablation Studies

```python
# In notebook 06
# Tests multiple hyperparameter configurations
# Provides optimal settings recommendations
```

### Compare Multiple Trained Models

```python
# In notebook 05, section 9
batch_evaluator = BatchEvaluator(output_dir="../experiments/comparison")

# Add models
batch_evaluator.add_evaluation("t5-small-baseline", evaluator_small_baseline, test_loader)
batch_evaluator.add_evaluation("t5-small-distilled", evaluator_small_distilled, test_loader)
batch_evaluator.add_evaluation("t5-base-distilled", evaluator_base_distilled, test_loader)

# Compare results
# Automatically generates comparison visualizations
```

## Output Directories

```
experiments/
├── ablation_studies/          # Ablation study results
│   ├── label_smoothing_results.csv
│   ├── data_size_results.csv
│   └── temperature_results.csv
├── degradation_analysis/      # Model size comparison
│   ├── model_size_comparison.csv
│   └── tradeoff_analysis.csv
├── batch_evaluation_test/     # Multi-model comparisons
│   └── batch_comparison_report.csv
└── evaluation_test/           # Individual evaluations
```

## Research Insights

### From Degradation Analysis

- Quantify performance loss from model compression
- Identify "sweet spot" for efficiency vs accuracy
- Measure inference speedup gains
- Calculate memory savings

### From Ablation Studies

- Optimal label smoothing: Usually 0.1-0.2
- Data efficiency: Identify minimum data requirements
- Temperature sensitivity: Find best generation settings
- Training budget: Determine sufficient training data size

## Tips

1. **Quick Testing**: Use subset of data (first 100 samples) in all notebooks
2. **Full Evaluation**: Increase to 1000+ samples for reliable metrics
3. **Ablation Order**: Run ablations first, then use optimal params for final training
4. **Degradation Analysis**: Essential for thesis - shows compression trade-offs
5. **Save Results**: All CSVs saved automatically for later analysis

## Requirements

All notebooks require:

- PyTorch
- Transformers
- Datasets
- Evaluation metrics (rouge-score, bert-score)
- Visualization (matplotlib, seaborn)

See `requirements.txt` for complete list.
