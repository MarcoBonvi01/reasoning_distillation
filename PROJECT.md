# Distilling Reasoning Patterns from Large Language Models into Lightweight Encoder–Decoder Architectures

## Project Overview

This project explores **knowledge distillation** in the context of reasoning and explanation generation, with the goal of training a smaller, efficient student model to imitate the reasoning behaviour and explanatory patterns of larger teacher LLMs.

Rather than focusing solely on raw answer accuracy, the emphasis is on **behavioural imitation**: learning how the teacher reasons, structures explanations, and communicates its reasoning process.

### Thesis Statement

**Lightweight encoder–decoder models can successfully distill reasoning patterns and explanation styles from large LLMs, achieving significant model compression (10–50×) while maintaining interpretability and reasonable reasoning quality.**

---

## High-Level Objectives

1. **Model Compression**: Demonstrate effective reduction in parameters (10–50×) while retaining reasoning capability.
2. **Efficiency**: Achieve lower inference cost, reduced latency, and smaller memory footprint.
3. **Interpretability**: Maintain or improve model interpretability through explicit explanation generation.
4. **Controlled Degradation**: Quantify and analyze the trade-offs between compression and reasoning quality.
5. **Reasoning Imitation**: Show that explanation quality and reasoning style can be distilled more effectively than raw accuracy.

---

## Domain and Tasks

### Domain

**Textual reasoning with explanations** – avoiding arithmetic-heavy or symbolic computation tasks.

### Core Reasoning Tasks

For each input example, the student model must learn to:

1. **Predict the correct label** (classification outcome)
2. **Generate a natural-language explanation** (rationale) that:
   - Is logically consistent with the predicted label
   - Aligns with the teacher's reasoning style
   - Explains the reasoning process, not just the answer

### Example Task: Natural Language Inference (NLI)

**Input:**

```
Premise: "A person on a horse jumps over a broken down airplane."
Hypothesis: "A person is training his horse for a competition."
```

**Output:**

```
Label: neutral
Explanation: "While a person is on a horse and jumping, it does not necessarily indicate
training for a competition. The premise provides no information about the intent or context
of the activity."
```

---

## Data Sources

### Primary Dataset: e-SNLI

- **Source**: Stanford Natural Language Inference (e-SNLI) corpus
- **Size**: ~550k examples with human-written explanations
- **Task**: Multi-class classification (entailment, neutral, contradiction) with rationale
- **Why e-SNLI?**:
  - Already contains high-quality, human-authored explanations
  - Encodings of teacher-level reasoning
  - No need for costly teacher data generation
  - Well-established benchmark

### Teacher-Level Supervision

Rather than using explicit teacher models, the project treats the datasets themselves as **teacher-authored supervision**. The explanations in e-SNLI and similar datasets reflect the reasoning patterns we want to distill into the student model.

---

## Model Architecture

### Teacher Models

**Implicit Teachers:**

- Large decoder/decoder-only LLMs (e.g., Qwen, LLaMA)
- Serve as knowledge sources embedded in datasets
- Optional for explicit comparison or multi-teacher extension

**Why Not Explicit Teacher?**

- Teacher knowledge is already encoded in high-quality datasets
- Reduces computational cost of distillation
- Aligns with practical knowledge distillation scenarios

### Student Models: FLAN-T5 Family

**Architecture**: Encoder–Decoder (T5-based)

**Why FLAN-T5?**

- Efficient encoder–decoder design
- Strong instruction-following capabilities
- Suitable for explanation generation
- Available in multiple sizes: small, base, large

**Model Specifications**:

| Model         | Parameters | Memory | Layers | Hidden Size |
| ------------- | ---------- | ------ | ------ | ----------- |
| FLAN-T5-small | ~77M       | ~300MB | 6      | 512         |
| FLAN-T5-base  | ~223M      | ~890MB | 12     | 768         |
| FLAN-T5-large | ~770M      | ~3GB   | 24     | 1024        |

---

## Distillation Strategy

### Core Approach: Sequence-Level & Behavioural Distillation

Rather than logit-level or hidden-state distillation, this project focuses on **what the model produces**:

1. **Final Predictions**: Correct label classification
2. **Explanation Structure**: Coherent, well-formed natural language
3. **Reasoning Style**: Verbosity, explanation depth, logical structure

### Training Objective

The student model is trained with a **multi-task loss**:

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{label}} + \beta \cdot \mathcal{L}_{\text{explanation}}$$

Where:

- $\mathcal{L}_{\text{label}}$: Cross-entropy loss for label prediction
- $\mathcal{L}_{\text{explanation}}$: Sequence generation loss (e.g., cross-entropy on tokens) for explanation generation
- $\alpha, \beta$: Hyperparameter weights balancing the two objectives

### Key Design Decisions

1. **No Logit Distillation**: Focus on final outputs, not intermediate representations
2. **Natural Language Explanations**: Reasoning chains expressed in text, not hidden states
3. **Multi-Task Learning**: Joint optimization of label and explanation tasks
4. **Supervised Signals**: Use ground-truth labels and human-written explanations from datasets

### Optional Extensions

- **Multi-Teacher Distillation**: Combine reasoning patterns from datasets labeled by different LLMs
- **Domain Adaptation**: Fine-tune on selected subsets to specialize for specific reasoning patterns
- **Confidence Calibration**: Train the model to express uncertainty in explanations

---

## Evaluation Framework

### Multi-Level Metrics

Evaluation goes far beyond accuracy, focusing on reasoning quality and efficiency.

#### 1. Label Accuracy

**Metric**: Exact match on predicted label

$$\text{Label Accuracy} = \frac{\text{# correct labels}}{\text{# total examples}}$$

- **Threshold**: Baseline to detect catastrophic failure
- **Limitation**: Insufficient alone; does not measure explanation quality

#### 2. Explanation Quality Metrics

##### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

$$\text{ROUGE-L} = \frac{2 \cdot P \cdot R}{P + R}$$

Where P = precision, R = recall of longest common subsequence

- **Measurement**: Surface-level overlap between generated and reference explanations
- **Range**: [0, 1] (1 = perfect overlap)

##### BERTScore

$$\text{BERTScore} = \frac{1}{|y|} \sum_{y_i \in y} \max_{x_j \in x} \cos(\text{BERT}(y_i), \text{BERT}(x_j))$$

- **Measurement**: Semantic similarity using contextualized embeddings
- **Advantage**: Captures meaning beyond surface word overlap
- **Range**: [0, 1]

#### 3. Faithfulness & Consistency

**Metric**: Explanation Faithfulness Score

Assesses whether the explanation logically supports the predicted label:

$$\text{Faithfulness} = \frac{\text{# explanations logically consistent with label}}{\text{# total examples}}$$

- **Measurement**: Manual annotation or automatic heuristics
- **Importance**: Core to reasoning distillation; a good explanation must justify the prediction

#### 4. Student–Teacher Agreement

**Metric**: Agreement between student and teacher explanations

$$\text{Agreement} = \text{BERTScore}(\text{student explanation}, \text{teacher explanation})$$

- **Purpose**: Measures style and reasoning pattern alignment
- **Insight**: Shows how well the student imitates teacher behaviour

#### 5. Efficiency Metrics

| Metric            | Description                           | Unit        |
| ----------------- | ------------------------------------- | ----------- |
| Parameter Count   | Total learnable parameters            | # params    |
| Memory Footprint  | RAM required for inference            | MB          |
| Inference Latency | Time per example                      | ms          |
| Throughput        | Examples per second                   | samples/sec |
| FLOPs             | Approximate floating-point operations | GFLOPs      |

### Comprehensive Evaluation Protocol

**Phase 1: Individual Metric Testing**

- Test each metric independently on sample predictions
- Validate metric implementations
- Establish baseline performance

**Phase 2: Per-Label Evaluation**

- Analyze performance separately for each label (entailment, neutral, contradiction)
- Detect label-specific bias or degradation
- Identify challenging reasoning patterns

**Phase 3: Error Analysis**

- Confusion matrices for misclassifications
- Categorize error types (label error, explanation error, both)
- Analyze failure modes

**Phase 4: Trade-off Analysis**

- Compression vs. accuracy
- Speed vs. quality
- Parameter efficiency
- Identify Pareto-optimal models

---

## Project Structure

```
reasoning-distillation/
├── README.md                          # Project documentation
├── PROJECT.md                         # This file (detailed project description)
├── requirements.txt                   # Python dependencies
│
├── configs/                           # Configuration files
│   ├── model_configs.yaml            # Model hyperparameters
│   ├── training_configs.yaml         # Training settings
│   └── eval_configs.yaml             # Evaluation settings
│
├── data/                              # Data directory
│   ├── raw/                          # Raw downloaded data
│   │   ├── e-snli/                  # e-SNLI dataset
│   │   └── alpaca/                  # Alpaca dataset
│   └── processed/                    # Preprocessed data
│       ├── train/                   # Training splits
│       ├── val/                     # Validation splits
│       └── test/                    # Test splits
│
├── src/                              # Source code
│   ├── __init__.py
│   │
│   ├── data/                        # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── data_loader.py          # Dataset loading utilities
│   │   ├── dataset.py              # PyTorch Dataset classes
│   │   └── preprocessor.py         # Text preprocessing and tokenization
│   │
│   ├── models/                      # Model implementations
│   │   ├── __init__.py
│   │   ├── student.py              # Student model (FLAN-T5)
│   │   └── teacher.py              # Teacher model interface
│   │
│   ├── training/                    # Training and distillation
│   │   ├── __init__.py
│   │   ├── trainer.py              # Main training loop
│   │   └── distillation.py         # Distillation loss functions
│   │
│   ├── evaluation/                  # Evaluation metrics and pipelines
│   │   ├── __init__.py
│   │   ├── metrics.py              # Metric implementations
│   │   └── evaluator.py            # Evaluation orchestration
│   │
│   └── utils/                       # Utility functions
│       ├── __init__.py
│       ├── logging.py              # Logging utilities
│       └── config.py               # Configuration management
│
├── notebooks/                        # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb           # Data analysis
│   ├── 02_preprocessing_and_datasets.ipynb # Dataset pipeline testing
│   ├── 03_model_testing.ipynb              # Model loading and forward pass
│   ├── 04_training_loop_testing.ipynb      # Training loop validation
│   ├── 05_evaluation_testing.ipynb         # Evaluation metrics testing
│   ├── 06_ablation_studies.ipynb           # Ablation study execution
│   └── README.md                           # Notebook guide
│
├── experiments/                      # Experiment outputs
│   ├── runs/                        # Training checkpoints
│   ├── evaluation_test/             # Evaluation results
│   ├── batch_evaluation_test/       # Multi-model evaluation
│   ├── degradation_analysis/        # Model size analysis
│   └── quick_eval_test/             # Quick eval outputs
│
└── outputs/                         # Final results and reports
    ├── metrics_summary.csv
    ├── degradation_curves.png
    └── final_report.md
```

---

## Notebooks Overview

### 01_data_exploration.ipynb

**Purpose**: Explore and understand the datasets

**Content**:

- Load e-SNLI and Alpaca datasets
- Analyze dataset statistics (size, label distribution, explanation length)
- Visualize reasoning patterns
- Identify potential issues or biases

**Expected Output**: Data insights and quality assessment

---

### 02_preprocessing_and_datasets.ipynb

**Purpose**: Test the data pipeline

**Content**:

- Initialize TaskFormatter for NLI and instruction tasks
- Test ReasoningPreprocessor tokenization
- Validate PyTorch Dataset classes
- Test DataLoader batching
- Validate caching mechanisms

**Expected Output**: Confirmed data pipeline functionality

---

### 03_model_testing.ipynb

**Purpose**: Validate model loading and forward pass

**Content**:

- Load FLAN-T5 student and teacher models
- Test forward pass on sample batches
- Validate output shapes and dimensions
- Analyze loss landscape
- Test generation with beam search

**Expected Output**: Model baseline performance

---

### 04_training_loop_testing.ipynb

**Purpose**: Test the training loop before large-scale training

**Content**:

- Initialize trainer with configuration
- Run a few training steps on sample data
- Validate loss computation
- Test checkpointing and resume
- Monitor gradient flow

**Expected Output**: Confirmed training functionality

---

### 05_evaluation_testing.ipynb

**Purpose**: Validate evaluation metrics and pipelines

**Content**:

- Test individual metrics (Accuracy, ROUGE, BERTScore, Faithfulness)
- Test full evaluation pipeline
- Error analysis and confusion matrices
- Per-label evaluation
- Model comparison across sizes
- Trade-off analysis (compression vs. quality)

**Expected Output**: Validated evaluation framework

---

### 06_ablation_studies.ipynb

**Purpose**: Run ablation studies on key components

**Content**:

- Multi-task weight variations ($\alpha$, $\beta$)
- Architecture variations (layer count, hidden size)
- Data augmentation effects
- Domain adaptation studies

**Expected Output**: Insights into component importance

---

## Key Concepts

### Knowledge Distillation

**Definition**: Training a smaller model to approximate the behaviour of a larger model.

**In This Project**:

- **Teacher**: Implicit (via high-quality datasets)
- **Student**: FLAN-T5 models
- **Target**: Reasoning and explanation patterns (not just labels)

### Behavioural Imitation

Rather than logit-level matching, we focus on:

- **What**: Which explanations the student generates
- **How**: The structure and style of explanations
- **Why**: The reasoning logic encoded in natural language

### Encoder–Decoder for Explanation Generation

**Why Encoder–Decoder?**

- Encoder processes input (premise + hypothesis)
- Decoder generates output (explanation token by token)
- Well-suited for sequence-to-sequence reasoning tasks
- Strong empirical results on text generation

### Multi-Task Learning

Student is trained jointly on:

1. Label prediction (classification)
2. Explanation generation (text generation)

This forces the model to learn to reason (via explanation) while predicting correctly.

---

## Training Procedure

### Phase 1: Data Preparation

1. **Load Dataset**: e-SNLI (training, validation, test splits)
2. **Preprocess**: Tokenize, truncate, pad sequences
3. **Create DataLoaders**: Batch and shuffle for training

### Phase 2: Model Initialization

1. **Load FLAN-T5**: Pretrained model from Hugging Face
2. **Freeze/Unfreeze Layers**: (Optional) freeze encoder, fine-tune decoder
3. **Initialize Optimizer**: AdamW with scheduled learning rate

### Phase 3: Training Loop

**For each epoch:**

1. For each batch:

   - Forward pass: compute label and explanation losses
   - Backward pass: compute gradients
   - Update weights: optimizer step
   - Log metrics: loss, accuracy, perplexity

2. Validate on validation set
3. Save checkpoint if validation improves
4. Early stopping if no improvement for N epochs

### Phase 4: Evaluation

1. **Full Evaluation**: Run complete metric suite
2. **Analysis**: Error analysis, per-label breakdown
3. **Trade-off Analysis**: Compression vs. quality
4. **Report Generation**: Summary statistics and visualizations

---

## Hyperparameters

### Model Hyperparameters

```yaml
model_name: "google/flan-t5-base"
max_source_length: 256
max_target_length: 128
encoder_layers: 12
decoder_layers: 12
hidden_size: 768
num_heads: 12
```

### Training Hyperparameters

```yaml
batch_size: 16
learning_rate: 1e-4
num_epochs: 10
warmup_steps: 500
gradient_accumulation_steps: 2
loss_weights:
  alpha: 0.5 # label loss weight
  beta: 0.5 # explanation loss weight
dropout: 0.1
weight_decay: 0.01
```

### Generation Hyperparameters

```yaml
num_beams: 4
max_length: 128
repetition_penalty: 1.2
length_penalty: 0.8
early_stopping: True
```

---

## Expected Results

### Performance Baseline (FLAN-T5-base)

Expected ranges based on literature and preliminary experiments:

| Metric         | Expected Value |
| -------------- | -------------- |
| Label Accuracy | 0.82–0.88      |
| ROUGE-L        | 0.35–0.45      |
| BERTScore      | 0.90–0.95      |
| Faithfulness   | 0.75–0.85      |

### Compression Trade-offs

#### FLAN-T5-small (vs. FLAN-T5-base)

- **Compression**: ~3.5× fewer parameters (~77M vs 223M)
- **Accuracy Degradation**: ~5–10%
- **Explanation Quality**: ~3–5% ROUGE-L drop
- **Speedup**: 2–3×
- **Recommendation**: Suitable for low-latency deployments

#### FLAN-T5-large (vs. FLAN-T5-base)

- **Expansion**: ~3.5× more parameters (~770M)
- **Accuracy Gain**: ~5–8%
- **Explanation Quality**: ~3–5% ROUGE-L improvement
- **Slowdown**: 2–3×
- **Use Case**: High-accuracy scenarios; not compression-focused

### Key Insights

1. **Graceful Degradation**: Explanation quality degrades more gracefully than label accuracy as model size decreases.
2. **Interpretability Benefit**: Explicit explanations improve model trustworthiness.
3. **Efficiency Wins**: Small models can achieve 90%+ performance retention at 3.5× compression.
4. **Reasoning Imitation**: Students successfully learn teacher reasoning patterns without explicit logit distillation.

---

## Experimental Design

### Baseline Experiments

1. **Standard Fine-Tuning**: FLAN-T5 models fine-tuned on e-SNLI
2. **Label-Only Training**: Train only on label prediction (no explanations)
3. **Explanation-Only Training**: Train only on explanation generation

### Ablation Studies

1. **Loss Weight Variations**: Test different $\alpha$ and $\beta$ values
2. **Architecture Variants**: Vary layer count, hidden size
3. **Data Variations**: Train on subsets, analyze domain sensitivity
4. **Multi-Task Variants**: Compare joint vs. sequential training

### Analysis Tracks

1. **Compression Analysis**: Model size vs. performance trade-offs
2. **Generalization**: Performance on held-out test set vs. training set
3. **Interpretability**: Explanation quality and consistency
4. **Efficiency**: Latency, throughput, memory usage

---

## Setup and Execution

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- 16GB+ RAM (for model loading)

### Installation

```bash
# Clone repository
git clone <repo-url>
cd reasoning-distillation

# Install dependencies
pip install -r requirements.txt

# Download datasets
python src/data/data_loader.py --download --output-dir data/raw
```

### Quick Start

```bash
# Run exploration notebooks
jupyter notebook notebooks/01_data_exploration.ipynb

# Test data pipeline
jupyter notebook notebooks/02_preprocessing_and_datasets.ipynb

# Run training
python src/training/trainer.py --config configs/training_configs.yaml

# Evaluate
python src/evaluation/evaluator.py --config configs/eval_configs.yaml
```

### GPU Considerations

- Single GPU (16GB): FLAN-T5-base batch size 16, no gradient accumulation
- Multi-GPU: Use distributed training (DistributedDataParallel)
- CPU Mode: Supported but slow (use for debugging only)

---

## Dependencies

Key libraries and versions:

```
torch>=1.13.0
transformers>=4.25.0
datasets>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
rouge-score>=0.1.2
bert-score>=0.3.13
matplotlib>=3.5.0
seaborn>=0.12.0
tqdm>=4.62.0
wandb>=0.13.0  # For experiment tracking
```

See [requirements.txt](requirements.txt) for complete list.

---

## Best Practices and Reproducibility

### Code Organization

- **Modular Design**: Separate data, models, training, evaluation
- **Configuration-Driven**: Use YAML configs, avoid hardcoded values
- **Logging**: Comprehensive logging at all stages
- **Error Handling**: Graceful handling of edge cases

### Experiment Tracking

- **Logging**: Save all metrics, hyperparameters, timestamps
- **Checkpointing**: Save model weights at regular intervals
- **Versioning**: Track dataset and code versions
- **Reproducibility**: Set random seeds for Python, NumPy, PyTorch

### Validation Practices

1. **Data Validation**: Check dataset statistics before training
2. **Model Validation**: Confirm model loads and forward pass works
3. **Loss Validation**: Ensure losses decrease during training
4. **Metric Validation**: Compare against published baselines

---

## Troubleshooting

### Common Issues

**Out of Memory (OOM)**

- Reduce batch size
- Use gradient checkpointing
- Reduce max sequence length

**Slow Training**

- Use GPU (check `torch.cuda.is_available()`)
- Increase batch size (if memory allows)
- Use mixed precision training (`torch.cuda.amp`)

**Poor Performance**

- Check learning rate (try 1e-5 to 1e-3)
- Verify data loading (check a few batches)
- Confirm loss is decreasing
- Try longer training (more epochs)

**Evaluation Failures**

- Ensure models are in evaluation mode (`.eval()`)
- Check for NaN or Inf in predictions
- Validate metric implementations

---

## Future Extensions

### Short-Term

1. **Multi-Teacher Distillation**: Combine reasoning from multiple LLM-generated datasets
2. **Domain Adaptation**: Fine-tune on specific reasoning types
3. **Confidence Calibration**: Train model to express uncertainty
4. **Analysis Tools**: Interactive visualization of explanations

### Medium-Term

1. **Cross-Lingual**: Apply distillation to non-English languages
2. **Other Tasks**: Extend to summarization, QA, semantic parsing
3. **Structured Output**: Generate explanations in structured formats
4. **Hybrid Approaches**: Combine symbolic and neural reasoning

### Long-Term

1. **Explainability Guarantees**: Formal verification of explanation correctness
2. **Federated Learning**: Distributed training without centralizing data
3. **Continual Learning**: Update models with new reasoning patterns
4. **Application Deployment**: Production systems with lightweight, interpretable models

---

## References

### Knowledge Distillation

- Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
- FitzPatrick et al., "How Much Knowledge Can You Pack Into the Parameters of a Language Model?" (2022)

### Natural Language Inference

- Bowman et al., "A large annotated corpus for learning natural language inference" (SNLI, 2015)
- Camburu et al., "e-SNLI: Natural Language Inference with Natural Language Explanations" (2018)

### Encoder–Decoder Models

- Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5, 2019)
- Chung et al., "Scaling Instruction-Finetuned Language Models" (FLAN, 2022)

### Evaluation Metrics

- Lin, "ROUGE: A Package for Automatic Evaluation of Summaries" (2004)
- Zhang et al., "BERTScore: Evaluating Text Generation with BERT" (2020)

---

## Contact and Attribution

**Project Title**: Distilling Reasoning Patterns from Large Language Models into Lightweight Encoder–Decoder Architectures

**Developed**: Academic project for Master's programme in NLP

**Last Updated**: January 2026

For questions, issues, or contributions, please refer to the project repository.

---

## License

This project is developed for academic purposes. See LICENSE file for details.

---

**End of Project Description**

This document provides a comprehensive overview of the project structure, objectives, methodology, and expected outcomes. Refer to individual notebooks for hands-on exploration and [README.md](README.md) for quick-start instructions.
