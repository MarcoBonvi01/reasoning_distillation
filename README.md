# README

## Project: The Apprentice Model – Reasoning Distillation

This project explores knowledge distillation by training a smaller model to imitate a larger LLM on a specific domain or reasoning task. The aim is to obtain lightweight, domain-specialized “expert” models while analyzing trade-offs between efficiency, specialization, and generalization.

### Core Pipeline

1. **Data Collection**: Use a large LLM to generate or label examples in a specific domain (e.g., medical, legal, or cultural) and use the large LLM as a teacher model.
2. **Model Distillation**: Train a smaller transformer (e.g., DistilBERT, TinyT5 or a custom model) to reproduce the teacher model’s predictions or reasoning chains.
3. **Evaluation**: Compare distilled vs. teacher model on domain benchmarks, focusing on performance drop, interpretability, and computational savings.
4. **Extension**: Experiment with multi-teacher distillation or domain adaptation through selective fine-tuning.

### Expected Outcomes

Students will measure trade-offs between model compactness, domain specialisation, and reasoning quality, demonstrating the degree to which distilled systems retain expert knowledge.

---

## Notebooks Overview

### 01_data_exploration.ipynb

**Purpose:** Explore and analyze the e-SNLI dataset. Load the data, compute statistics (size, label distribution, explanation length), visualize reasoning patterns, and identify potential issues or biases. Output: Data insights and quality assessment.

### 02_preprocessing_and_datasets.ipynb

**Purpose:** Test and validate the data pipeline. Initialize and test the TaskFormatter for NLI and instruction tasks, ReasoningPreprocessor for tokenization, PyTorch Dataset classes, DataLoader batching, and caching mechanisms. Output: Confirmed data pipeline functionality.

### 03_student_model.ipynb

**Purpose:** Load and test the FLAN-T5 student model. Run forward passes on sample batches, validate output shapes and dimensions, analyze the loss landscape, and test generation with beam search. Output: Model baseline performance and readiness for training.

### 04_teacher_model.ipynb

**Purpose:** Load and test the teacher model (LLM or dataset-based supervision). Compare outputs with the student model, analyze reasoning style, and prepare for distillation. Output: Teacher reference for distillation and evaluation.

### 05_ablation_studies.ipynb

**Purpose:** Run ablation studies on key components. Vary multi-task weights (α, β), architecture (layer count, hidden size), data augmentation, and domain adaptation. Output: Insights into the importance of each component and their effect on performance.

### 06_training_loop.ipynb

**Purpose:** Validate the training loop. Initialize the trainer, run training steps on sample data, validate loss computation, test checkpointing and resume, and monitor gradient flow. Output: Confirmed training functionality and stability.

### 06.1_baseline_training.ipynb

**Purpose:** Run baseline training for the student model. Fine-tune on e-SNLI, monitor metrics, and save checkpoints. Output: Baseline model for comparison in ablation and evaluation.

### 07_evaluation.ipynb

**Purpose:** Validate evaluation metrics and pipelines. Test metrics (Accuracy, ROUGE, BERTScore, Faithfulness), run full evaluation, error analysis, confusion matrices, per-label evaluation, model comparison, and trade-off analysis. Output: Validated evaluation framework and results.

### 08_compression_analysis.ipynb

**Purpose:** Analyze model compression and efficiency. Compare student and teacher models on parameter count, memory, latency, throughput, and performance trade-offs. Output: Insights into compression, efficiency, and quality retention.

---

## Project Structure

See the folder tree for organization. Key folders: `src/` (code), `data/` (datasets), `notebooks/` (exploration and experiments), `experiments/` (outputs), `configs/` (YAML configs).

---

## Methodology

The project follows these main steps, in order:

1. **Dataset Download**: Download the e-SNLI and other relevant datasets.
2. **Preprocessing**: Preprocess and format the data for NLI and instruction-following tasks.
3. **Model Loading & Testing**: Load and test both the student (FLAN-T5) and teacher models to ensure correct setup and outputs.
4. **Ablation Studies**: Perform ablation studies to explore the impact of architectural choices, loss weighting, and data augmentation, and to select optimal training parameters.
5. **Student Training with Distillation**: Train the student model using the logits and supervision from the teacher model (knowledge distillation).
6. **Student Baseline Training**: Train the student model without teacher supervision to establish a baseline.
7. **Evaluation**: Evaluate the trained models on the test set using metrics such as accuracy, ROUGE, BERTScore, and faithfulness to assess distillation effectiveness.
8. **Compression Analysis**: Analyze the compression and efficiency of the student model, comparing it to the teacher in terms of size, speed, and performance.

---

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Download datasets: `python src/data/data_loader.py --download --output-dir data/raw`
3. Run notebooks for exploration and validation.
4. Train and evaluate models using scripts in `src/`.

---

## References

See the end of this document for key papers and resources on knowledge distillation, NLI, encoder–decoder models, and evaluation metrics.

---

## License & Attribution

Academic project for Master's programme in NLP. See LICENSE for details.

Code generated with Claude Sonet 4.

---

**End of README**
