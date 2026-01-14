"""
Evaluator Module for Reasoning Distillation Project

Implements comprehensive evaluation pipeline for student models,
including metrics computation, result analysis, and comparison.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from .metrics import (
    compute_all_metrics,
    MetricsConfig,
    LabelAccuracyMetric,
    ROUGEMetric,
    BERTScoreMetric,
    ExplanationFaithfulnessMetric,
    format_metrics
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    # Metrics
    metrics_config: MetricsConfig = None
    
    # Generation
    num_beams: int = 4
    max_length: int = 128
    temperature: float = 1.0
    do_sample: bool = False
    
    # Output
    save_predictions: bool = True
    save_detailed_results: bool = True
    output_dir: str = "experiments/evaluation"
    
    # Analysis
    analyze_errors: bool = True
    num_error_examples: int = 10
    
    def __post_init__(self):
        if self.metrics_config is None:
            self.metrics_config = MetricsConfig()


class Evaluator:
    """
    Main evaluator class for comprehensive model evaluation.
    
    Handles:
    - Generation on test/validation sets
    - Metrics computation
    - Error analysis
    - Result saving and visualization
    - Model comparison
    """
    
    def __init__(self,
                 model,
                 config: Optional[EvaluationConfig] = None):
        """
        Args:
            model: Student model to evaluate
            config: Evaluation configuration
        """
        self.model = model
        self.config = config or EvaluationConfig()
        
        # Setup output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Evaluator initialized with output dir: {self.output_dir}")
    
    def evaluate(self,
                dataloader: DataLoader,
                split_name: str = "test") -> Dict[str, any]:
        """
        Run full evaluation on a dataset.
        
        Args:
            dataloader: DataLoader for evaluation data
            split_name: Name of the split (for logging)
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Starting evaluation on {split_name} split...")
        logger.info(f"Total batches: {len(dataloader)}")
        
        self.model.model.eval()
        
        # Collect predictions and references
        all_predictions = []
        all_references = []
        all_inputs = []
        all_losses = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
                # Move to device
                batch = {k: v.to(self.model.config.device) for k, v in batch.items()}
                
                # Compute loss
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                all_losses.append(outputs['loss'].item())
                
                # Generate predictions
                generated_ids = self.model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_length=self.config.max_length,
                    num_beams=self.config.num_beams,
                    temperature=self.config.temperature,
                    do_sample=self.config.do_sample
                )
                
                # Decode
                predictions = self.model.decode_batch(generated_ids)
                
                # Decode labels
                labels = batch['labels'].clone()
                labels[labels == -100] = self.model.tokenizer.pad_token_id
                references = self.model.decode_batch(labels)
                
                # Decode inputs
                inputs = self.model.decode_batch(batch['input_ids'])
                
                all_predictions.extend(predictions)
                all_references.extend(references)
                all_inputs.extend(inputs)
        
        eval_time = time.time() - start_time
        
        logger.info(f"Generation complete in {eval_time:.2f}s")
        logger.info(f"Generated {len(all_predictions)} predictions")
        
        # Compute metrics
        logger.info("Computing metrics...")
        metrics = compute_all_metrics(
            all_predictions,
            all_references,
            self.config.metrics_config
        )
        
        # Add loss
        metrics['loss'] = np.mean(all_losses)
        
        # Add timing info
        metrics['eval_time_seconds'] = eval_time
        metrics['samples_per_second'] = len(all_predictions) / eval_time
        
        # Analyze errors
        error_analysis = None
        if self.config.analyze_errors:
            logger.info("Analyzing errors...")
            error_analysis = self._analyze_errors(
                all_inputs,
                all_predictions,
                all_references
            )
        
        # Save results
        if self.config.save_predictions or self.config.save_detailed_results:
            self._save_results(
                split_name,
                metrics,
                all_inputs,
                all_predictions,
                all_references,
                error_analysis
            )
        
        # Log metrics
        logger.info(f"\n{'='*70}")
        logger.info(f"Evaluation Results - {split_name}")
        logger.info(f"{'='*70}")
        logger.info(format_metrics(metrics))
        logger.info(f"{'='*70}")
        
        return {
            'metrics': metrics,
            'predictions': all_predictions,
            'references': all_references,
            'inputs': all_inputs,
            'error_analysis': error_analysis
        }
    
    def _analyze_errors(self,
                       inputs: List[str],
                       predictions: List[str],
                       references: List[str]) -> Dict[str, any]:
        """
        Analyze prediction errors.
        
        Args:
            inputs: Input texts
            predictions: Predicted texts
            references: Reference texts
            
        Returns:
            Dictionary with error analysis
        """
        label_metric = LabelAccuracyMetric()
        
        # Extract labels
        pred_labels = [label_metric.extract_label(p) for p in predictions]
        ref_labels = [label_metric.extract_label(r) for r in references]
        
        # Find errors
        errors = []
        for i, (inp, pred, ref, pred_lbl, ref_lbl) in enumerate(
            zip(inputs, predictions, references, pred_labels, ref_labels)
        ):
            if pred_lbl != ref_lbl and pred_lbl is not None and ref_lbl is not None:
                errors.append({
                    'index': i,
                    'input': inp,
                    'prediction': pred,
                    'reference': ref,
                    'predicted_label': pred_lbl,
                    'true_label': ref_lbl
                })
        
        # Confusion matrix
        confusion = {}
        for true_label in ['entailment', 'neutral', 'contradiction']:
            confusion[true_label] = {}
            for pred_label in ['entailment', 'neutral', 'contradiction']:
                count = sum(1 for r, p in zip(ref_labels, pred_labels)
                          if r == true_label and p == pred_label)
                confusion[true_label][pred_label] = count
        
        # Sample errors
        sample_errors = errors[:self.config.num_error_examples]
        
        return {
            'total_errors': len(errors),
            'error_rate': len(errors) / len(predictions) if predictions else 0,
            'confusion_matrix': confusion,
            'sample_errors': sample_errors
        }
    
    def _save_results(self,
                     split_name: str,
                     metrics: Dict[str, float],
                     inputs: List[str],
                     predictions: List[str],
                     references: List[str],
                     error_analysis: Optional[Dict] = None):
        """
        Save evaluation results to disk.
        
        Args:
            split_name: Split name
            metrics: Computed metrics
            inputs: Input texts
            predictions: Predictions
            references: References
            error_analysis: Error analysis results
        """
        # Save metrics
        metrics_file = self.output_dir / f"{split_name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_file}")
        
        # Save predictions
        if self.config.save_predictions:
            predictions_file = self.output_dir / f"{split_name}_predictions.jsonl"
            with open(predictions_file, 'w') as f:
                for inp, pred, ref in zip(inputs, predictions, references):
                    f.write(json.dumps({
                        'input': inp,
                        'prediction': pred,
                        'reference': ref
                    }) + '\n')
            logger.info(f"Saved predictions to {predictions_file}")
        
        # Save detailed results
        if self.config.save_detailed_results and error_analysis:
            analysis_file = self.output_dir / f"{split_name}_analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump(error_analysis, f, indent=2)
            logger.info(f"Saved error analysis to {analysis_file}")
    
    def compare_models(self,
                      other_evaluator: 'Evaluator',
                      dataloader: DataLoader,
                      model1_name: str = "Model 1",
                      model2_name: str = "Model 2") -> Dict[str, any]:
        """
        Compare two models side-by-side.
        
        Args:
            other_evaluator: Another evaluator instance
            dataloader: DataLoader for comparison
            model1_name: Name of first model (self)
            model2_name: Name of second model
            
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing {model1_name} vs {model2_name}")
        
        # Evaluate both models
        results1 = self.evaluate(dataloader, split_name=model1_name)
        results2 = other_evaluator.evaluate(dataloader, split_name=model2_name)
        
        # Compare metrics
        comparison = {
            'model1': model1_name,
            'model2': model2_name,
            'metrics_comparison': {}
        }
        
        for key in results1['metrics'].keys():
            if key in results2['metrics']:
                val1 = results1['metrics'][key]
                val2 = results2['metrics'][key]
                
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    diff = val2 - val1
                    rel_diff = (diff / val1 * 100) if val1 != 0 else 0
                    
                    comparison['metrics_comparison'][key] = {
                        model1_name: val1,
                        model2_name: val2,
                        'difference': diff,
                        'relative_difference_pct': rel_diff
                    }
        
        # Save comparison
        comparison_file = self.output_dir / "model_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Saved comparison to {comparison_file}")
        
        return comparison
    
    def evaluate_by_label(self,
                         dataloader: DataLoader) -> Dict[str, Dict[str, float]]:
        """
        Evaluate performance broken down by label.
        
        Args:
            dataloader: DataLoader for evaluation
            
        Returns:
            Dictionary with per-label metrics
        """
        logger.info("Evaluating by label...")
        
        # Collect all predictions
        results = self.evaluate(dataloader, split_name="by_label")
        
        predictions = results['predictions']
        references = results['references']
        
        # Extract labels
        label_metric = LabelAccuracyMetric()
        pred_labels = [label_metric.extract_label(p) for p in predictions]
        ref_labels = [label_metric.extract_label(r) for r in references]
        
        # Group by reference label
        label_groups = {'entailment': [], 'neutral': [], 'contradiction': []}
        
        for pred, ref, pred_lbl, ref_lbl in zip(predictions, references, pred_labels, ref_labels):
            if ref_lbl in label_groups:
                label_groups[ref_lbl].append((pred, ref))
        
        # Compute metrics for each group
        per_label_metrics = {}
        
        for label, pairs in label_groups.items():
            if not pairs:
                continue
            
            preds = [p for p, _ in pairs]
            refs = [r for _, r in pairs]
            
            metrics = compute_all_metrics(preds, refs, self.config.metrics_config)
            per_label_metrics[label] = metrics
        
        logger.info("\nPer-Label Metrics:")
        for label, metrics in per_label_metrics.items():
            logger.info(f"\n{label.upper()}:")
            logger.info(format_metrics(metrics))
        
        return per_label_metrics


class BatchEvaluator:
    """
    Evaluator for batch comparison of multiple models or checkpoints.
    """
    
    def __init__(self, output_dir: str = "experiments/batch_evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def add_evaluation(self,
                      model_name: str,
                      evaluator: Evaluator,
                      dataloader: DataLoader):
        """
        Add a model evaluation to the batch.
        
        Args:
            model_name: Name/identifier for the model
            evaluator: Evaluator instance
            dataloader: DataLoader for evaluation
        """
        logger.info(f"Evaluating {model_name}...")
        
        results = evaluator.evaluate(dataloader, split_name=model_name)
        
        self.results.append({
            'model_name': model_name,
            'metrics': results['metrics']
        })
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """
        Generate comparison table for all evaluated models.
        
        Returns:
            DataFrame with comparison
        """
        if not self.results:
            return pd.DataFrame()
        
        # Extract all metric keys
        all_metrics = set()
        for result in self.results:
            all_metrics.update(result['metrics'].keys())
        
        # Build dataframe
        data = []
        for result in self.results:
            row = {'model': result['model_name']}
            for metric in all_metrics:
                row[metric] = result['metrics'].get(metric, None)
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        csv_file = self.output_dir / "comparison_table.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"Saved comparison table to {csv_file}")
        
        return df
    
    def find_best_model(self, metric: str = "label_accuracy") -> Tuple[str, float]:
        """
        Find the best model based on a specific metric.
        
        Args:
            metric: Metric to optimize
            
        Returns:
            Tuple of (model_name, metric_value)
        """
        if not self.results:
            return None, None
        
        best_model = None
        best_value = -float('inf')
        
        for result in self.results:
            value = result['metrics'].get(metric, -float('inf'))
            if value > best_value:
                best_value = value
                best_model = result['model_name']
        
        return best_model, best_value


def quick_evaluate(model,
                  dataloader: DataLoader,
                  output_dir: str = "experiments/quick_eval") -> Dict[str, float]:
    """
    Quick evaluation helper function.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
        output_dir: Output directory
        
    Returns:
        Dictionary with metrics
    """
    config = EvaluationConfig(
        save_predictions=False,
        save_detailed_results=False,
        analyze_errors=False,
        output_dir=output_dir
    )
    
    evaluator = Evaluator(model, config)
    results = evaluator.evaluate(dataloader, split_name="quick_eval")
    
    return results['metrics']