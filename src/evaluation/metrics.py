"""
Metrics Module for Reasoning Distillation Project

Implements various evaluation metrics for assessing student model quality:
- Label accuracy
- ROUGE scores (explanation quality)
- BERTScore (semantic similarity)
- Explanation faithfulness
- Student-teacher agreement
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from collections import Counter
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MetricsConfig:
    """Configuration for metrics computation"""
    compute_rouge: bool = True
    compute_bertscore: bool = True
    compute_faithfulness: bool = True
    bertscore_model: str = "microsoft/deberta-base-mnli"
    rouge_types: List[str] = None
    
    def __post_init__(self):
        if self.rouge_types is None:
            self.rouge_types = ["rouge1", "rouge2", "rougeL"]


class LabelAccuracyMetric:
    """
    Computes label accuracy for NLI tasks.
    Extracts labels from predictions and compares with ground truth.
    """
    
    def __init__(self):
        self.label_map = {
            'entailment': 0,
            'neutral': 1,
            'contradiction': 2
        }
    
    def extract_label(self, text: str) -> Optional[str]:
        """
        Extract label from prediction text.
        
        Args:
            text: Prediction text
            
        Returns:
            Extracted label or None
        """
        text_lower = text.lower().strip()
        
        # Check for labels at the start
        for label in self.label_map.keys():
            if text_lower.startswith(label):
                return label
        
        # Check if label appears anywhere
        for label in self.label_map.keys():
            if label in text_lower:
                return label
        
        return None
    
    def compute(self,
                predictions: List[str],
                references: List[str]) -> Dict[str, float]:
        """
        Compute label accuracy.
        
        Args:
            predictions: List of prediction strings
            references: List of reference strings
            
        Returns:
            Dictionary with accuracy metrics
        """
        pred_labels = []
        ref_labels = []
        
        for pred, ref in zip(predictions, references):
            pred_label = self.extract_label(pred)
            ref_label = self.extract_label(ref)
            
            if pred_label and ref_label:
                pred_labels.append(pred_label)
                ref_labels.append(ref_label)
        
        if not pred_labels:
            return {'label_accuracy': 0.0, 'valid_predictions': 0}
        
        # Compute accuracy
        correct = sum(1 for p, r in zip(pred_labels, ref_labels) if p == r)
        accuracy = correct / len(pred_labels)
        
        # Compute per-class accuracy
        per_class = {}
        for label in self.label_map.keys():
            label_refs = [r for r in ref_labels if r == label]
            if label_refs:
                label_correct = sum(1 for p, r in zip(pred_labels, ref_labels) 
                                   if r == label and p == r)
                per_class[f'accuracy_{label}'] = label_correct / len(label_refs)
        
        return {
            'label_accuracy': accuracy,
            'valid_predictions': len(pred_labels),
            **per_class
        }


class ROUGEMetric:
    """
    Computes ROUGE scores for explanation quality.
    Measures n-gram overlap between predictions and references.
    """
    
    def __init__(self, rouge_types: Optional[List[str]] = None):
        """
        Args:
            rouge_types: List of ROUGE types to compute
        """
        self.rouge_types = rouge_types or ["rouge1", "rouge2", "rougeL"]
        
        # Try to import rouge_score
        try:
            from rouge_score import rouge_scorer
            self.scorer = rouge_scorer.RougeScorer(self.rouge_types, use_stemmer=True)
            self.available = True
        except ImportError:
            logger.warning("rouge_score not available. Install with: pip install rouge-score")
            self.available = False
    
    def extract_explanation(self, text: str) -> str:
        """
        Extract explanation part from text (after label).
        
        Args:
            text: Full text with label and explanation
            
        Returns:
            Extracted explanation
        """
        text_lower = text.lower()
        
        # Look for "explanation:" marker
        if "explanation:" in text_lower:
            parts = text_lower.split("explanation:", 1)
            return parts[1].strip()
        
        # Remove label from beginning
        labels = ['entailment', 'neutral', 'contradiction']
        for label in labels:
            if text_lower.startswith(label):
                return text[len(label):].strip()
        
        return text.strip()
    
    def compute(self,
                predictions: List[str],
                references: List[str]) -> Dict[str, float]:
        """
        Compute ROUGE scores.
        
        Args:
            predictions: List of prediction strings
            references: List of reference strings
            
        Returns:
            Dictionary with ROUGE scores
        """
        if not self.available:
            return {}
        
        # Extract explanations
        pred_explanations = [self.extract_explanation(p) for p in predictions]
        ref_explanations = [self.extract_explanation(r) for r in references]
        
        # Compute ROUGE for each pair
        all_scores = {rouge_type: [] for rouge_type in self.rouge_types}
        
        for pred, ref in zip(pred_explanations, ref_explanations):
            if pred.strip() and ref.strip():
                scores = self.scorer.score(ref, pred)
                for rouge_type in self.rouge_types:
                    all_scores[rouge_type].append(scores[rouge_type].fmeasure)
        
        # Compute averages
        result = {}
        for rouge_type, scores in all_scores.items():
            if scores:
                result[rouge_type] = np.mean(scores)
            else:
                result[rouge_type] = 0.0
        
        return result


class BERTScoreMetric:
    """
    Computes BERTScore for semantic similarity.
    Uses contextualized embeddings to measure similarity.
    """
    
    def __init__(self, model_name: str = "microsoft/deberta-base-mnli"):
        """
        Args:
            model_name: Model to use for BERTScore
        """
        self.model_name = model_name
        
        # Try to import bert_score
        try:
            from bert_score import score as bert_score
            self.bert_score = bert_score
            self.available = True
        except ImportError:
            logger.warning("bert_score not available. Install with: pip install bert-score")
            self.available = False
    
    def extract_explanation(self, text: str) -> str:
        """Extract explanation part from text."""
        text_lower = text.lower()
        
        if "explanation:" in text_lower:
            parts = text_lower.split("explanation:", 1)
            return parts[1].strip()
        
        labels = ['entailment', 'neutral', 'contradiction']
        for label in labels:
            if text_lower.startswith(label):
                return text[len(label):].strip()
        
        return text.strip()
    
    def compute(self,
                predictions: List[str],
                references: List[str],
                batch_size: int = 32) -> Dict[str, float]:
        """
        Compute BERTScore.
        
        Args:
            predictions: List of prediction strings
            references: List of reference strings
            batch_size: Batch size for BERTScore computation
            
        Returns:
            Dictionary with BERTScore metrics
        """
        if not self.available:
            return {}
        
        # Extract explanations
        pred_explanations = [self.extract_explanation(p) for p in predictions]
        ref_explanations = [self.extract_explanation(r) for r in references]
        
        # Filter empty explanations
        valid_pairs = [(p, r) for p, r in zip(pred_explanations, ref_explanations)
                      if p.strip() and r.strip()]
        
        if not valid_pairs:
            return {'bertscore_f1': 0.0, 'bertscore_precision': 0.0, 'bertscore_recall': 0.0}
        
        pred_valid = [p for p, _ in valid_pairs]
        ref_valid = [r for _, r in valid_pairs]
        
        # Compute BERTScore
        P, R, F1 = self.bert_score(
            pred_valid,
            ref_valid,
            model_type=self.model_name,
            batch_size=batch_size,
            verbose=False
        )
        
        return {
            'bertscore_precision': P.mean().item(),
            'bertscore_recall': R.mean().item(),
            'bertscore_f1': F1.mean().item()
        }


class ExplanationFaithfulnessMetric:
    """
    Measures if explanation is faithful to the predicted label.
    Checks for logical consistency between label and explanation.
    """
    
    def __init__(self):
        self.label_map = {
            'entailment': 0,
            'neutral': 1,
            'contradiction': 2
        }
        
        # Keywords that suggest specific labels
        self.entailment_keywords = [
            'therefore', 'thus', 'definitely', 'certainly', 'clearly',
            'must be', 'implies', 'confirms', 'supports'
        ]
        
        self.neutral_keywords = [
            'might', 'could', 'possibly', 'maybe', 'not necessarily',
            'unclear', 'unknown', 'cannot determine', 'may or may not'
        ]
        
        self.contradiction_keywords = [
            'however', 'but', 'not', 'cannot', 'impossible',
            'contradicts', 'opposes', 'different', 'opposite'
        ]
    
    def extract_label_and_explanation(self, text: str) -> Tuple[Optional[str], str]:
        """
        Extract label and explanation from text.
        
        Args:
            text: Full prediction text
            
        Returns:
            Tuple of (label, explanation)
        """
        text_lower = text.lower().strip()
        
        # Find label
        label = None
        for lbl in self.label_map.keys():
            if text_lower.startswith(lbl):
                label = lbl
                break
        
        # Extract explanation
        if "explanation:" in text_lower:
            explanation = text_lower.split("explanation:", 1)[1].strip()
        elif label:
            explanation = text_lower[len(label):].strip()
        else:
            explanation = text_lower
        
        return label, explanation
    
    def check_faithfulness(self, label: str, explanation: str) -> bool:
        """
        Check if explanation is faithful to label.
        
        Args:
            label: Predicted label
            explanation: Explanation text
            
        Returns:
            True if faithful, False otherwise
        """
        if not label or not explanation:
            return False
        
        explanation_lower = explanation.lower()
        
        # Check for keywords matching the label
        if label == 'entailment':
            # Should have entailment keywords, avoid contradiction keywords
            has_positive = any(kw in explanation_lower for kw in self.entailment_keywords)
            has_negative = any(kw in explanation_lower for kw in self.contradiction_keywords)
            return has_positive or not has_negative
        
        elif label == 'neutral':
            # Should have uncertainty keywords
            return any(kw in explanation_lower for kw in self.neutral_keywords)
        
        elif label == 'contradiction':
            # Should have contradiction keywords
            return any(kw in explanation_lower for kw in self.contradiction_keywords)
        
        return True  # Default to faithful if unclear
    
    def compute(self, predictions: List[str]) -> Dict[str, float]:
        """
        Compute faithfulness score.
        
        Args:
            predictions: List of prediction strings
            
        Returns:
            Dictionary with faithfulness metrics
        """
        faithful_count = 0
        valid_count = 0
        
        for pred in predictions:
            label, explanation = self.extract_label_and_explanation(pred)
            
            if label and explanation:
                valid_count += 1
                if self.check_faithfulness(label, explanation):
                    faithful_count += 1
        
        if valid_count == 0:
            return {'faithfulness': 0.0, 'valid_predictions': 0}
        
        return {
            'faithfulness': faithful_count / valid_count,
            'valid_predictions': valid_count
        }


class StudentTeacherAgreementMetric:
    """
    Measures agreement between student and teacher explanations.
    Useful for distillation quality assessment.
    """
    
    def __init__(self):
        self.label_accuracy = LabelAccuracyMetric()
        self.rouge = ROUGEMetric()
    
    def compute(self,
                predictions: List[str],
                references: List[str]) -> Dict[str, float]:
        """
        Compute student-teacher agreement.
        
        Args:
            predictions: Student predictions
            references: Teacher predictions (from dataset)
            
        Returns:
            Dictionary with agreement metrics
        """
        # Label agreement
        label_metrics = self.label_accuracy.compute(predictions, references)
        
        # Explanation overlap (ROUGE)
        explanation_metrics = self.rouge.compute(predictions, references)
        
        # Combine
        agreement = {
            'label_agreement': label_metrics.get('label_accuracy', 0.0),
            'explanation_rouge1': explanation_metrics.get('rouge1', 0.0),
            'explanation_rouge2': explanation_metrics.get('rouge2', 0.0),
            'explanation_rougeL': explanation_metrics.get('rougeL', 0.0)
        }
        
        return agreement


def compute_all_metrics(
    predictions: List[str],
    references: List[str],
    config: Optional[MetricsConfig] = None
) -> Dict[str, float]:
    """
    Compute all available metrics.
    
    Args:
        predictions: List of prediction strings
        references: List of reference strings
        config: Metrics configuration
        
    Returns:
        Dictionary with all computed metrics
    """
    config = config or MetricsConfig()
    all_metrics = {}
    
    # Label accuracy
    label_metric = LabelAccuracyMetric()
    all_metrics.update(label_metric.compute(predictions, references))
    
    # ROUGE
    if config.compute_rouge:
        rouge_metric = ROUGEMetric(config.rouge_types)
        all_metrics.update(rouge_metric.compute(predictions, references))
    
    # BERTScore
    if config.compute_bertscore:
        bertscore_metric = BERTScoreMetric(config.bertscore_model)
        all_metrics.update(bertscore_metric.compute(predictions, references))
    
    # Faithfulness
    if config.compute_faithfulness:
        faithfulness_metric = ExplanationFaithfulnessMetric()
        all_metrics.update(faithfulness_metric.compute(predictions))
    
    return all_metrics


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """
    Format metrics for display.
    
    Args:
        metrics: Dictionary of metrics
        precision: Number of decimal places
        
    Returns:
        Formatted string
    """
    lines = []
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.{precision}f}")
        else:
            lines.append(f"  {key}: {value}")
    
    return "\n".join(lines)


# Example usage function
def compute_metrics_for_evaluation(predictions: List[str], 
                                   references: List[str]) -> Dict[str, float]:
    """
    Convenience function for computing metrics during evaluation.
    This is the function that can be passed to Trainer.
    
    Args:
        predictions: Model predictions
        references: Ground truth references
        
    Returns:
        Dictionary with computed metrics
    """
    config = MetricsConfig(
        compute_rouge=True,
        compute_bertscore=True,
        compute_faithfulness=True
    )
    
    return compute_all_metrics(predictions, references, config)