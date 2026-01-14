"""
Quality Analysis Module for Explanation Evaluation

This module provides functions to assess the quality of generated explanations,
including tautology detection and similarity metrics.
"""

from typing import Dict, List
import numpy as np


def detect_tautology(prediction: str, input_text: str, threshold: float = 0.7) -> Dict:
    """
    Detect if prediction is a tautology (mostly repeats the input).
    
    Args:
        prediction: Generated explanation text
        input_text: Original input (premise + hypothesis)
        threshold: Similarity threshold above which to flag as tautology (default: 0.7)
    
    Returns:
        Dictionary with:
        - similarity: Overlap ratio between prediction and input
        - is_tautology: Boolean flag
        - common_words: Number of overlapping words
        - total_words: Total unique words in prediction
    """
    # Normalize text
    pred_words = set(prediction.lower().split())
    input_words = set(input_text.lower().split())
    
    # Calculate overlap
    common_words = pred_words & input_words
    if len(pred_words) == 0:
        similarity = 0.0
    else:
        similarity = len(common_words) / len(pred_words)
    
    return {
        'similarity': similarity,
        'is_tautology': similarity > threshold,
        'common_words': len(common_words),
        'total_words': len(pred_words)
    }


def calculate_explanation_metrics(
    prediction: str,
    ground_truth: str,
    input_text: str,
    tautology_threshold: float = 0.7
) -> Dict:
    """
    Calculate comprehensive quality metrics for explanations.
    
    Args:
        prediction: Generated explanation
        ground_truth: Reference explanation
        input_text: Original input
        tautology_threshold: Threshold for tautology detection
    
    Returns:
        Dictionary with:
        - is_tautology: Whether prediction is a tautology
        - tautology_similarity: Overlap with input (0-1)
        - similarity_to_ground_truth: Token overlap with GT (0-1)
        - prediction_length: Word count of prediction
        - ground_truth_length: Word count of GT
        - length_ratio: Prediction length / GT length
    """
    # Tautology detection
    tautology_score = detect_tautology(prediction, input_text, tautology_threshold)
    
    # Similarity with ground truth (simple token overlap)
    pred_tokens = set(prediction.lower().split())
    gt_tokens = set(ground_truth.lower().split())
    common = pred_tokens & gt_tokens
    similarity_to_gt = len(common) / max(len(pred_tokens), len(gt_tokens)) if pred_tokens else 0.0
    
    # Length comparison
    pred_len = len(prediction.split())
    gt_len = len(ground_truth.split())
    
    return {
        'is_tautology': tautology_score['is_tautology'],
        'tautology_similarity': tautology_score['similarity'],
        'similarity_to_ground_truth': similarity_to_gt,
        'prediction_length': pred_len,
        'ground_truth_length': gt_len,
        'length_ratio': pred_len / gt_len if gt_len > 0 else 0.0
    }


def analyze_batch_quality(
    predictions: List[str],
    ground_truths: List[str],
    inputs: List[str],
    tautology_threshold: float = 0.7
) -> Dict:
    """
    Analyze quality metrics for a batch of explanations.
    
    Args:
        predictions: List of generated explanations
        ground_truths: List of reference explanations
        inputs: List of inputs
        tautology_threshold: Threshold for tautology detection
    
    Returns:
        Dictionary with:
        - all_metrics: List of metric dicts for each sample
        - tautology_count: Total number of tautologies
        - tautology_percentage: Percentage of tautologies
        - avg_tautology_similarity: Average tautology similarity
        - avg_gt_similarity: Average ground truth similarity
        - avg_prediction_length: Average prediction length
        - avg_gt_length: Average GT length
        - avg_length_ratio: Average length ratio
        - tautology_examples: Indices of first 3 tautologies
    """
    all_metrics = []
    tautology_count = 0
    tautology_indices = []
    
    for i in range(len(predictions)):
        metrics = calculate_explanation_metrics(
            predictions[i],
            ground_truths[i],
            inputs[i],
            tautology_threshold
        )
        all_metrics.append(metrics)
        if metrics['is_tautology']:
            tautology_count += 1
            tautology_indices.append(i)
    
    # Calculate aggregated metrics
    tautology_similarity = [m['tautology_similarity'] for m in all_metrics]
    gt_similarity = [m['similarity_to_ground_truth'] for m in all_metrics]
    pred_lengths = [m['prediction_length'] for m in all_metrics]
    gt_lengths = [m['ground_truth_length'] for m in all_metrics]
    length_ratios = [m['length_ratio'] for m in all_metrics]
    
    return {
        'all_metrics': all_metrics,
        'tautology_count': tautology_count,
        'tautology_percentage': (tautology_count / len(predictions) * 100) if predictions else 0.0,
        'avg_tautology_similarity': np.mean(tautology_similarity) if tautology_similarity else 0.0,
        'avg_gt_similarity': np.mean(gt_similarity) if gt_similarity else 0.0,
        'avg_prediction_length': np.mean(pred_lengths) if pred_lengths else 0.0,
        'avg_gt_length': np.mean(gt_lengths) if gt_lengths else 0.0,
        'avg_length_ratio': np.mean(length_ratios) if length_ratios else 0.0,
        'tautology_examples': tautology_indices[:3]  # First 3 tautologies
    }


def print_quality_analysis(quality_stats: Dict) -> None:
    """
    Pretty print quality analysis results.
    
    Args:
        quality_stats: Dictionary returned from analyze_batch_quality()
    """
    print("=" * 70)
    print("EXPLANATION QUALITY ANALYSIS")
    print("=" * 70)
    
    print(f"\n📊 Tautology Detection:")
    print(f"  Tautologies: {quality_stats['tautology_count']} ({quality_stats['tautology_percentage']:.1f}%)")
    print(f"  Avg tautology similarity: {quality_stats['avg_tautology_similarity']:.4f}")
    
    print(f"\n🎯 Similarity to Ground Truth:")
    print(f"  Avg GT similarity: {quality_stats['avg_gt_similarity']:.4f}")
    
    print(f"\n📏 Length Analysis:")
    print(f"  Avg prediction length: {quality_stats['avg_prediction_length']:.1f} words")
    print(f"  Avg GT length: {quality_stats['avg_gt_length']:.1f} words")
    print(f"  Avg length ratio (pred/GT): {quality_stats['avg_length_ratio']:.2f}")
