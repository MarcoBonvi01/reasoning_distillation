"""
Distillation Module for Reasoning Distillation Project

Implements various distillation strategies for transferring reasoning
patterns from teacher to student models.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    """Configuration for distillation training"""
    # Loss weights
    ce_weight: float = 1.0  # Cross-entropy (standard) loss weight
    distill_weight: float = 0.0  # Distillation loss weight (0 = no distillation)
    
    # Temperature for distillation
    temperature: float = 2.0
    
    # Distillation type
    distillation_type: str = "sequence_level"  # "sequence_level" or "token_level"
    
    # Teacher forcing
    use_teacher_forcing: bool = False
    
    # Label smoothing
    label_smoothing: float = 0.0


class DistillationLoss(nn.Module):
    """
    Implements various distillation loss functions.
    
    For sequence-level distillation (recommended for this project):
    - Student learns from final predictions and explanations
    - No need for teacher model at training time (using dataset as teacher)
    
    For token-level distillation (optional, requires teacher model):
    - Student learns from teacher's token-level probability distributions
    - Requires teacher model to be loaded during training
    """
    
    def __init__(self, config: Optional[DistillationConfig] = None):
        super().__init__()
        self.config = config or DistillationConfig()
        
        logger.info(f"Initialized DistillationLoss with config:")
        logger.info(f"  CE weight: {self.config.ce_weight}")
        logger.info(f"  Distill weight: {self.config.distill_weight}")
        logger.info(f"  Temperature: {self.config.temperature}")
        logger.info(f"  Type: {self.config.distillation_type}")
    
    def forward(self,
                student_logits: torch.Tensor,
                labels: torch.Tensor,
                teacher_logits: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute combined distillation loss.
        
        Args:
            student_logits: Student model logits [batch, seq_len, vocab_size]
            labels: Ground truth labels [batch, seq_len]
            teacher_logits: Optional teacher logits [batch, seq_len, vocab_size]
            
        Returns:
            Dictionary with total loss and component losses
        """
        # Standard cross-entropy loss
        ce_loss = self._compute_ce_loss(student_logits, labels)
        
        losses = {
            'ce_loss': ce_loss,
            'total_loss': self.config.ce_weight * ce_loss
        }
        
        # Add distillation loss if teacher logits provided
        if teacher_logits is not None and self.config.distill_weight > 0:
            distill_loss = self._compute_distillation_loss(
                student_logits, teacher_logits, labels
            )
            losses['distill_loss'] = distill_loss
            losses['total_loss'] = losses['total_loss'] + self.config.distill_weight * distill_loss
        
        return losses
    
    def _compute_ce_loss(self,
                        logits: torch.Tensor,
                        labels: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss with optional label smoothing.
        
        Args:
            logits: Model logits [batch, seq_len, vocab_size]
            labels: Ground truth labels [batch, seq_len]
            
        Returns:
            Scalar loss
        """
        # Reshape for loss computation
        vocab_size = logits.size(-1)
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        
        # Compute loss (ignore -100 labels)
        if self.config.label_smoothing > 0:
            loss = self._label_smoothing_loss(logits_flat, labels_flat)
        else:
            loss = F.cross_entropy(
                logits_flat,
                labels_flat,
                ignore_index=-100,
                reduction='mean'
            )
        
        return loss
    
    def _label_smoothing_loss(self,
                             logits: torch.Tensor,
                             labels: torch.Tensor) -> torch.Tensor:
        """
        Cross-entropy with label smoothing.
        
        Args:
            logits: Flattened logits [N, vocab_size]
            labels: Flattened labels [N]
            
        Returns:
            Scalar loss
        """
        vocab_size = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Create smoothed labels
        smooth_labels = torch.zeros_like(log_probs)
        smooth_labels.fill_(self.config.label_smoothing / (vocab_size - 1))
        
        # Mask for valid labels (not -100)
        mask = labels != -100
        valid_labels = labels[mask]
        
        # Set true label probability
        if len(valid_labels) > 0:
            smooth_labels[mask] = smooth_labels[mask].scatter_(
                1, valid_labels.unsqueeze(1), 1.0 - self.config.label_smoothing
            )
        
        # Compute loss
        loss = -(smooth_labels * log_probs).sum(dim=-1)
        loss = loss[mask].mean() if mask.any() else loss.mean()
        
        return loss
    
    def _compute_distillation_loss(self,
                                  student_logits: torch.Tensor,
                                  teacher_logits: torch.Tensor,
                                  labels: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence distillation loss.
        
        Args:
            student_logits: Student logits [batch, seq_len, vocab_size]
            teacher_logits: Teacher logits [batch, seq_len, vocab_size]
            labels: Ground truth labels [batch, seq_len]
            
        Returns:
            Scalar distillation loss
        """
        # Apply temperature
        T = self.config.temperature
        
        # Compute soft predictions
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        
        # Compute KL divergence
        distill_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction='none'
        ).sum(dim=-1)  # Sum over vocabulary
        
        # Mask out padding tokens
        mask = (labels != -100).float()
        distill_loss = (distill_loss * mask).sum() / mask.sum()
        
        # Scale by temperature squared (standard practice)
        distill_loss = distill_loss * (T ** 2)
        
        return distill_loss


class SequenceLevelDistillation:
    """
    Sequence-level distillation using dataset as teacher.
    
    This is the recommended approach for this project since:
    - e-SNLI and Alpaca already contain teacher-quality explanations
    - No need to run teacher model during training (efficient)
    - Focus on mimicking reasoning patterns, not exact distributions
    """
    
    def __init__(self, config: Optional[DistillationConfig] = None):
        self.config = config or DistillationConfig()
        self.loss_fn = DistillationLoss(self.config)
        
        logger.info("Initialized SequenceLevelDistillation")
    
    def compute_loss(self,
                    student_logits: torch.Tensor,
                    labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute loss for sequence-level distillation.
        
        Args:
            student_logits: Student model logits
            labels: Ground truth labels (from teacher/dataset)
            
        Returns:
            Dictionary with loss components
        """
        return self.loss_fn(student_logits, labels, teacher_logits=None)


class TokenLevelDistillation:
    """
    Token-level distillation requiring teacher model.
    
    Optional advanced approach:
    - Requires teacher model to be loaded during training
    - More expensive but potentially better for small students
    - Learns from soft probability distributions at each token
    """
    
    def __init__(self,
                 teacher_model,
                 config: Optional[DistillationConfig] = None):
        self.teacher_model = teacher_model
        self.config = config or DistillationConfig()
        self.loss_fn = DistillationLoss(self.config)
        
        # Set teacher to eval mode
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        logger.info("Initialized TokenLevelDistillation with teacher model")
    
    def compute_loss(self,
                    student_logits: torch.Tensor,
                    labels: torch.Tensor,
                    input_ids: torch.Tensor,
                    attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute loss with teacher logits.
        
        Args:
            student_logits: Student model logits
            labels: Ground truth labels
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Dictionary with loss components
        """
        # Get teacher logits
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            teacher_logits = teacher_outputs['logits']
        
        # Compute combined loss
        return self.loss_fn(student_logits, labels, teacher_logits)


class MultiTaskDistillation:
    """
    Handles distillation from multiple tasks simultaneously.
    
    Useful when combining e-SNLI and Alpaca datasets.
    """
    
    def __init__(self,
                 task_weights: Optional[Dict[str, float]] = None,
                 config: Optional[DistillationConfig] = None):
        """
        Args:
            task_weights: Weights for different tasks (e.g., {'nli': 0.7, 'instruction': 0.3})
            config: Distillation configuration
        """
        self.task_weights = task_weights or {'nli': 1.0, 'instruction': 1.0}
        self.config = config or DistillationConfig()
        self.loss_fn = DistillationLoss(self.config)
        
        logger.info(f"Initialized MultiTaskDistillation with task weights: {self.task_weights}")
    
    def compute_loss(self,
                    student_logits: torch.Tensor,
                    labels: torch.Tensor,
                    task_type: str) -> Dict[str, torch.Tensor]:
        """
        Compute weighted loss for specific task.
        
        Args:
            student_logits: Student model logits
            labels: Ground truth labels
            task_type: Type of task ('nli' or 'instruction')
            
        Returns:
            Dictionary with loss components
        """
        losses = self.loss_fn(student_logits, labels, teacher_logits=None)
        
        # Apply task weight
        task_weight = self.task_weights.get(task_type, 1.0)
        losses['total_loss'] = losses['total_loss'] * task_weight
        losses['task_weight'] = torch.tensor(task_weight)
        
        return losses


class CurriculumDistillation:
    """
    Implements curriculum learning for distillation.
    
    Progressively increases task difficulty or adjusts loss weights
    during training.
    """
    
    def __init__(self,
                 config: Optional[DistillationConfig] = None,
                 warmup_steps: int = 1000):
        self.config = config or DistillationConfig()
        self.loss_fn = DistillationLoss(self.config)
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
        logger.info(f"Initialized CurriculumDistillation with warmup_steps={warmup_steps}")
    
    def compute_loss(self,
                    student_logits: torch.Tensor,
                    labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute loss with curriculum strategy.
        
        Args:
            student_logits: Student model logits
            labels: Ground truth labels
            
        Returns:
            Dictionary with loss components
        """
        losses = self.loss_fn(student_logits, labels, teacher_logits=None)
        
        # Apply curriculum weight (ramp up from 0 to 1)
        if self.current_step < self.warmup_steps:
            curriculum_weight = self.current_step / self.warmup_steps
            losses['total_loss'] = losses['total_loss'] * curriculum_weight
            losses['curriculum_weight'] = torch.tensor(curriculum_weight)
        
        return losses
    
    def step(self):
        """Increment curriculum step."""
        self.current_step += 1


def create_distillation_strategy(
    strategy_type: str = "sequence_level",
    teacher_model = None,
    config: Optional[DistillationConfig] = None,
    **kwargs
):
    """
    Factory function to create distillation strategy.
    
    Args:
        strategy_type: Type of distillation strategy
            - "sequence_level": Standard sequence-level (recommended)
            - "token_level": Token-level with teacher model
            - "multi_task": Multi-task distillation
            - "curriculum": Curriculum learning
        teacher_model: Optional teacher model (required for token_level)
        config: Distillation configuration
        **kwargs: Additional strategy-specific parameters
        
    Returns:
        Distillation strategy instance
    """
    if strategy_type == "sequence_level":
        return SequenceLevelDistillation(config)
    
    elif strategy_type == "token_level":
        if teacher_model is None:
            raise ValueError("teacher_model required for token_level distillation")
        return TokenLevelDistillation(teacher_model, config)
    
    elif strategy_type == "multi_task":
        task_weights = kwargs.get('task_weights', None)
        return MultiTaskDistillation(task_weights, config)
    
    elif strategy_type == "curriculum":
        warmup_steps = kwargs.get('warmup_steps', 1000)
        return CurriculumDistillation(config, warmup_steps)
    
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")


def compare_distillation_strategies():
    """
    Print comparison of different distillation strategies.
    """
    print("=" * 70)
    print("DISTILLATION STRATEGIES")
    print("=" * 70)
    
    print("\n1. Sequence-Level Distillation (RECOMMENDED)")
    print("   ✓ Uses dataset as implicit teacher")
    print("   ✓ No teacher model needed during training")
    print("   ✓ Efficient and scalable")
    print("   ✓ Focus on final predictions and explanations")
    print("   ✗ Doesn't capture intermediate reasoning")
    
    print("\n2. Token-Level Distillation")
    print("   ✓ Learns from soft probability distributions")
    print("   ✓ Can capture richer knowledge")
    print("   ✗ Requires teacher model during training")
    print("   ✗ Much slower and memory intensive")
    print("   ✗ Overkill for explanation generation")
    
    print("\n3. Multi-Task Distillation")
    print("   ✓ Handles multiple datasets/tasks")
    print("   ✓ Task-specific loss weighting")
    print("   ✓ Good for combined e-SNLI + Alpaca")
    print("   ✗ Requires careful weight tuning")
    
    print("\n4. Curriculum Distillation")
    print("   ✓ Gradually increases difficulty")
    print("   ✓ Can improve convergence")
    print("   ✓ Useful for complex reasoning")
    print("   ✗ Adds hyperparameter complexity")
    
    print("=" * 70)


# Recommended configuration for this project
RECOMMENDED_CONFIG = DistillationConfig(
    ce_weight=1.0,
    distill_weight=0.0,  # No explicit distillation, using dataset as teacher
    temperature=1.0,
    distillation_type="sequence_level",
    label_smoothing=0.1  # Slight smoothing helps generalization
)