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


class TokenLevelDistillation:
    """
    Token-level distillation requiring teacher model.
    
    Implements the distillation pipeline:
    
    Dataset → Teacher Model → Soft Logits (probabilities)
           ↘                ↗
             Student Model
             
    Loss = α·CE(student, labels) + β·KL(student||teacher)
    
    Where:
    - α (ce_weight): Weight for cross-entropy loss with hard labels
    - β (distill_weight): Weight for KL divergence with teacher soft logits
    """
    
    def __init__(self,
                 teacher_model,
                 config: Optional[DistillationConfig] = None):
        """
        Initialize token-level distillation.
        
        Args:
            teacher_model: Teacher model (e.g., FlanT5Teacher)
            config: Distillation configuration with ce_weight (α) and distill_weight (β)
        """
        self.teacher_model = teacher_model
        self.config = config or DistillationConfig()
        self.loss_fn = DistillationLoss(self.config)
        
        # Ensure teacher is in eval mode with frozen parameters
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        logger.info("Initialized TokenLevelDistillation with teacher model")
        logger.info(f"  α (CE weight): {self.config.ce_weight}")
        logger.info(f"  β (Distill weight): {self.config.distill_weight}")
        logger.info(f"  Temperature: {self.config.temperature}")
    
    def compute_loss(self,
                    student_logits: torch.Tensor,
                    labels: torch.Tensor,
                    input_ids: torch.Tensor,
                    attention_mask: torch.Tensor,
                    decoder_input_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute combined distillation loss.
        
        Loss = α·CE(student, labels) + β·KL(student||teacher)
        
        Args:
            student_logits: Student model logits [batch, seq_len, vocab_size]
            labels: Ground truth labels [batch, seq_len]
            input_ids: Encoder input token IDs [batch, src_len]
            attention_mask: Encoder attention mask [batch, src_len]
            decoder_input_ids: Decoder input IDs [batch, tgt_len]
            
        Returns:
            Dictionary with loss components:
            - total_loss: Combined loss
            - ce_loss: Cross-entropy loss component
            - distill_loss: KL divergence loss component
        """
        # Get teacher logits (soft targets)
        with torch.no_grad():
            if decoder_input_ids is None:
                # Create decoder input ids from labels (shift right)
                decoder_input_ids = self._shift_right(labels)
            
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids
            )
            teacher_logits = teacher_outputs['logits']
        
        # Handle vocabulary size mismatch between teacher and student
        if teacher_logits.size(-1) != student_logits.size(-1):
            teacher_logits = self._align_vocab_sizes(
                teacher_logits, student_logits.size(-1)
            )
        
        # Compute combined loss: α·CE + β·KL
        return self.loss_fn(student_logits, labels, teacher_logits)
    
    def _shift_right(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Shift labels right to create decoder input ids.
        T5 uses pad_token_id as the start token for decoder.
        """
        pad_token_id = 0  # T5 pad token id
        shifted = labels.new_zeros(labels.shape)
        shifted[:, 1:] = labels[:, :-1].clone()
        shifted[:, 0] = pad_token_id
        # Replace -100 (ignore index) with pad_token_id
        shifted[shifted == -100] = pad_token_id
        return shifted
    
    def _align_vocab_sizes(self, 
                           teacher_logits: torch.Tensor,
                           target_vocab_size: int) -> torch.Tensor:
        """
        Align teacher logits vocabulary size to match student.
        
        Args:
            teacher_logits: Teacher logits [batch, seq_len, teacher_vocab_size]
            target_vocab_size: Target vocabulary size
            
        Returns:
            Aligned logits [batch, seq_len, target_vocab_size]
        """
        current_vocab_size = teacher_logits.size(-1)
        
        if current_vocab_size > target_vocab_size:
            # Truncate teacher vocab
            return teacher_logits[:, :, :target_vocab_size]
        else:
            # Pad teacher vocab with very negative values
            padding = torch.full(
                (*teacher_logits.shape[:-1], target_vocab_size - current_vocab_size),
                fill_value=-1e9,
                device=teacher_logits.device,
                dtype=teacher_logits.dtype
            )
            return torch.cat([teacher_logits, padding], dim=-1)

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
    
    print("\n1. Token-Level Distillation with Teacher Model (USED)")
    print("   Uses soft probability distributions from teacher (dark knowledge)")
    print("   Loss = α·CE(student, labels) + β·KL(student||teacher)")
    print("   Captures richer knowledge than hard labels alone")
    print("   ✗ Requires teacher model during training (more memory)")

    print("\n2. Sequence-Level Distillation")
    print("   Standard distillation using hard labels only")
    print("   Simpler and less resource-intensive")
    print("   ✗ Does not leverage teacher's soft predictions")
    print("   Loss = α·CE(student, labels) + β·CE(teacher, labels)")

    print("\n3. Multi-Task Distillation")
    print("   Combines multiple distillation tasks")
    print("   Allows weighting different tasks")
    print("   ✗ More complex to implement and tune")   
