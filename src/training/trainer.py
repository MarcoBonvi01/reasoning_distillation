"""
Trainer Module for Reasoning Distillation Project

Implements training loop with support for distillation, evaluation,
checkpointing, and experiment tracking.
"""

import logging
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
    SequentialLR
)
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training"""
    # Training hyperparameters
    num_epochs: int = 5
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    
    # Learning rate schedule
    lr_scheduler_type: str = "linear"  # "linear", "cosine", "constant"
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 1
    
    # Evaluation
    eval_steps: int = 500  # Evaluate every N steps
    eval_strategy: str = "steps"  # "steps" or "epoch"
    save_strategy: str = "steps"  # "steps" or "epoch"
    save_steps: int = 1000
    save_total_limit: int = 3  # Keep only N best checkpoints
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Logging
    logging_steps: int = 100
    log_level: str = "info"
    
    # Output
    output_dir: str = "experiments/runs/default"
    
    # Mixed precision
    fp16: bool = False
    
    # Seed
    seed: int = 42


class Trainer:
    """
    Main trainer class for distillation training.
    
    Handles:
    - Training loop with gradient accumulation
    - Evaluation and metric tracking
    - Checkpointing and model saving
    - Learning rate scheduling
    - Early stopping
    - Logging and experiment tracking
    """
    
    def __init__(self,
                 model,
                 train_dataloader: DataLoader,
                 eval_dataloader: Optional[DataLoader] = None,
                 distillation_strategy = None,
                 config: Optional[TrainingConfig] = None,
                 compute_metrics: Optional[Callable] = None):
        """
        Args:
            model: Student model to train
            train_dataloader: Training data loader
            eval_dataloader: Optional evaluation data loader
            distillation_strategy: Distillation strategy instance
            config: Training configuration
            compute_metrics: Optional function to compute evaluation metrics
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.distillation_strategy = distillation_strategy
        self.config = config or TrainingConfig()
        self.compute_metrics = compute_metrics
        
        # Setup device
        self.device = self.model.config.device
        
        # Setup output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        self._save_config()
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('inf')
        self.epochs_without_improvement = 0
        
        # Metrics tracking
        self.train_history = []
        self.eval_history = []
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.config.fp16 else None
        
        logger.info(f"Trainer initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Total training steps: {len(train_dataloader) * self.config.num_epochs}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with weight decay."""
        # Separate parameters with/without weight decay
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.model.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        total_steps = len(self.train_dataloader) * self.config.num_epochs
        
        if self.config.lr_scheduler_type == "linear":
            # Warmup + linear decay
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.config.warmup_steps
            )
            
            decay_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=total_steps - self.config.warmup_steps
            )
            
            scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, decay_scheduler],
                milestones=[self.config.warmup_steps]
            )
            
        elif self.config.lr_scheduler_type == "cosine":
            # Warmup + cosine annealing
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.config.warmup_steps
            )
            
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - self.config.warmup_steps,
                eta_min=0
            )
            
            scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.config.warmup_steps]
            )
            
        else:  # constant
            scheduler = None
        
        return scheduler
    
    def train(self):
        """
        Main training loop.
        
        Returns:
            Dictionary with training history
        """
        logger.info("=" * 70)
        logger.info("STARTING TRAINING")
        logger.info("=" * 70)
        logger.info(f"Epochs: {self.config.num_epochs}")
        logger.info(f"Train batches per epoch: {len(self.train_dataloader)}")
        if self.eval_dataloader:
            logger.info(f"Eval batches: {len(self.eval_dataloader)}")
        
        self.model.model.train()
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            logger.info(f"\n{'='*70}")
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            logger.info(f"{'='*70}")
            
            epoch_metrics = self._train_epoch()
            
            # Log epoch metrics
            logger.info(f"\nEpoch {epoch + 1} metrics:")
            for key, value in epoch_metrics.items():
                logger.info(f"  {key}: {value:.4f}")
            
            # Evaluate at end of epoch if configured
            if self.config.eval_strategy == "epoch" and self.eval_dataloader:
                eval_metrics = self.evaluate()
                self._log_eval_metrics(eval_metrics)
                
                # Check for improvement
                if self._check_improvement(eval_metrics):
                    self._save_checkpoint(is_best=True)
                else:
                    self.epochs_without_improvement += 1
                
                # Early stopping
                if self.epochs_without_improvement >= self.config.early_stopping_patience:
                    logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
            
            # Save checkpoint at end of epoch if configured
            if self.config.save_strategy == "epoch":
                self._save_checkpoint()
        
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 70)
        
        # Final evaluation
        if self.eval_dataloader:
            logger.info("\nRunning final evaluation...")
            final_metrics = self.evaluate()
            logger.info("\nFinal evaluation metrics:")
            for key, value in final_metrics.items():
                logger.info(f"  {key}: {value:.4f}")
        
        # Save final model
        self._save_checkpoint(is_final=True)
        
        return {
            'train_history': self.train_history,
            'eval_history': self.eval_history
        }
    
    def _train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with epoch metrics
        """
        epoch_loss = 0.0
        epoch_ce_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            loss_dict = self._training_step(batch)
            loss = loss_dict['total_loss']
            
            # Backward pass
            if self.config.fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights every gradient_accumulation_steps
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Clip gradients
                if self.config.fp16:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.model.parameters(),
                    self.config.max_grad_norm
                )
                
                # Optimizer step
                if self.config.fp16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # Scheduler step
                if self.scheduler:
                    self.scheduler.step()
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            # Track metrics
            epoch_loss += loss.item()
            epoch_ce_loss += loss_dict.get('ce_loss', loss).item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Logging
            if self.global_step % self.config.logging_steps == 0:
                self._log_training_step(loss_dict)
            
            # Evaluation
            if (self.config.eval_strategy == "steps" and 
                self.global_step % self.config.eval_steps == 0 and
                self.eval_dataloader):
                
                eval_metrics = self.evaluate()
                self._log_eval_metrics(eval_metrics)
                
                # Check for improvement
                if self._check_improvement(eval_metrics):
                    self._save_checkpoint(is_best=True)
                
                # Back to training mode
                self.model.model.train()
            
            # Checkpointing
            if (self.config.save_strategy == "steps" and
                self.global_step % self.config.save_steps == 0):
                self._save_checkpoint()
        
        # Compute epoch averages
        epoch_metrics = {
            'loss': epoch_loss / num_batches,
            'ce_loss': epoch_ce_loss / num_batches
        }
        
        self.train_history.append(epoch_metrics)
        
        return epoch_metrics
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Single training step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Dictionary with loss components
        """
        # Forward pass through model
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        # Compute distillation loss if strategy provided
        if self.distillation_strategy:
            loss_dict = self.distillation_strategy.compute_loss(
                outputs['logits'],
                batch['labels']
            )
        else:
            # Standard training without distillation
            loss_dict = {
                'total_loss': outputs['loss'],
                'ce_loss': outputs['loss']
            }
        
        return loss_dict
    
    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation on eval_dataloader.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.eval_dataloader:
            return {}
        
        logger.info("\nRunning evaluation...")
        self.model.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                total_loss += outputs['loss'].item()
                num_batches += 1
                
                # Generate predictions if compute_metrics provided
                if self.compute_metrics:
                    generated_ids = self.model.generate(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask']
                    )
                    
                    predictions = self.model.decode_batch(generated_ids)
                    
                    # Decode labels
                    labels = batch['labels'].clone()
                    labels[labels == -100] = self.model.tokenizer.pad_token_id
                    labels_text = self.model.decode_batch(labels)
                    
                    all_predictions.extend(predictions)
                    all_labels.extend(labels_text)
        
        # Compute metrics
        eval_metrics = {'eval_loss': total_loss / num_batches}
        
        if self.compute_metrics and all_predictions:
            additional_metrics = self.compute_metrics(all_predictions, all_labels)
            eval_metrics.update(additional_metrics)
        
        self.eval_history.append(eval_metrics)
        
        return eval_metrics
    
    def _check_improvement(self, eval_metrics: Dict[str, float]) -> bool:
        """
        Check if current metrics show improvement over best.
        
        Args:
            eval_metrics: Current evaluation metrics
            
        Returns:
            True if improved, False otherwise
        """
        # Use eval_loss as primary metric
        current_metric = eval_metrics.get('eval_loss', float('inf'))
        
        if current_metric < self.best_metric - self.config.early_stopping_threshold:
            self.best_metric = current_metric
            self.epochs_without_improvement = 0
            return True
        
        return False
    
    def _log_training_step(self, loss_dict: Dict[str, torch.Tensor]):
        """Log training step metrics."""
        log_str = f"Step {self.global_step} | "
        log_str += " | ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items()])
        log_str += f" | LR: {self.optimizer.param_groups[0]['lr']:.2e}"
        logger.info(log_str)
    
    def _log_eval_metrics(self, eval_metrics: Dict[str, float]):
        """Log evaluation metrics."""
        logger.info(f"\nEvaluation at step {self.global_step}:")
        for key, value in eval_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
    
    def _save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
            is_final: Whether this is the final model
        """
        if is_final:
            save_path = self.output_dir / "final_model"
        elif is_best:
            save_path = self.output_dir / "best_model"
        else:
            save_path = self.output_dir / f"checkpoint-{self.global_step}"
        
        logger.info(f"Saving checkpoint to {save_path}")
        
        # Save model
        self.model.save_model(str(save_path))
        
        # Save training state
        state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_metric': self.best_metric,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_history': self.train_history,
            'eval_history': self.eval_history
        }
        
        if self.scheduler:
            state['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(state, save_path / "training_state.pt")
        
        # Manage checkpoint limit
        if not (is_best or is_final):
            self._manage_checkpoints()
    
    def _manage_checkpoints(self):
        """Remove old checkpoints to respect save_total_limit."""
        checkpoints = sorted(
            [d for d in self.output_dir.iterdir() 
             if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda x: int(x.name.split("-")[1])
        )
        
        # Keep only the most recent save_total_limit checkpoints
        if len(checkpoints) > self.config.save_total_limit:
            for checkpoint in checkpoints[:-self.config.save_total_limit]:
                logger.info(f"Removing old checkpoint: {checkpoint}")
                import shutil
                shutil.rmtree(checkpoint)
    
    def _save_config(self):
        """Save training configuration to output directory."""
        config_path = self.output_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        logger.info(f"Training config saved to {config_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model and training state from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint_path = Path(checkpoint_path)
        
        # Load training state
        state_path = checkpoint_path / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path)
            
            self.global_step = state['global_step']
            self.epoch = state['epoch']
            self.best_metric = state['best_metric']
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            self.train_history = state['train_history']
            self.eval_history = state['eval_history']
            
            if self.scheduler and 'scheduler_state_dict' in state:
                self.scheduler.load_state_dict(state['scheduler_state_dict'])
            
            logger.info(f"Resumed from step {self.global_step}, epoch {self.epoch}")
        else:
            logger.warning("No training state found, starting fresh")


def create_trainer(
    model,
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader] = None,
    distillation_config: Optional[Dict] = None,
    training_config: Optional[TrainingConfig] = None,
    **kwargs
) -> Trainer:
    """
    Factory function to create trainer with distillation strategy.
    
    Args:
        model: Student model
        train_dataloader: Training data loader
        eval_dataloader: Optional evaluation data loader
        distillation_config: Configuration for distillation strategy
        training_config: Training configuration
        **kwargs: Additional arguments for Trainer
        
    Returns:
        Trainer instance
    """
    from .distillation import create_distillation_strategy
    
    # Create distillation strategy if config provided
    distillation_strategy = None
    if distillation_config:
        distillation_strategy = create_distillation_strategy(**distillation_config)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        distillation_strategy=distillation_strategy,
        config=training_config,
        **kwargs
    )
    
    return trainer