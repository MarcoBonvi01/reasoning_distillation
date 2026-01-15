"""
Student Model Module for Reasoning Distillation Project

Implements FLAN-T5 student models with various sizes and configurations
for knowledge distillation from teacher LLMs.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    GenerationConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StudentConfig:
    """Configuration for student model"""
    model_name: str = "google/flan-t5-base"
    max_source_length: int = 512
    max_target_length: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Generation parameters
    num_beams: int = 4
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = False
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    
    # Training parameters
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    gradient_checkpointing: bool = False


class StudentModel(nn.Module):
    """
    FLAN-T5 Student Model for reasoning distillation.
    
    Wraps the HuggingFace T5ForConditionalGeneration model with
    additional utilities for distillation training and generation.
    """
    
    def __init__(self, config: Optional[StudentConfig] = None):
        super().__init__()
        self.config = config or StudentConfig()
        
        logger.info(f"Initializing student model: {self.config.model_name}")
        
        # Load model and tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.config.model_name,
            dtype=torch.float32
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=True
        )
        
        # Enable gradient checkpointing if requested (saves memory)
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        # Move to device
        self.model.to(self.config.device)
        
        # Setup generation config
        self.generation_config = GenerationConfig(
            max_length=self.config.max_target_length,
            num_beams=self.config.num_beams,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            do_sample=self.config.do_sample,
            repetition_penalty=self.config.repetition_penalty,
            length_penalty=self.config.length_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        logger.info(f"Model loaded successfully on {self.config.device}")
        logger.info(f"Model parameters: {self.count_parameters():,}")
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target token IDs [batch_size, target_len], optional
            
        Returns:
            Dictionary containing loss and logits
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        result = {
            'logits': outputs.logits
        }
        
        if labels is not None:
            result['loss'] = outputs.loss
        
        return result
    
    def generate(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                **generation_kwargs) -> torch.Tensor:
        """
        Generate sequences from input.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            **generation_kwargs: Override generation config parameters
            
        Returns:
            Generated token IDs [batch_size, generated_len]
        """
        # Merge generation kwargs with config
        gen_config = GenerationConfig(**{
            **self.generation_config.to_dict(),
            **generation_kwargs
        })
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=gen_config
            )
        
        return generated_ids
    
    def generate_with_labels(self,
                            input_ids: torch.Tensor,
                            attention_mask: torch.Tensor,
                            labels: torch.Tensor,
                            **generation_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences and compute loss against labels.
        Useful for evaluation.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Ground truth labels
            **generation_kwargs: Generation parameters
            
        Returns:
            Tuple of (generated_ids, loss)
        """
        # Get loss
        outputs = self.forward(input_ids, attention_mask, labels)
        loss = outputs['loss']
        
        # Generate
        generated_ids = self.generate(input_ids, attention_mask, **generation_kwargs)
        
        return generated_ids, loss
    
    def decode_batch(self, 
                    token_ids: torch.Tensor,
                    skip_special_tokens: bool = True) -> List[str]:
        """
        Decode a batch of token IDs to text.
        
        Args:
            token_ids: Token IDs [batch_size, seq_len]
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            List of decoded strings
        """
        return self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_memory_footprint(self) -> Dict[str, float]:
        """
        Get model memory footprint.
        
        Returns:
            Dictionary with memory statistics in MB
        """
        param_size = sum(p.numel() * p.element_size() 
                        for p in self.model.parameters()) / (1024**2)
        buffer_size = sum(b.numel() * b.element_size() 
                         for b in self.model.buffers()) / (1024**2)
        
        return {
            'parameters_mb': param_size,
            'buffers_mb': buffer_size,
            'total_mb': param_size + buffer_size
        }
    
    def freeze_encoder(self):
        """Freeze encoder parameters (useful for decoder-only fine-tuning)."""
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.model.encoder.parameters():
            param.requires_grad = True
        logger.info("Encoder unfrozen")
    
    def freeze_decoder(self):
        """Freeze decoder parameters."""
        for param in self.model.decoder.parameters():
            param.requires_grad = False
        logger.info("Decoder frozen")
    
    def unfreeze_decoder(self):
        """Unfreeze decoder parameters."""
        for param in self.model.decoder.parameters():
            param.requires_grad = True
        logger.info("Decoder unfrozen")
    
    def save_model(self, save_path: str):
        """
        Save model and tokenizer.
        
        Args:
            save_path: Directory path to save model
        """
        logger.info(f"Saving model to {save_path}")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info("Model saved successfully")
    
    @classmethod
    def load_model(cls, load_path: str, config: Optional[StudentConfig] = None):
        """
        Load a saved model.
        
        Args:
            load_path: Directory path containing saved model
            config: Optional config (will use saved config if not provided)
            
        Returns:
            StudentModel instance
        """
        logger.info(f"Loading model from {load_path}")
        
        if config is None:
            config = StudentConfig()
        
        # Update model name to load path
        config.model_name = load_path
        
        return cls(config)
    
    def get_model_info(self) -> Dict:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary with model details
        """
        memory = self.get_memory_footprint()
        
        return {
            'model_name': self.config.model_name,
            'device': self.config.device,
            'parameters': self.count_parameters(),
            'memory_mb': memory['total_mb'],
            'encoder_layers': len(self.model.encoder.block),
            'decoder_layers': len(self.model.decoder.block),
            'hidden_size': self.model.config.d_model,
            'num_heads': self.model.config.num_heads,
            'vocab_size': self.model.config.vocab_size
        }
    
def create_student_model(
    model_size: str = "base",
    device: Optional[str] = None,
    **kwargs
) -> StudentModel:
    """
    Factory function to create student models of different sizes.
    
    Args:
        model_size: Size of model ("small", "base", "large", "xl", "xxl")
        device: Device to load model on
        **kwargs: Additional config parameters
        
    Returns:
        StudentModel instance
    """
    model_name_map = {
        "small": "google/flan-t5-small",
        "base": "google/flan-t5-base",
    }
    
    if model_size not in model_name_map:
        raise ValueError(f"Unknown model size: {model_size}. "
                        f"Choose from {list(model_name_map.keys())}")
    
    config = StudentConfig(
        model_name=model_name_map[model_size],
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        **kwargs
    )
    
    return StudentModel(config)


def compare_model_sizes() -> None:
    """
    Print comparison of different FLAN-T5 model sizes.
    Useful for selecting appropriate student model.
    """
    sizes_info = {
        "small": {"params": "80M", "layers": "6/6", "hidden": 512},
        "base": {"params": "250M", "layers": "12/12", "hidden": 768},
    }
    
    print("=" * 70)
    print("FLAN-T5 MODEL SIZES")
    print("=" * 70)
    print(f"{'Size':<10} {'Parameters':<15} {'Layers (E/D)':<15} {'Hidden Size':<15}")
    print("-" * 70)
    
    for size, info in sizes_info.items():
        print(f"{size:<10} {info['params']:<15} {info['layers']:<15} {info['hidden']:<15}")
    
    print("=" * 70)
    print("\nRecommendations:")
    print("  • small/base: Fast training, good for experimentation")
    print("  • large: Balanced performance/efficiency")
    print("  • xl/xxl: Best quality, but requires significant compute")