"""
Teacher Model Module for Reasoning Distillation Project

Implements various teacher models for knowledge distillation from datasets
and neural teacher models like FLAN-T5-XL.
"""

import logging
from typing import Dict, Optional, Tuple
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
class TeacherConfig:
    """Configuration for teacher model"""
    model_name: str = "google/flan-t5-xl"
    max_source_length: int = 512
    max_target_length: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use float16 for efficiency (teacher doesn't need gradients)
    use_fp16: bool = True
    
    # Generation parameters
    num_beams: int = 4
    temperature: float = 1.0
    do_sample: bool = False


class FlanT5Teacher(nn.Module):
    """
    FLAN-T5-XL Teacher Model for Knowledge Distillation.
    
    This model is used to generate soft logits (probability distributions)
    that the student model learns to mimic via KL divergence.
    
    The teacher is always in eval mode with frozen parameters.
    """
    
    def __init__(self, config: Optional[TeacherConfig] = None):
        super().__init__()
        self.config = config or TeacherConfig()
        
        logger.info(f"Initializing teacher model: {self.config.model_name}")
        
        # Determine dtype
        dtype = torch.float16 if self.config.use_fp16 and torch.cuda.is_available() else torch.float32
        
        # Load model and tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=True
        )
        
        # Move to device
        self.model.to(self.config.device)
        
        # Set to eval mode and freeze parameters (teacher never trains)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Setup generation config
        self.generation_config = GenerationConfig(
            max_length=self.config.max_target_length,
            num_beams=self.config.num_beams,
            temperature=self.config.temperature,
            do_sample=self.config.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        logger.info(f"Teacher model loaded on {self.config.device} with dtype {dtype}")
        logger.info(f"Teacher parameters: {self.count_parameters():,} (frozen)")
    
    def count_parameters(self) -> int:
        """Count total model parameters."""
        return sum(p.numel() for p in self.model.parameters())
    
    @torch.no_grad()
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                decoder_input_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass to get teacher logits.
        
        Args:
            input_ids: Encoder input token IDs [batch_size, src_len]
            attention_mask: Encoder attention mask [batch_size, src_len]
            decoder_input_ids: Decoder input IDs [batch_size, tgt_len]
            labels: Target labels [batch_size, tgt_len]
            
        Returns:
            Dictionary containing logits (soft targets)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )
        
        return {
            'logits': outputs.logits,
            'loss': outputs.loss if labels is not None else None
        }
    
    @torch.no_grad()
    def get_soft_labels(self,
                        input_ids: torch.Tensor,
                        attention_mask: torch.Tensor,
                        decoder_input_ids: torch.Tensor,
                        temperature: float = 1.0) -> torch.Tensor:
        """
        Get soft probability distributions from the teacher.
        
        Args:
            input_ids: Encoder input token IDs
            attention_mask: Encoder attention mask
            decoder_input_ids: Decoder input IDs
            temperature: Temperature for softening probabilities
            
        Returns:
            Soft probabilities [batch_size, seq_len, vocab_size]
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids
        )
        
        # Apply temperature and softmax
        soft_labels = torch.softmax(outputs.logits / temperature, dim=-1)
        
        return soft_labels
    
    @torch.no_grad()
    def generate(self,
                 input_ids: torch.Tensor,
                 attention_mask: torch.Tensor,
                 **generation_kwargs) -> torch.Tensor:
        """
        Generate sequences from input.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            **generation_kwargs: Override generation config
            
        Returns:
            Generated token IDs
        """
        gen_config = GenerationConfig(**{
            **self.generation_config.to_dict(),
            **generation_kwargs
        })
        
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=gen_config
        )
    
    def to(self, device):
        """Move model to device."""
        self.model.to(device)
        self.config.device = str(device)
        return self


class DatasetTeacher:
    """
    Uses pre-existing datasets as implicit teachers.
    This is the recommended approach as it's much more efficient.
    """
    
    def __init__(self):
        logger.info("Initialized DatasetTeacher (using pre-generated explanations)")
    
    def extract_teacher_knowledge(self, sample: Dict, task_type: str = "nli") -> Dict:
        """
        Extract teacher knowledge from a dataset sample.
        
        Args:
            sample: Dataset sample with explanations
            task_type: Type of task
            
        Returns:
            Dictionary with teacher's reasoning
        """
        if task_type == "nli":
            return {
                'label': sample.get('label'),
                'explanation': sample.get('explanation_1', ''),
                'alternative_explanations': [
                    sample.get('explanation_2', ''),
                    sample.get('explanation_3', '')
                ]
            }
        elif task_type == "instruction_following":
            return {
                'response': sample.get('output', ''),
                'instruction': sample.get('instruction', ''),
                'input': sample.get('input', '')
            }
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def get_best_explanation(self, sample: Dict) -> str:
        """
        Get the best available explanation from a sample.
        For e-SNLI, chooses the first non-empty explanation.
        
        Args:
            sample: Dataset sample
            
        Returns:
            Best explanation string
        """
        # Try explanation_1 first
        exp1 = sample.get('explanation_1', '').strip()
        if exp1:
            return exp1
        
        # Fallback to other explanations
        exp2 = sample.get('explanation_2', '').strip()
        if exp2:
            return exp2
        
        exp3 = sample.get('explanation_3', '').strip()
        if exp3:
            return exp3
        
        return ""


def create_teacher(
    teacher_type: str = "flan-t5-xl",
    config: Optional[TeacherConfig] = None,
    **kwargs
):
    """
    Factory function to create a teacher model.
    
    Args:
        teacher_type: Type of teacher
            - "flan-t5-xl": FLAN-T5-XL neural teacher (default)
            - "flan-t5-large": FLAN-T5-Large neural teacher
            - "flan-t5-base": FLAN-T5-Base neural teacher
            - "dataset": Dataset-based teacher (no model)
        config: Teacher configuration
        **kwargs: Additional configuration overrides
        
    Returns:
        Teacher instance
    """
    if teacher_type == "dataset":
        return DatasetTeacher()
    
    # Neural teacher models
    model_map = {
        "flan-t5-xl": "google/flan-t5-xl",
        "flan-t5-large": "google/flan-t5-large",
        "flan-t5-base": "google/flan-t5-base",
    }
    
    if teacher_type in model_map:
        if config is None:
            config = TeacherConfig(model_name=model_map[teacher_type])
        else:
            config.model_name = model_map[teacher_type]
        
        # Apply any kwargs overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return FlanT5Teacher(config)
    
    else:
        raise ValueError(f"Unknown teacher type: {teacher_type}. "
                        f"Choose from: {list(model_map.keys()) + ['dataset']}")