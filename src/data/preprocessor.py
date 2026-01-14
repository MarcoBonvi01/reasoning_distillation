"""
Preprocessor Module for Reasoning Distillation Project

Handles tokenization, prompt formatting, and data transformation
for FLAN-T5 student models.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing"""
    model_name: str = "google/flan-t5-base"
    max_source_length: int = 512
    max_target_length: int = 256
    padding: str = "max_length"
    truncation: bool = True
    add_prefix: bool = True  # Add task-specific prefixes
    include_explanation: bool = True  # Include explanations in target


class TaskFormatter:
    """
    Handles task-specific prompt formatting.
    Creates structured inputs that help the model understand the task.
    """
    
    @staticmethod
    def format_nli(premise: str, 
                   hypothesis: str,
                   label: Optional[int] = None,
                   explanation: Optional[str] = None,
                   include_label_in_input: bool = False) -> Tuple[str, Optional[str]]:
        """
        Format Natural Language Inference task.
        
        Args:
            premise: The premise sentence
            hypothesis: The hypothesis sentence
            label: Label (0=entailment, 1=neutral, 2=contradiction)
            explanation: Optional explanation text
            include_label_in_input: Whether to include label in input (for teacher forcing)
            
        Returns:
            Tuple of (formatted_input, formatted_target)
        """
        label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
        
        # Format input
        source = f"nli premise: {premise} hypothesis: {hypothesis}"
        
        # Format target (label + explanation)
        if label is not None:
            label_text = label_map[label]
            if explanation and explanation.strip():
                target = f"{label_text} explanation: {explanation}"
            else:
                target = label_text
        else:
            target = None
            
        return source, target
    
    @staticmethod
    
    @staticmethod
    def format_multitask(task_type: str, **kwargs) -> Tuple[str, Optional[str]]:
        """
        Route to appropriate formatter based on task type.
        
        Args:
            task_type: Type of task ('nli' or 'instruction_following')
            **kwargs: Task-specific arguments
            
        Returns:
            Tuple of (formatted_input, formatted_target)
        """
        if task_type == 'nli':
            return TaskFormatter.format_nli(**kwargs)
        else:
            raise ValueError(f"Unknown task type: {task_type}")


class ReasoningPreprocessor:
    """
    Main preprocessor for reasoning distillation.
    Handles tokenization and formatting for FLAN-T5 models.
    """
    
    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
        
        # Load tokenizer
        logger.info(f"Loading tokenizer: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=True
        )
        
        self.formatter = TaskFormatter()
        
    def preprocess_esnli_sample(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """
        Preprocess a single e-SNLI sample.
        
        Args:
            sample: Raw e-SNLI sample
            
        Returns:
            Dictionary with tokenized inputs and labels
        """
        # Format the sample
        source, target = self.formatter.format_nli(
            premise=sample['premise'],
            hypothesis=sample['hypothesis'],
            label=sample['label'],
            explanation=sample.get('explanation_1', '') if self.config.include_explanation else None
        )
        
        # Tokenize
        return self._tokenize_pair(source, target)
    
    
    def _tokenize_pair(self, source: str, target: Optional[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize source and target pair.
        
        Args:
            source: Input text
            target: Target text (can be None for inference)
            
        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        # Tokenize source
        source_encoding = self.tokenizer(
            source,
            max_length=self.config.max_source_length,
            padding=self.config.padding,
            truncation=self.config.truncation,
            return_tensors="pt"
        )
        
        result = {
            "input_ids": source_encoding["input_ids"].squeeze(0),
            "attention_mask": source_encoding["attention_mask"].squeeze(0)
        }
        
        # Tokenize target if provided
        if target is not None:
            target_encoding = self.tokenizer(
                target,
                max_length=self.config.max_target_length,
                padding=self.config.padding,
                truncation=self.config.truncation,
                return_tensors="pt"
            )
            
            # Replace padding token id with -100 so it's ignored in loss
            labels = target_encoding["input_ids"].squeeze(0)
            labels[labels == self.tokenizer.pad_token_id] = -100
            result["labels"] = labels
        
        return result
    
    def preprocess_batch_esnli(self, samples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Preprocess a batch of e-SNLI samples.
        
        Args:
            samples: List of raw e-SNLI samples
            
        Returns:
            Batched dictionary with tokenized inputs
        """
        # Format all samples
        sources = []
        targets = []
        
        for sample in samples:
            source, target = self.formatter.format_nli(
                premise=sample['premise'],
                hypothesis=sample['hypothesis'],
                label=sample['label'],
                explanation=sample.get('explanation_1', '') if self.config.include_explanation else None
            )
            sources.append(source)
            targets.append(target)
        
        return self._tokenize_batch(sources, targets)
    
    
    def _tokenize_batch(self, 
                       sources: List[str], 
                       targets: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize batches of sources and targets.
        
        Args:
            sources: List of input texts
            targets: List of target texts
            
        Returns:
            Batched dictionary with tokenized data
        """
        # Tokenize sources
        source_encoding = self.tokenizer(
            sources,
            max_length=self.config.max_source_length,
            padding=self.config.padding,
            truncation=self.config.truncation,
            return_tensors="pt"
        )
        
        result = {
            "input_ids": source_encoding["input_ids"],
            "attention_mask": source_encoding["attention_mask"]
        }
        
        # Tokenize targets
        target_encoding = self.tokenizer(
            targets,
            max_length=self.config.max_target_length,
            padding=self.config.padding,
            truncation=self.config.truncation,
            return_tensors="pt"
        )
        
        # Replace padding with -100 for loss computation
        labels = target_encoding["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        result["labels"] = labels
        
        return result
    
    def decode_prediction(self, 
                         token_ids: Union[torch.Tensor, List[int]],
                         skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text string
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy()
        
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )
    
    def extract_label_from_prediction(self, prediction: str) -> Optional[str]:
        """
        Extract the predicted label from model output.
        Useful for NLI tasks where output is "label explanation: ..."
        
        Args:
            prediction: Model's generated text
            
        Returns:
            Extracted label or None
        """
        prediction = prediction.lower().strip()
        
        # Check for NLI labels at the start
        for label in ['entailment', 'neutral', 'contradiction']:
            if prediction.startswith(label):
                return label
        
        return None
    
    def extract_explanation_from_prediction(self, prediction: str) -> Optional[str]:
        """
        Extract the explanation from model output.
        
        Args:
            prediction: Model's generated text
            
        Returns:
            Extracted explanation or None
        """
        # Look for "explanation:" marker
        if "explanation:" in prediction.lower():
            parts = prediction.lower().split("explanation:", 1)
            if len(parts) == 2:
                return parts[1].strip()
        
        # If no marker, return everything after the label
        for label in ['entailment', 'neutral', 'contradiction']:
            if prediction.lower().startswith(label):
                return prediction[len(label):].strip()
        
        return prediction.strip()
    
    def get_tokenizer_info(self) -> Dict:
        """
        Get information about the tokenizer.
        
        Returns:
            Dictionary with tokenizer details
        """
        return {
            "model_name": self.config.model_name,
            "vocab_size": self.tokenizer.vocab_size,
            "max_source_length": self.config.max_source_length,
            "max_target_length": self.config.max_target_length,
            "pad_token": self.tokenizer.pad_token,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token": self.tokenizer.eos_token,
            "eos_token_id": self.tokenizer.eos_token_id
        }


# Utility function for quick preprocessing
def quick_preprocess_sample(sample: Dict, 
                            task_type: str = "nli",
                            model_name: str = "google/flan-t5-base") -> Dict[str, torch.Tensor]:
    """
    Quick helper to preprocess a single sample.
    
    Args:
        sample: Raw sample dictionary
        task_type: Type of task ('nli' or 'instruction_following')
        model_name: Model name for tokenizer
        
    Returns:
        Preprocessed sample with tokenized inputs
    """
    config = PreprocessConfig(model_name=model_name)
    preprocessor = ReasoningPreprocessor(config)
    
    if task_type == "nli":
        return preprocessor.preprocess_esnli_sample(sample)
    else:
        raise ValueError(f"Unknown task type: {task_type}")