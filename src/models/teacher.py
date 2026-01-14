"""
Teacher Model Module for Reasoning Distillation Project

Wrapper for teacher LLMs (e.g., Qwen, LLaMA) to generate explanations
and reasoning traces for distillation. Also handles dataset-as-teacher
scenarios where pre-generated explanations are used.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TeacherConfig:
    """Configuration for teacher model"""
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    load_in_8bit: bool = False  # Quantization for memory efficiency
    load_in_4bit: bool = False
    max_length: int = 2048
    
    # Generation parameters
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    do_sample: bool = True
    num_return_sequences: int = 1
    
    # Teacher-specific
    use_dataset_as_teacher: bool = True  # Use pre-generated explanations
    generate_augmentations: bool = False  # Generate additional examples


class TeacherModel:
    """
    Teacher model wrapper for large LLMs.
    
    Supports two modes:
    1. Dataset-as-teacher: Use pre-generated explanations from datasets
    2. Online generation: Generate explanations on-the-fly (expensive)
    """
    
    def __init__(self, config: Optional[TeacherConfig] = None):
        self.config = config or TeacherConfig()
        
        # If using dataset as teacher, don't load model
        if self.config.use_dataset_as_teacher:
            logger.info("Using dataset as teacher (no model loading)")
            self.model = None
            self.tokenizer = None
            return
        
        logger.info(f"Initializing teacher model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        # Setup quantization config if requested
        quantization_config = None
        if self.config.load_in_4bit or self.config.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                
                if self.config.load_in_4bit:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True
                    )
                    logger.info("Loading in 4-bit")
                else:
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    logger.info("Loading in 8-bit")
            except ImportError:
                logger.warning("bitsandbytes not available, loading in full precision")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quantization_config,
            device_map="auto" if quantization_config else None,
            torch_dtype=torch.float16 if not quantization_config else None,
            trust_remote_code=True
        )
        
        if not quantization_config:
            self.model.to(self.config.device)
        
        self.model.eval()  # Teacher is always in eval mode
        
        logger.info(f"Teacher model loaded on {self.config.device}")
    
    def generate_explanation(self,
                           premise: str,
                           hypothesis: str,
                           label: str,
                           task_type: str = "nli") -> str:
        """
        Generate explanation for an NLI example.
        
        Args:
            premise: The premise sentence
            hypothesis: The hypothesis sentence
            label: The label (entailment/neutral/contradiction)
            task_type: Type of reasoning task
            
        Returns:
            Generated explanation string
        """
        if self.config.use_dataset_as_teacher:
            raise RuntimeError("Cannot generate with dataset-as-teacher mode")
        
        # Create prompt
        prompt = self._create_nli_prompt(premise, hypothesis, label)
        
        # Generate
        explanation = self._generate(prompt)
        
        return explanation
    
    def generate_instruction_response(self,
                                     instruction: str,
                                     input_text: str = "") -> str:
        """
        Generate response for an instruction-following task.
        
        Args:
            instruction: The instruction
            input_text: Optional input context
            
        Returns:
            Generated response
        """
        if self.config.use_dataset_as_teacher:
            raise RuntimeError("Cannot generate with dataset-as-teacher mode")
        
        # Create prompt
        prompt = self._create_instruction_prompt(instruction, input_text)
        
        # Generate
        response = self._generate(prompt)
        
        return response
    
    def _create_nli_prompt(self, 
                          premise: str,
                          hypothesis: str,
                          label: str) -> str:
        """Create prompt for NLI explanation generation."""
        return f"""Given the following premise and hypothesis, explain why the relationship is "{label}".

Premise: {premise}
Hypothesis: {hypothesis}
Label: {label}

Provide a clear, concise explanation:"""
    
    def _create_instruction_prompt(self,
                                  instruction: str,
                                  input_text: str) -> str:
        """Create prompt for instruction following."""
        if input_text:
            return f"""Instruction: {instruction}

Input: {input_text}

Response:"""
        else:
            return f"""Instruction: {instruction}

Response:"""
    
    def _generate(self, prompt: str) -> str:
        """
        Generate text from prompt using teacher model.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        if self.model is None:
            raise RuntimeError("Teacher model not loaded")
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode (remove prompt)
        input_length = inputs['input_ids'].shape[1]
        generated_text = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def get_model_info(self) -> Dict:
        """Get teacher model information."""
        if self.config.use_dataset_as_teacher:
            return {
                'mode': 'dataset_as_teacher',
                'model_name': 'N/A (using pre-generated data)'
            }
        
        return {
            'mode': 'online_generation',
            'model_name': self.config.model_name,
            'device': self.config.device,
            'quantization': '4-bit' if self.config.load_in_4bit else (
                '8-bit' if self.config.load_in_8bit else 'none'
            )
        }


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


class MultiTeacherEnsemble:
    """
    Combines knowledge from multiple teachers.
    Useful for dataset mixing or multi-teacher distillation.
    """
    
    def __init__(self, teachers: List[Union[TeacherModel, DatasetTeacher]]):
        """
        Args:
            teachers: List of teacher instances
        """
        self.teachers = teachers
        self.num_teachers = len(teachers)
        
        logger.info(f"Initialized multi-teacher ensemble with {self.num_teachers} teachers")
    
    def aggregate_explanations(self, 
                              explanations: List[str],
                              strategy: str = "concatenate") -> str:
        """
        Aggregate explanations from multiple teachers.
        
        Args:
            explanations: List of explanation strings
            strategy: How to combine ("concatenate", "vote", "longest")
            
        Returns:
            Combined explanation
        """
        if strategy == "concatenate":
            # Join all explanations
            return " ".join(exp for exp in explanations if exp.strip())
        
        elif strategy == "longest":
            # Return longest explanation
            return max(explanations, key=len)
        
        elif strategy == "vote":
            # Return most common (simple majority)
            from collections import Counter
            counts = Counter(explanations)
            return counts.most_common(1)[0][0]
        
        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")


def create_teacher_from_config(config: Dict) -> Union[TeacherModel, DatasetTeacher]:
    """
    Factory function to create appropriate teacher based on config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Teacher instance
    """
    use_dataset = config.get('use_dataset_as_teacher', True)
    
    if use_dataset:
        return DatasetTeacher()
    else:
        teacher_config = TeacherConfig(**config)
        return TeacherModel(teacher_config)


def compare_teacher_modes():
    """
    Print comparison of different teacher modes.
    """
    print("=" * 70)
    print("TEACHER MODEL MODES")
    print("=" * 70)
    
    print("\n1. Dataset-as-Teacher (RECOMMENDED)")
    print("   ✓ Uses pre-generated explanations from e-SNLI")
    print("   ✓ No additional compute cost")
    print("   ✓ High-quality human or LLM-generated explanations")
    print("   ✓ Reproducible and fast")
    print("   ✗ Limited to existing dataset coverage")
    
    print("\n2. Online Generation")
    print("   ✓ Can generate explanations for any input")
    print("   ✓ Flexible for augmentation")
    print("   ✗ Requires large model (7B+ parameters)")
    print("   ✗ Slow and computationally expensive")
    print("   ✗ Quality varies with prompt engineering")
    
    print("\n3. Multi-Teacher Ensemble")
    print("   ✓ Combines knowledge from multiple sources")
    print("   ✓ More robust reasoning patterns")
    print("   ✗ Increased complexity")
    print("   ✗ May introduce inconsistencies")
    
    print("\n" + "=" * 70)
    print("\nFor this project: Use Dataset-as-Teacher (mode 1)")
    print("Reasons:")
    print("  • e-SNLI has high-quality human explanations")
    print("  • Much faster and more reproducible")
    print("  • Sufficient for demonstrating distillation")
    print("=" * 70)


# Recommended teacher configuration
RECOMMENDED_CONFIG = {
    'use_dataset_as_teacher': True,
    'generate_augmentations': False
}