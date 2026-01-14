"""
Teacher Model Module for Reasoning Distillation Project

Implements various teacher models for knowledge distillation from datasets.
"""

import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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