"""
Data Loader Module for Reasoning Distillation Project

Handles downloading, parsing, and validation of datasets:
- e-SNLI (Natural Language Inference with explanations)
- Alpaca/Self-Instruct style datasets
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset management"""
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    cache_dir: str = "data/cache"
    test_size: float = 0.1
    val_size: float = 0.1
    random_seed: int = 42


class TeacherDataLoader:
    """
    Main class for loading and managing teacher datasets.
    Supports e-SNLI and Alpaca-style instruction datasets.
    """
    
    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()
        self._setup_directories()
        
    def _setup_directories(self):
        """Create necessary directories if they don't exist"""
        for dir_path in [self.config.raw_data_dir, 
                        self.config.processed_data_dir,
                        self.config.cache_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
    def load_esnli(self, split: Optional[str] = None) -> DatasetDict:
        """
        Load e-SNLI dataset with explanations.
        
        Args:
            split: Specific split to load ('train', 'validation', 'test') or None for all
            
        Returns:
            DatasetDict containing the requested splits
        """
        logger.info("Loading e-SNLI dataset...")
        
        try:
            # Try loading with script support (for older datasets versions)
            try:
                dataset = load_dataset(
                    "esnli",
                    cache_dir=self.config.cache_dir,
                    trust_remote_code=True
                )
            except Exception as e:
                # If that fails, try without trust_remote_code
                if "no longer supported" in str(e).lower() or "script" in str(e).lower():
                    logger.warning(f"Dataset script loading failed: {e}. Trying alternative method...")
                    dataset = load_dataset(
                        "esnli",
                        cache_dir=self.config.cache_dir
                    )
                else:
                    raise
            
            if split:
                dataset = {split: dataset[split]}
                
            # Save raw data
            save_path = Path(self.config.raw_data_dir) / "e-snli"
            save_path.mkdir(exist_ok=True)
            
            logger.info(f"e-SNLI loaded successfully. Splits: {list(dataset.keys())}")
            logger.info(f"Sample counts: {[(k, len(v)) for k, v in dataset.items()]}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading e-SNLI: {e}")
            raise
    
    def load_alpaca(self, 
                   dataset_name: str = "tatsu-lab/alpaca",
                   max_samples: Optional[int] = None) -> Dataset:
        """
        Load Alpaca-style instruction dataset.
        
        Args:
            dataset_name: HuggingFace dataset identifier
            max_samples: Maximum number of samples to load (for testing)
            
        Returns:
            Dataset object
        """
        logger.info(f"Loading Alpaca dataset: {dataset_name}...")
        
        try:
            dataset = load_dataset(
                dataset_name,
                cache_dir=self.config.cache_dir,
                split="train"
            )
            
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            # Save raw data
            save_path = Path(self.config.raw_data_dir) / "alpaca"
            save_path.mkdir(exist_ok=True)
            
            logger.info(f"Alpaca loaded successfully. Samples: {len(dataset)}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading Alpaca: {e}")
            raise
    
    def parse_esnli_sample(self, sample: Dict) -> Dict:
        """
        Parse a single e-SNLI sample into standard format.
        
        Args:
            sample: Raw e-SNLI sample
            
        Returns:
            Parsed sample with keys: premise, hypothesis, label, explanation
        """
        # e-SNLI has three explanations - we'll use the first one
        explanation_keys = ['explanation_1', 'explanation_2', 'explanation_3']
        explanations = [sample.get(key, '') for key in explanation_keys]
        # Filter out empty explanations
        valid_explanations = [exp for exp in explanations if exp and exp.strip()]
        
        return {
            'premise': sample['premise'],
            'hypothesis': sample['hypothesis'],
            'label': sample['label'],  # 0: entailment, 1: neutral, 2: contradiction
            'explanation': valid_explanations[0] if valid_explanations else '',
            'all_explanations': valid_explanations,
            'task_type': 'nli'
        }
    
    def parse_alpaca_sample(self, sample: Dict) -> Dict:
        """
        Parse a single Alpaca sample into standard format.
        
        Args:
            sample: Raw Alpaca sample
            
        Returns:
            Parsed sample with keys: instruction, input, output
        """
        return {
            'instruction': sample.get('instruction', ''),
            'input': sample.get('input', ''),
            'output': sample.get('output', ''),
            'task_type': 'instruction_following'
        }
    
    def validate_esnli(self, dataset: DatasetDict) -> Dict[str, any]:
        """
        Validate e-SNLI dataset and compute statistics.
        
        Args:
            dataset: e-SNLI DatasetDict
            
        Returns:
            Dictionary with validation statistics
        """
        logger.info("Validating e-SNLI dataset...")
        
        stats = {
            'splits': {},
            'label_distribution': {},
            'explanation_stats': {}
        }
        
        for split_name, split_data in dataset.items():
            n_samples = len(split_data)
            
            # Parse samples
            parsed = [self.parse_esnli_sample(sample) for sample in tqdm(
                split_data, desc=f"Parsing {split_name}"
            )]
            
            # Label distribution
            labels = [s['label'] for s in parsed]
            label_counts = pd.Series(labels).value_counts().to_dict()
            
            # Explanation statistics
            exp_lengths = [len(s['explanation'].split()) for s in parsed if s['explanation']]
            
            stats['splits'][split_name] = n_samples
            stats['label_distribution'][split_name] = label_counts
            stats['explanation_stats'][split_name] = {
                'mean_length': sum(exp_lengths) / len(exp_lengths) if exp_lengths else 0,
                'min_length': min(exp_lengths) if exp_lengths else 0,
                'max_length': max(exp_lengths) if exp_lengths else 0,
                'samples_with_explanation': len(exp_lengths)
            }
        
        logger.info("Validation complete.")
        return stats
    
    def validate_alpaca(self, dataset: Dataset) -> Dict[str, any]:
        """
        Validate Alpaca dataset and compute statistics.
        
        Args:
            dataset: Alpaca Dataset
            
        Returns:
            Dictionary with validation statistics
        """
        logger.info("Validating Alpaca dataset...")
        
        parsed = [self.parse_alpaca_sample(sample) for sample in tqdm(
            dataset, desc="Parsing Alpaca"
        )]
        
        # Compute statistics
        instruction_lengths = [len(s['instruction'].split()) for s in parsed]
        output_lengths = [len(s['output'].split()) for s in parsed]
        samples_with_input = sum(1 for s in parsed if s['input'].strip())
        
        stats = {
            'total_samples': len(parsed),
            'samples_with_input': samples_with_input,
            'instruction_length': {
                'mean': sum(instruction_lengths) / len(instruction_lengths),
                'min': min(instruction_lengths),
                'max': max(instruction_lengths)
            },
            'output_length': {
                'mean': sum(output_lengths) / len(output_lengths),
                'min': min(output_lengths),
                'max': max(output_lengths)
            }
        }
        
        logger.info("Validation complete.")
        return stats
    
    def save_processed_data(self, 
                           dataset: Dataset,
                           name: str,
                           split: str = "train"):
        """
        Save processed dataset to disk.
        
        Args:
            dataset: Dataset to save
            name: Dataset name (e.g., 'esnli', 'alpaca')
            split: Split name (e.g., 'train', 'val', 'test')
        """
        save_path = Path(self.config.processed_data_dir) / name
        save_path.mkdir(exist_ok=True)
        
        output_file = save_path / f"{split}.json"
        
        # Convert to list of dicts for JSON serialization
        data = [dict(sample) for sample in dataset]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(data)} samples to {output_file}")
    
    def get_sample_examples(self, 
                           dataset: Dataset, 
                           n_samples: int = 3) -> List[Dict]:
        """
        Get random sample examples for inspection.
        
        Args:
            dataset: Dataset to sample from
            n_samples: Number of samples to return
            
        Returns:
            List of sample dictionaries
        """
        import random
        indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
        return [dataset[i] for i in indices]


# Utility function for quick loading
def quick_load_esnli(max_samples: Optional[int] = None) -> Tuple[DatasetDict, Dict]:
    """
    Quick helper to load and validate e-SNLI.
    
    Args:
        max_samples: Optional limit on samples per split
        
    Returns:
        Tuple of (dataset, validation_stats)
    """
    loader = TeacherDataLoader()
    dataset = loader.load_esnli()
    
    if max_samples:
        dataset = DatasetDict({
            k: v.select(range(min(max_samples, len(v))))
            for k, v in dataset.items()
        })
    
    stats = loader.validate_esnli(dataset)
    return dataset, stats


def quick_load_alpaca(max_samples: Optional[int] = 1000) -> Tuple[Dataset, Dict]:
    """
    Quick helper to load and validate Alpaca.
    
    Args:
        max_samples: Optional limit on samples
        
    Returns:
        Tuple of (dataset, validation_stats)
    """
    loader = TeacherDataLoader()
    dataset = loader.load_alpaca(max_samples=max_samples)
    stats = loader.validate_alpaca(dataset)
    return dataset, stats