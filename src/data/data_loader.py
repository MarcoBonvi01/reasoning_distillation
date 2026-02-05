"""
Data Loader Module for Reasoning Distillation Project

Handles downloading, parsing, and validation of datasets:
- e-SNLI (Natural Language Inference with explanations)
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
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
    Supports e-SNLI instruction datasets.
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
        Load e-SNLI dataset from GitHub repository.
        
        Source: https://github.com/OanaMariaCamburu/e-SNLI
        
        Returns normalized format:
            - premise: string (from Sentence1)
            - hypothesis: string (from Sentence2)  
            - label: int (0=entailment, 1=neutral, 2=contradiction)
            - explanation_1, explanation_2, explanation_3: string
        
        Args:
            split: Specific split to load ('train', 'validation', 'test') or None for all
        
        Returns:
            DatasetDict with normalized columns
        """
        logger.info("Loading e-SNLI dataset from GitHub (OanaMariaCamburu/e-SNLI)...")
        
        github_base = "https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset"
        
        def normalize_esnli(ds):
            """Normalize column names and convert labels"""
            def process(example):
                label_map = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
                label_str = str(example.get('gold_label', 'neutral')).strip().lower()
                
                return {
                    'premise': str(example.get('Sentence1', '')).strip(),
                    'hypothesis': str(example.get('Sentence2', '')).strip(),
                    'label': label_map.get(label_str, 1),
                    'explanation_1': str(example.get('Explanation_1', '')).strip(),
                    'explanation_2': str(example.get('Explanation_2', '')).strip(),
                    'explanation_3': str(example.get('Explanation_3', '')).strip()
                }
            
            return ds.map(process, remove_columns=ds.column_names)
        
        try:
            # Train: 2 files da concatenare
            train1 = load_dataset("csv", data_files=f"{github_base}/esnli_train_1.csv", cache_dir=self.config.cache_dir)['train']
            train2 = load_dataset("csv", data_files=f"{github_base}/esnli_train_2.csv", cache_dir=self.config.cache_dir)['train']
            train = concatenate_datasets([normalize_esnli(train1), normalize_esnli(train2)])
            
            # Validation
            val = load_dataset("csv", data_files=f"{github_base}/esnli_dev.csv", cache_dir=self.config.cache_dir)['train']
            validation = normalize_esnli(val)
            
            # Test
            tst = load_dataset("csv", data_files=f"{github_base}/esnli_test.csv", cache_dir=self.config.cache_dir)['train']
            test = normalize_esnli(tst)
            
            dataset = DatasetDict({
                'train': train,
                'validation': validation,
                'test': test
            })
            
            logger.info(f"Loaded e-SNLI: train={len(train)}, val={len(validation)}, test={len(test)}")
            
        except Exception as e:
            logger.error(f"Error loading e-SNLI: {e}")
            raise RuntimeError(f"Failed to load e-SNLI from GitHub: {e}") from e
        
        if split:
            if split not in dataset:
                raise ValueError(f"Split '{split}' not found. Available: {list(dataset.keys())}")
            dataset = DatasetDict({split: dataset[split]})
        
        logger.info(f"e-SNLI loaded successfully. Splits: {list(dataset.keys())}")
        logger.info(f"Sample counts: {[(k, len(v)) for k, v in dataset.items()]}")
        
        return dataset
        
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
    
    def save_processed_data(self, 
                           dataset: Dataset,
                           name: str,
                           split: str = "train"):
        """
        Save processed dataset to disk.
        
        Args:
            dataset: Dataset to save
            name: Dataset name (e.g., 'esnli')
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