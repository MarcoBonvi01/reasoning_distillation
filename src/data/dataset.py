"""
Dataset Module for Reasoning Distillation Project

PyTorch Dataset classes for e-SNLI dataset,
with support for caching and data augmentation.
"""

import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datasets import Dataset as HFDataset, DatasetDict
import json

from .preprocessor import ReasoningPreprocessor, PreprocessConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ESNLIDataset(Dataset):
    """
    PyTorch Dataset for e-SNLI (Natural Language Inference with explanations).
    """
    
    def __init__(self,
                 data: Union[HFDataset, List[Dict]],
                 preprocessor: ReasoningPreprocessor,
                 cache_dir: Optional[str] = None,
                 use_cache: bool = True):
        """
        Args:
            data: HuggingFace Dataset or list of dictionaries
            preprocessor: ReasoningPreprocessor instance
            cache_dir: Directory for caching preprocessed samples
            use_cache: Whether to use caching
        """
        self.data = data
        self.preprocessor = preprocessor
        self.use_cache = use_cache
        self.cache = {} if use_cache else None
        
        if cache_dir and use_cache:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
        
        logger.info(f"Initialized ESNLIDataset with {len(data)} samples")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a preprocessed sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        # Check cache first
        if self.cache is not None and idx in self.cache:
            return self.cache[idx]
        
        # Get raw sample
        sample = self.data[idx]
        
        # Preprocess
        processed = self.preprocessor.preprocess_esnli_sample(sample)
        
        # Cache if enabled
        if self.cache is not None:
            self.cache[idx] = processed
        
        return processed
    
    def get_raw_sample(self, idx: int) -> Dict:
        """Get the raw, unprocessed sample."""
        return self.data[idx]




class DataCollator:
    """
    Custom collator for batching samples.
    Handles dynamic padding and ensures all tensors are properly batched.
    """
    
    def __init__(self, pad_token_id: int = 0):
        """
        Args:
            pad_token_id: ID to use for padding
        """
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.
        
        Args:
            batch: List of sample dictionaries
            
        Returns:
            Batched dictionary with stacked tensors
        """
        # Stack all tensors
        batched = {}
        
        for key in batch[0].keys():
            tensors = [sample[key] for sample in batch]
            
            # Stack tensors (they should already be padded to max_length)
            batched[key] = torch.stack(tensors, dim=0)
        
        return batched


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    pad_token_id: int = 0,
    shuffle_train: bool = True
) -> Union[DataLoader, tuple]:
    """
    Create DataLoader(s) for training and validation.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Optional validation dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        pad_token_id: Padding token ID
        shuffle_train: Whether to shuffle training data
        
    Returns:
        DataLoader or tuple of (train_loader, val_loader)
    """
    collator = DataCollator(pad_token_id=pad_token_id)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True
        )
        return train_loader, val_loader
    
    return train_loader


def load_datasets_from_config(
    esnli_data: Optional[Union[HFDataset, DatasetDict]] = None,
    preprocessor_config: Optional[PreprocessConfig] = None,
    cache_dir: Optional[str] = None
) -> Dict[str, Dataset]:
    """
    Load and configure datasets based on provided data.
    
    Args:
        esnli_data: e-SNLI data (single split or DatasetDict)
        preprocessor_config: Configuration for preprocessor
        cache_dir: Cache directory
        
    Returns:
        Dictionary with dataset splits
    """
    # Initialize preprocessor
    preprocessor = ReasoningPreprocessor(preprocessor_config)
    
    datasets = {}
    
    # Handle e-SNLI
    if esnli_data:
        if isinstance(esnli_data, DatasetDict):
            # Multiple splits
            for split_name, split_data in esnli_data.items():
                datasets[f"esnli_{split_name}"] = ESNLIDataset(
                    split_data,
                    preprocessor,
                    cache_dir=cache_dir
                )
        else:
            # Single split
            datasets["esnli"] = ESNLIDataset(
                esnli_data,
                preprocessor,
                cache_dir=cache_dir
            )
    
    return datasets


# Quick utility functions
def quick_create_esnli_dataloader(
    data: HFDataset,
    batch_size: int = 16,
    model_name: str = "google/flan-t5-base"
) -> DataLoader:
    """Quick helper to create e-SNLI dataloader."""
    config = PreprocessConfig(model_name=model_name)
    preprocessor = ReasoningPreprocessor(config)
    dataset = ESNLIDataset(data, preprocessor)
    
    return create_dataloaders(
        dataset,
        batch_size=batch_size,
        pad_token_id=preprocessor.tokenizer.pad_token_id
    )
