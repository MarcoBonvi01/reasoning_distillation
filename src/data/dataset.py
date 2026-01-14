"""
Dataset Module for Reasoning Distillation Project

PyTorch Dataset classes for e-SNLI and Alpaca datasets,
with support for caching, multi-task training, and data augmentation.
"""

import logging
import random
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
        if self.use_cache and idx in self.cache:
            return self.cache[idx]
        
        # Get raw sample
        sample = self.data[idx]
        
        # Preprocess
        processed = self.preprocessor.preprocess_esnli_sample(sample)
        
        # Cache if enabled
        if self.use_cache:
            self.cache[idx] = processed
        
        return processed
    
    def get_raw_sample(self, idx: int) -> Dict:
        """Get the raw, unprocessed sample."""
        return self.data[idx]


class AlpacaDataset(Dataset):
    """
    PyTorch Dataset for Alpaca-style instruction-following data.
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
        
        logger.info(f"Initialized AlpacaDataset with {len(data)} samples")
    
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
        if self.use_cache and idx in self.cache:
            return self.cache[idx]
        
        # Get raw sample
        sample = self.data[idx]
        
        # Preprocess
        processed = self.preprocessor.preprocess_alpaca_sample(sample)
        
        # Cache if enabled
        if self.use_cache:
            self.cache[idx] = processed
        
        return processed
    
    def get_raw_sample(self, idx: int) -> Dict:
        """Get the raw, unprocessed sample."""
        return self.data[idx]


class MultiTaskDataset(Dataset):
    """
    Combined dataset for multi-task training.
    Alternates between e-SNLI and Alpaca samples.
    """
    
    def __init__(self,
                 esnli_dataset: Optional[ESNLIDataset] = None,
                 alpaca_dataset: Optional[AlpacaDataset] = None,
                 sampling_strategy: str = "balanced"):
        """
        Args:
            esnli_dataset: ESNLIDataset instance
            alpaca_dataset: AlpacaDataset instance
            sampling_strategy: How to sample from datasets
                - "balanced": Equal probability for each dataset
                - "proportional": Sample based on dataset sizes
                - "esnli_only": Only use e-SNLI
                - "alpaca_only": Only use Alpaca
        """
        self.esnli_dataset = esnli_dataset
        self.alpaca_dataset = alpaca_dataset
        self.sampling_strategy = sampling_strategy
        
        # Calculate total length and probabilities
        self.esnli_len = len(esnli_dataset) if esnli_dataset else 0
        self.alpaca_len = len(alpaca_dataset) if alpaca_dataset else 0
        self.total_len = self.esnli_len + self.alpaca_len
        
        if self.total_len == 0:
            raise ValueError("At least one dataset must be provided")
        
        # Set sampling probabilities
        if sampling_strategy == "balanced":
            self.esnli_prob = 0.5 if (esnli_dataset and alpaca_dataset) else (1.0 if esnli_dataset else 0.0)
        elif sampling_strategy == "proportional":
            self.esnli_prob = self.esnli_len / self.total_len if self.total_len > 0 else 0.0
        elif sampling_strategy == "esnli_only":
            self.esnli_prob = 1.0
        elif sampling_strategy == "alpaca_only":
            self.esnli_prob = 0.0
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
        
        logger.info(f"MultiTaskDataset: {self.esnli_len} e-SNLI, {self.alpaca_len} Alpaca")
        logger.info(f"Sampling strategy: {sampling_strategy}, e-SNLI prob: {self.esnli_prob:.2f}")
    
    def __len__(self) -> int:
        return self.total_len
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from one of the datasets based on sampling strategy.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        # Determine which dataset to sample from
        use_esnli = random.random() < self.esnli_prob
        
        if use_esnli and self.esnli_dataset:
            # Sample from e-SNLI
            dataset_idx = idx % self.esnli_len
            return self.esnli_dataset[dataset_idx]
        elif self.alpaca_dataset:
            # Sample from Alpaca
            dataset_idx = idx % self.alpaca_len
            return self.alpaca_dataset[dataset_idx]
        else:
            # Fallback
            if self.esnli_dataset:
                dataset_idx = idx % self.esnli_len
                return self.esnli_dataset[dataset_idx]
            else:
                dataset_idx = idx % self.alpaca_len
                return self.alpaca_dataset[dataset_idx]


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
    alpaca_data: Optional[HFDataset] = None,
    preprocessor_config: Optional[PreprocessConfig] = None,
    cache_dir: Optional[str] = None,
    use_multitask: bool = False,
    sampling_strategy: str = "balanced"
) -> Dict[str, Dataset]:
    """
    Load and configure datasets based on provided data.
    
    Args:
        esnli_data: e-SNLI data (single split or DatasetDict)
        alpaca_data: Alpaca data
        preprocessor_config: Configuration for preprocessor
        cache_dir: Cache directory
        use_multitask: Whether to create multi-task dataset
        sampling_strategy: Sampling strategy for multi-task
        
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
    
    # Handle Alpaca
    if alpaca_data:
        datasets["alpaca"] = AlpacaDataset(
            alpaca_data,
            preprocessor,
            cache_dir=cache_dir
        )
    
    # Create multi-task dataset if requested
    if use_multitask and "esnli_train" in datasets and "alpaca" in datasets:
        datasets["multitask_train"] = MultiTaskDataset(
            esnli_dataset=datasets["esnli_train"],
            alpaca_dataset=datasets["alpaca"],
            sampling_strategy=sampling_strategy
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


def quick_create_alpaca_dataloader(
    data: HFDataset,
    batch_size: int = 16,
    model_name: str = "google/flan-t5-base"
) -> DataLoader:
    """Quick helper to create Alpaca dataloader."""
    config = PreprocessConfig(model_name=model_name)
    preprocessor = ReasoningPreprocessor(config)
    dataset = AlpacaDataset(data, preprocessor)
    
    return create_dataloaders(
        dataset,
        batch_size=batch_size,
        pad_token_id=preprocessor.tokenizer.pad_token_id
    )