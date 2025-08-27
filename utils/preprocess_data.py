#!/usr/bin/env python3
"""
Data preprocessing pipeline for arithmetic training.
Converts JSON arithmetic data to format suitable for training.
"""

import os
import json
import torch
from pathlib import Path
from tokenizers import Tokenizer
from config_loader import load_config
from data import JSONDataLoader


def create_arithmetic_dataset(data_loader: JSONDataLoader, output_path: str):
    """
    Create preprocessed dataset from JSON data loader.
    
    Args:
        data_loader: JSONDataLoader instance
        output_path: Path to save preprocessed data
    """
    print(f"Processing {len(data_loader)} examples...")
    
    all_inputs = []
    all_targets = []
    
    for idx in range(len(data_loader)):
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{len(data_loader)} examples")
        
        try:
            # Get tokenized example
            tokens = data_loader.get_example(idx)
            
            # Create input/target pairs for language modeling
            # Input: all tokens except the last one
            # Target: all tokens except the first one (shifted by 1)
            if len(tokens) > 1:
                inputs = tokens[:-1]
                targets = tokens[1:]
                
                all_inputs.append(inputs)
                all_targets.append(targets)
        
        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            continue
    
    # Convert to tensors
    if all_inputs:
        # Pad sequences to same length
        max_len = max(len(seq) for seq in all_inputs)
        pad_token_id = data_loader.config.get('tokenizer.token_ids.pad_token_id', 3)
        
        padded_inputs = []
        padded_targets = []
        
        for inputs, targets in zip(all_inputs, all_targets):
            # Pad inputs
            padded_input = torch.cat([
                inputs,
                torch.full((max_len - len(inputs),), pad_token_id, dtype=torch.long)
            ])
            # Pad targets (use -100 for padding in targets to ignore in loss)
            padded_target = torch.cat([
                targets,
                torch.full((max_len - len(targets),), -100, dtype=torch.long)
            ])
            
            padded_inputs.append(padded_input)
            padded_targets.append(padded_target)
        
        # Stack into tensors
        input_tensor = torch.stack(padded_inputs)
        target_tensor = torch.stack(padded_targets)
        
        # Save preprocessed data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save({
            'inputs': input_tensor,
            'targets': target_tensor,
            'vocab_size': data_loader.get_vocab_size(),
            'num_examples': len(all_inputs),
            'max_length': max_len
        }, output_path)
        
        print(f"Saved {len(all_inputs)} examples to {output_path}")
        print(f"Input shape: {input_tensor.shape}")
        print(f"Target shape: {target_tensor.shape}")
        print(f"Vocabulary size: {data_loader.get_vocab_size()}")
        
        return input_tensor, target_tensor
    else:
        raise RuntimeError("No valid examples found")


def create_simple_dataloader(preprocessed_path: str, batch_size: int = 32, shuffle: bool = True):
    """
    Create a simple PyTorch DataLoader from preprocessed data.
    
    Args:
        preprocessed_path: Path to preprocessed data file
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    from torch.utils.data import Dataset, DataLoader
    
    class ArithmeticDataset(Dataset):
        def __init__(self, preprocessed_path):
            data = torch.load(preprocessed_path)
            self.inputs = data['inputs']
            self.targets = data['targets']
            self.vocab_size = data['vocab_size']
            self.max_length = data['max_length']
        
        def __len__(self):
            return len(self.inputs)
        
        def __getitem__(self, idx):
            return {
                'input_ids': self.inputs[idx],
                'labels': self.targets[idx]
            }
    
    dataset = ArithmeticDataset(preprocessed_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    print(f"Created DataLoader with {len(dataset)} examples")
    print(f"Vocabulary size: {dataset.vocab_size}")
    print(f"Max length: {dataset.max_length}")
    
    return dataloader, dataset.vocab_size


def update_config_vocab_size(config, vocab_size: int):
    """
    Update configuration with actual vocabulary size from tokenizer.
    
    Args:
        config: Configuration object
        vocab_size: Actual vocabulary size
    """
    config.set('training.model.vocab_size', vocab_size)
    config.save()
    print(f"Updated config vocab_size to {vocab_size}")


def main():
    """Main preprocessing function."""
    print("Starting data preprocessing pipeline...")
    
    # Load configuration
    config = load_config()
    
    # Get file paths
    data_dir = config.get('paths.data_dir', 'data')
    train_file = os.path.join(data_dir, config.get('dataset.train_file', 'train.json'))
    test_file = os.path.join(data_dir, config.get('dataset.test_file', 'test.json'))
    
    # Check if files exist
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    # Check if tokenizer exists
    tokenizer_path = config.get('tokenizer.save_path', 'data/tokenizer.json')
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer not found at {tokenizer_path}")
        print("Creating tokenizer first...")
        
        # Import and run tokenizer creation
        from create_tokenizer import main as create_tokenizer_main
        create_tokenizer_main()
    
    # Create data loaders
    print("Loading training data...")
    train_loader = JSONDataLoader(train_file, config)
    
    print("Loading test data...")
    test_loader = JSONDataLoader(test_file, config)
    
    # Get vocabulary size and update config
    vocab_size = train_loader.get_vocab_size()
    update_config_vocab_size(config, vocab_size)
    
    # Preprocess datasets
    print("Preprocessing training data...")
    train_output = os.path.join(data_dir, 'train_preprocessed.pt')
    create_arithmetic_dataset(train_loader, train_output)
    
    print("Preprocessing test data...")
    test_output = os.path.join(data_dir, 'test_preprocessed.pt')
    create_arithmetic_dataset(test_loader, test_output)
    
    # Test DataLoader creation
    print("Testing DataLoader creation...")
    batch_size = config.get('data_loader.batch_size', 32)
    train_dataloader, vocab_size = create_simple_dataloader(train_output, batch_size, shuffle=True)
    test_dataloader, _ = create_simple_dataloader(test_output, batch_size, shuffle=False)
    
    # Show sample batch
    print("\nSample from training data:")
    sample_batch = next(iter(train_dataloader))
    print(f"Input batch shape: {sample_batch['input_ids'].shape}")
    print(f"Target batch shape: {sample_batch['labels'].shape}")
    print(f"Sample input: {sample_batch['input_ids'][0][:20]}...")  # Show first 20 tokens
    print(f"Sample target: {sample_batch['labels'][0][:20]}...")    # Show first 20 tokens
    
    print("\nData preprocessing completed successfully!")
    print(f"Training data: {train_output}")
    print(f"Test data: {test_output}")
    print(f"Final vocabulary size: {vocab_size}")


if __name__ == "__main__":
    main()