#!/usr/bin/env python3

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast
import torch
import json
import glob
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config_loader import load_config


def get_vocab(config):
    """
    Extract vocabulary from arithmetic JSON data files.
    
    Args:
        config: Configuration object
        
    Returns:
        set: Set of unique tokens found in the data
    """
    # Get data files from config using new path structure
    dataset_dir = config.get('paths.dataset_dir', 'data/datasets')
    train_file = config.get('paths.train_file', 'train.json')
    test_file = config.get('paths.test_file', 'test.json')
    
    # Build full paths
    train_path = os.path.join(dataset_dir, train_file)
    test_path = os.path.join(dataset_dir, test_file)
    
    data_files = []
    if os.path.exists(train_path):
        data_files.append(train_path)
    if os.path.exists(test_path):
        data_files.append(test_path)
    
    # Also check for any additional JSON files in dataset directory
    additional_files = glob.glob(f"{dataset_dir}/*.json")
    data_files.extend([f for f in additional_files if f not in data_files])
    
    print(f"Found data files: {data_files}")
    
    if not data_files:
        raise FileNotFoundError(f"No JSON data files found in {dataset_dir}")
        
    all_data = []
    for file_path in data_files:
        with open(file_path, "r") as f:
            file_data = json.load(f)
            all_data.extend(file_data)

    # Extract text from all examples
    data = [item["text"] for item in all_data]
    print(f"Total examples: {len(data)}")
    
    # Join all text and split on whitespace to get vocabulary
    combined_text = " ".join(data)
    vocab = set(combined_text.split())
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Vocabulary tokens: {sorted(vocab)}")
    
    return vocab


def get_tokenizer(vocab, config):
    """
    Create and save tokenizer with the given vocabulary.
    
    Args:
        vocab: Set of vocabulary tokens
        config: Configuration object
        
    Returns:
        Tokenizer: The created tokenizer
    """
    # Get special tokens from config
    special_tokens = config.get('tokenizer.special_tokens', {
        'bos_token': '[BOS]',
        'eos_token': '[EOS]',
        'unk_token': '[UNK]',
        'pad_token': '[PAD]',
        'mask_token': '[MASK]'
    })
    
    # Create vocabulary mapping with special tokens first
    vocab_list = []
    vocab_list.extend(special_tokens.values())  # Add special tokens first
    vocab_list.extend(sorted(vocab))  # Add data vocabulary
    
    # Remove duplicates while preserving order
    seen = set()
    unique_vocab = []
    for token in vocab_list:
        if token not in seen:
            seen.add(token)
            unique_vocab.append(token)
    
    # Create vocabulary mapping
    vocab_dict = {token: i for i, token in enumerate(unique_vocab)}
    
    print(f"Final vocabulary size: {len(vocab_dict)}")
    print(f"Special token mappings:")
    for name, token in special_tokens.items():
        print(f"  {name}: {token} -> {vocab_dict.get(token, 'NOT FOUND')}")
    
    # Initialize tokenizer with complete vocabulary
    tokenizer = Tokenizer(WordLevel(vocab_dict, unk_token=special_tokens['unk_token']))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    
    # Add special tokens
    tokenizer.add_special_tokens(list(special_tokens.values()))
    
    # Save tokenizer using new path structure
    tokenizer_dir = config.get('paths.tokenizer_dir', 'data/tokenizers')
    tokenizer_filename = config.get('paths.tokenizer_file', 'tokenizer.json')
    vocab_filename = config.get('paths.vocab_info_file', 'tokenizer_vocab.json')
    
    # Create full paths
    os.makedirs(tokenizer_dir, exist_ok=True)
    save_path = os.path.join(tokenizer_dir, tokenizer_filename)
    vocab_save_path = os.path.join(tokenizer_dir, vocab_filename)
    
    tokenizer.save(save_path)
    print(f"Tokenizer saved to: {save_path}")
    
    # Also save vocabulary mapping for reference
    with open(vocab_save_path, 'w') as f:
        json.dump({
            'vocab': vocab_dict,
            'special_tokens': special_tokens,
            'vocab_size': len(vocab_dict)
        }, f, indent=2)
    print(f"Vocabulary mapping saved to: {vocab_save_path}")

    return tokenizer


def create_hf_tokenizer(tokenizer_path, config):
    """
    Create HuggingFace-compatible tokenizer wrapper.
    
    Args:
        tokenizer_path: Path to the saved tokenizer
        config: Configuration object
        
    Returns:
        PreTrainedTokenizerFast: HuggingFace tokenizer wrapper
    """
    # Load the tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # Get special tokens
    special_tokens = config.get('tokenizer.special_tokens', {
        'bos_token': '[BOS]',
        'eos_token': '[EOS]',
        'unk_token': '[UNK]',
        'pad_token': '[PAD]',
        'mask_token': '[MASK]'
    })
    
    # Create HuggingFace wrapper
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        **special_tokens
    )
    
    # Save HuggingFace tokenizer using new path structure  
    tokenizer_dir = config.get('paths.tokenizer_dir', 'data/tokenizers')
    hf_save_path = os.path.join(tokenizer_dir, 'hf_tokenizer')
    os.makedirs(hf_save_path, exist_ok=True)
    hf_tokenizer.save_pretrained(hf_save_path)
    print(f"HuggingFace tokenizer saved to: {hf_save_path}")
    
    return hf_tokenizer


def test_tokenizer(tokenizer_path, config):
    """
    Test the created tokenizer on sample arithmetic expressions.
    
    Args:
        tokenizer_path: Path to the saved tokenizer
        config: Configuration object
    """
    print("\n" + "="*50)
    print("TESTING TOKENIZER")
    print("="*50)
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # Test examples
    test_examples = [
        "1 + 2 = 3",
        "1 0 + 5 = 5 1",
        "9 9 + 1 = 0 0 1",
        "5 + 5 = 0 1"
    ]
    
    for example in test_examples:
        encoded = tokenizer.encode(example)
        decoded = tokenizer.decode(encoded.ids)
        print(f"Original: '{example}'")
        print(f"Tokens:   {encoded.tokens}")
        print(f"IDs:      {encoded.ids}")
        print(f"Decoded:  '{decoded}'")
        print("-" * 30)


def main():
    """Main function to create tokenizer."""
    print("Creating arithmetic tokenizer...")
    
    # Load configuration
    config = load_config()
    
    # Extract vocabulary from data
    vocab = get_vocab(config)
    
    # Create and save tokenizer
    tokenizer = get_tokenizer(vocab, config)
    
    # Create HuggingFace-compatible version using new path structure
    tokenizer_dir = config.get('paths.tokenizer_dir', 'data/tokenizers')
    tokenizer_filename = config.get('paths.tokenizer_file', 'tokenizer.json')
    tokenizer_path = os.path.join(tokenizer_dir, tokenizer_filename)
    hf_tokenizer = create_hf_tokenizer(tokenizer_path, config)
    
    # Test the tokenizer
    test_tokenizer(tokenizer_path, config)
    
    print("\nTokenizer creation completed successfully!")


if __name__ == "__main__":
    main()