import json
import torch
import os
from typing import List, Dict, Callable, Optional
from tokenizers import Tokenizer
from .config_loader import load_config


class JSONDataLoader:
    def __init__(self, json_file: str, config=None, mask_token: int = None, tokenizer_encode_fn=None):
        """
        Initialize the JSON data loader.
        
        Args:
            json_file: Path to JSON file with format [{"text": "example 1"}, {"text": "example 2"}, ...]
            config: Configuration object. If provided, overrides individual parameters.
            mask_token: Token ID to use as mask/separator (default from config or 50256)
            tokenizer_encode_fn: Function to encode text to token IDs (default from config)
        """
        if config is None:
            config = load_config()
        
        self.config = config
        self.mask_token = mask_token if mask_token is not None else config.get('dataset.mask_token', 50256)
        self.max_length = config.get('data_loader.max_length', 64)
        
        # Load tokenizer if available and enabled
        self.tokenizer = None
        self.use_tokenizer = config.get('data_loader.use_tokenizer', True)
        if self.use_tokenizer:
            self.tokenizer = self._load_tokenizer()
            
        # Use provided tokenizer function or the loaded tokenizer
        if tokenizer_encode_fn is not None:
            self.tokenizer_encode_fn = tokenizer_encode_fn
        elif self.tokenizer is not None:
            self.tokenizer_encode_fn = self._tokenizer_encode
        else:
            self.tokenizer_encode_fn = config.get('data_loader.tokenizer_encode_fn')
            
        self.examples = self._load_data(json_file)
    
    def _load_data(self, json_file: str) -> List[Dict]:
        """Load data from JSON file."""
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of examples")
        
        for item in data:
            if not isinstance(item, dict) or "text" not in item:
                raise ValueError("Each example must be a dict with 'text' key")
        
        return data
    
    def _load_tokenizer(self) -> Optional[Tokenizer]:
        """Load tokenizer from config path."""
        try:
            # Use new path structure
            tokenizer_dir = self.config.get('paths.tokenizer_dir', 'data/tokenizers')
            tokenizer_filename = self.config.get('paths.tokenizer_file', 'tokenizer.json')
            tokenizer_path = os.path.join(tokenizer_dir, tokenizer_filename)
            
            # Fallback to old path for backward compatibility
            if not os.path.exists(tokenizer_path):
                tokenizer_path = self.config.get('tokenizer.save_path', 'data/tokenizer.json')
            if os.path.exists(tokenizer_path):
                tokenizer = Tokenizer.from_file(tokenizer_path)
                print(f"Loaded tokenizer from {tokenizer_path}")
                return tokenizer
            else:
                print(f"Tokenizer not found at {tokenizer_path}. Run create_tokenizer.py first.")
                return None
        except Exception as e:
            print(f"Failed to load tokenizer: {e}")
            return None
    
    def _tokenizer_encode(self, text: str) -> List[int]:
        """Encode text using the loaded tokenizer."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        
        # Add BOS and EOS tokens
        special_tokens = self.config.get('tokenizer.special_tokens', {})
        bos_token = special_tokens.get('bos_token', '[BOS]')
        eos_token = special_tokens.get('eos_token', '[EOS]')
        
        # Format text with special tokens
        formatted_text = f"{bos_token} {text} {eos_token}"
        
        # Tokenize
        encoded = self.tokenizer.encode(formatted_text)
        tokens = encoded.ids
        
        # Truncate or pad to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        elif len(tokens) < self.max_length:
            pad_token_id = self.config.get('tokenizer.token_ids.pad_token_id', 3)
            tokens.extend([pad_token_id] * (self.max_length - len(tokens)))
        
        return tokens
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size from tokenizer."""
        if self.tokenizer is not None:
            return self.tokenizer.get_vocab_size()
        else:
            # Fallback to config
            return self.config.get('training.model.vocab_size', 20)
    
    def get_masked_example(self, idx: int) -> torch.Tensor:
        """
        Get an example with tokens before split_token masked out for training.
        For arithmetic: "1 + 2 = 3" -> mask tokens before and including "=" (only train on "3")
        For other tasks: "question: answer" -> mask tokens before and including ":" (only train on "answer")
        
        Args:
            idx: Index of example to retrieve
            
        Returns:
            torch.Tensor: Token sequence with tokens before split_token set to -100
        """
        if idx >= len(self.examples):
            raise IndexError(f"Index {idx} out of range for {len(self.examples)} examples")
        
        text = self.examples[idx]["text"]
        
        # Tokenize using available method
        if self.tokenizer_encode_fn is not None:
            tokens = self.tokenizer_encode_fn(text)
        else:
            # Fallback to space-separated parsing
            if isinstance(text, str):
                try:
                    tokens = [int(x) for x in text.split()]
                except ValueError:
                    raise ValueError("Text must be space-separated integers if no tokenizer provided")
            else:
                tokens = list(text)
        
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        # Find the configurable split token and mask everything before it
        split_token = self.config.get('data_loader.split_token', '=')
        
        if self.tokenizer is not None:
            # Get split token ID from tokenizer
            split_encoding = self.tokenizer.encode(split_token)
            if len(split_encoding.ids) > 0:
                split_token_id = split_encoding.ids[0]
                mask_positions = (tokens == split_token_id).nonzero(as_tuple=True)[0]
            else:
                mask_positions = []
        else:
            # Fallback: look for the mask_token directly
            mask_positions = (tokens == self.mask_token).nonzero(as_tuple=True)[0]
        
        if len(mask_positions) > 0:
            # Mask all tokens before and including the first occurrence of split_token
            # This way we only train on predicting the target tokens
            mask_pos = mask_positions[0].item()
            masked_tokens = tokens.clone()
            masked_tokens[:mask_pos + 1] = -100  # -100 is standard ignore index
            return masked_tokens
        else:
            # No split token found, return original tokens
            return tokens
    
    def get_example(self, idx: int) -> torch.Tensor:
        """
        Get an example without masking.
        
        Args:
            idx: Index of example to retrieve
            
        Returns:
            torch.Tensor: Original token sequence
        """
        if idx >= len(self.examples):
            raise IndexError(f"Index {idx} out of range for {len(self.examples)} examples")
        
        text = self.examples[idx]["text"]
        
        # Tokenize using available method
        if self.tokenizer_encode_fn is not None:
            tokens = self.tokenizer_encode_fn(text)
        else:
            # Fallback to space-separated parsing
            if isinstance(text, str):
                try:
                    tokens = [int(x) for x in text.split()]
                except ValueError:
                    raise ValueError("Text must be space-separated integers if no tokenizer provided")
            else:
                tokens = list(text)
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get raw example dict."""
        return self.examples[idx]


def load_json_data(json_file: str, config=None, mask_token: int = None, tokenizer_encode_fn=None) -> JSONDataLoader:
    """
    Convenience function to create a JSONDataLoader.
    
    Args:
        json_file: Path to JSON file
        config: Configuration object
        mask_token: Token to use for masking (default from config)
        tokenizer_encode_fn: Optional tokenizer function (default from config)
        
    Returns:
        JSONDataLoader instance
    """
    return JSONDataLoader(json_file, config, mask_token, tokenizer_encode_fn)