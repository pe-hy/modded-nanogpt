#!/usr/bin/env python3
"""
Inference script for modded-nanogpt adapted for arithmetic tasks.
Loads config-based model architecture and arithmetic tokenizer.

Usage:
python inference.py checkpoints/model.pt --prompt "[BOS] 5 + 3 =" --max-length 10
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tokenizers import Tokenizer
from utils.config_loader import load_config
import json

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

def norm(x):
    """Layer normalization without learnable parameters (matches train_gpt.py)"""
    return F.layer_norm(x, x.shape[-1:], eps=1e-6)

class CastedLinear(nn.Linear):
    """Simplified CastedLinear for inference (no FP8, no special training flags)"""
    def __init__(self, in_features, out_features, bias=True, dtype=None):
        super().__init__(in_features, out_features, bias=bias, dtype=dtype)

class Rotary(nn.Module):
    """Rotary positional embeddings - matches train_gpt.py exactly"""
    def __init__(self, dim: int, max_seq_len: int, config=None):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        base_freq = config.get('training.model.rope_base_freq', 1024) if config else 1024
        angular_freq = (1 / base_freq) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)
        
    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    """Simplified attention for inference - matches train_gpt.py structure"""
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, config=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = config.get('training.model.head_dim', 128) if config else 128
        hdim = num_heads * self.head_dim
        
        # Merged QKV weights (matches train_gpt.py exactly)
        self.qkv_w = nn.Parameter(torch.randn(3, hdim, dim))  # Will be loaded from checkpoint
        self.rotary = Rotary(self.head_dim, max_seq_len, config)
        self.c_proj = CastedLinear(hdim, dim, bias=False)
        
        # Attention scale (matches train_gpt.py)
        self.attn_scale = config.get('training.model.attn_scale', 0.12) if config else 0.12

    def forward(self, x: Tensor, ve: Tensor | None, lambdas: Tensor):
        B, T = x.size(0), x.size(1)
        
        # Apply merged QKV transformation (matches train_gpt.py line 346)
        qkv = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x))
        q, k, v = qkv.view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        
        # QK norm (matches train_gpt.py line 347)
        q, k = norm(q), norm(k)
        
        # Apply rotary embeddings (matches train_gpt.py dimension expectations)
        q, k = self.rotary(q), self.rotary(k)
        
        # Value embedding mixing (matches train_gpt.py lines 349-352)
        if ve is not None:
            v = lambdas[0] * v + lambdas[1] * ve.view_as(v)
        else:
            v = lambdas[0] * v
        
        # Simplified attention (no FlexAttention for inference)
        q = q.transpose(1, 2)  # (B, num_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention computation with custom scale
        att = (q @ k.transpose(-2, -1)) * self.attn_scale
        
        # Causal mask - only attend to previous and current positions
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~causal_mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        # Debug: check if attention pattern looks reasonable (disabled for clean output)
        # if T <= 10:  # Only for short sequences
        #     print(f"Attention shape: {att.shape}, mean attention: {att.mean():.4f}")
        
        y = att @ v  # (B, num_heads, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        
        return self.c_proj(y)

class MLP(nn.Module):
    """MLP block - matches train_gpt.py exactly"""
    def __init__(self, dim: int, config=None):
        super().__init__()
        mlp_ratio = config.get('training.model.mlp_ratio', 4) if config else 4
        hdim = mlp_ratio * dim
        self.c_fc = CastedLinear(dim, hdim, bias=False)
        self.c_proj = CastedLinear(hdim, dim, bias=False)

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = F.relu(x).square()  # ReLU^2 activation (matches train_gpt.py line 368)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    """Transformer block - matches train_gpt.py structure"""
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int, num_layers: int, config=None):
        super().__init__()
        # Skip attention of certain layers (adapt for different num_layers)
        skip_layer = num_layers // 2 + 1 if num_layers > 7 else None
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len, config) if layer_idx != skip_layer else None
        self.mlp = MLP(dim, config)

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, lambdas: Tensor, sa_lambdas: Tensor):
        # Skip connection mixing (matches train_gpt.py line 380)
        x = lambdas[0] * x + lambdas[1] * x0
        
        if self.attn is not None:
            x = x + self.attn(norm(x), ve, sa_lambdas)
        x = x + self.mlp(norm(x))
        return x

class GPT(nn.Module):
    """Main GPT model - matches train_gpt.py architecture exactly"""
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int, config=None):
        super().__init__()
        self.config = config  # Store config for use in forward method
        vocab_multiple = config.get('training.model.vocab_multiple', 128) if config else 128
        vocab_size = next_multiple_of_n(vocab_size, n=vocab_multiple)
        
        # Embeddings (matches train_gpt.py lines 396-399)
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        
        # Transformer blocks
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, max_seq_len, i, num_layers, config) for i in range(num_layers)])
        
        # Language modeling head (simplified, no FP8)
        self.lm_head = CastedLinear(model_dim, vocab_size, bias=False)
        
        # Scalars parameter (matches checkpoint size)
        # For 6-layer model, the checkpoint has 32 scalars
        if config:
            if num_layers == 6:
                scalar_size = config.get('training.model.scalars_size_6_layers', 32)
            else:
                base = config.get('training.model.scalars_size_formula_base', 64)
                mult = config.get('training.model.scalars_size_formula_multiplier', 5)
                offset = config.get('training.model.scalars_size_formula_offset', 16)
                scalar_size = max(base, mult * num_layers + offset)
        else:
            scalar_size = 32 if num_layers == 6 else max(64, 5 * num_layers + 16)
        self.scalars = nn.Parameter(torch.ones(scalar_size))
        
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

    def forward(self, input_seq: Tensor, return_logits=True):
        assert input_seq.ndim == 2  # (batch_size, seq_len)
        B, T = input_seq.shape
        
        # Token value embeddings (adapt pattern for different num_layers)
        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        if self.num_layers >= 6:
            # Original pattern for 6+ layers: 012...012
            ve_pattern = [ve[0], ve[1], ve[2]] + [None] * (self.num_layers - 6) + [ve[0], ve[1], ve[2]]
        else:
            # For fewer layers, distribute the available embeddings
            ve_pattern = []
            for i in range(self.num_layers):
                if i < len(ve):
                    ve_pattern.append(ve[i])
                else:
                    ve_pattern.append(None)
        assert len(ve_pattern) == len(self.blocks)

        # Initial embedding (matches train_gpt.py line 474)
        x = x0 = norm(self.embed(input_seq))

        # Extract scalars (matches train_gpt.py lines 478-480)
        skip_weights = self.scalars[:(len(self.blocks) // 2)]
        lambdas = self.scalars[1 * len(self.blocks): 3 * len(self.blocks)].view(-1, 2)
        sa_lambdas = self.scalars[3 * len(self.blocks): 5 * len(self.blocks)].view(-1, 2)

        # U-net design (matches train_gpt.py lines 482-490)
        skip_connections = []
        n = len(self.blocks) // 2

        for i in range(len(self.blocks)):
            if i >= n:
                x = x + skip_weights[i - n] * skip_connections.pop()
            x = self.blocks[i](x, ve_pattern[i], x0, lambdas[i], sa_lambdas[i])
            if i < n:
                skip_connections.append(x)

        # Final processing (matches train_gpt.py lines 491-494)
        x = norm(x)
        logits = self.lm_head(x).float()
        
        # Tanh softcapping (matches train_gpt.py line 494) - use config values
        softcap_scale = self.config.get('training.model.tanh_softcap_scale', 30.0) if self.config else 30.0
        softcap_divisor = self.config.get('training.model.tanh_softcap_divisor', 7.5) if self.config else 7.5
        logits = softcap_scale * torch.tanh(logits / (softcap_divisor * x.size(-1)**0.5))
        
        return logits

def load_arithmetic_tokenizer(config):
    """Load the arithmetic tokenizer"""
    # Use new path structure
    tokenizer_dir = config.get('paths.tokenizer_dir', 'data/tokenizers')
    tokenizer_filename = config.get('paths.tokenizer_file', 'tokenizer.json')
    vocab_filename = config.get('paths.vocab_info_file', 'tokenizer_vocab.json')
    
    tokenizer_path = os.path.join(tokenizer_dir, tokenizer_filename)
    
    # Fallback to old path for backward compatibility
    if not os.path.exists(tokenizer_path):
        tokenizer_path = config.get('tokenizer.save_path', 'data/tokenizer.json')
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Run create_tokenizer.py first.")
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # Load vocabulary info using new or old path structure  
    if tokenizer_path.startswith(tokenizer_dir):
        vocab_path = os.path.join(tokenizer_dir, vocab_filename)
    else:
        vocab_path = tokenizer_path.replace('.json', '_vocab.json')
    vocab_info = {}
    if os.path.exists(vocab_path):
        with open(vocab_path, 'r') as f:
            vocab_info = json.load(f)
    
    return tokenizer, vocab_info

def load_checkpoint_arithmetic(checkpoint_path: str, config, device='cuda'):
    """Load model with arithmetic-specific architecture"""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"Checkpoint loaded - Step: {checkpoint.get('step', 'Unknown')}")
    
    # Get parameters from config
    tokenizer, vocab_info = load_arithmetic_tokenizer(config)
    vocab_size = vocab_info.get('vocab_size', config.get('training.model.vocab_size', 20))
    
    model = GPT(
        vocab_size=vocab_size,
        num_layers=config.get('training.model.num_layers', 6),
        num_heads=config.get('training.model.num_heads', 4),
        model_dim=config.get('training.model.model_dim', 512),
        max_seq_len=config.get('training.model.max_seq_len', 64),
        config=config
    )
    
    print(f"Model architecture: vocab_size={vocab_size}, num_layers={model.num_layers}, "
          f"num_heads={model.num_heads}, model_dim={model.model_dim}")
    
    # Load state dict with proper key mapping
    state_dict = checkpoint['model']
    new_state_dict = {}
    
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key[10:]  # Remove _orig_mod. prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # Load with strict=False to handle missing/extra keys
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    
    if missing_keys:
        print(f"Missing keys: {len(missing_keys)}")
        for key in missing_keys[:5]:
            print(f"  - {key}")
        if len(missing_keys) > 5:
            print(f"  ... and {len(missing_keys) - 5} more")
    
    if unexpected_keys:
        print(f"Unexpected keys: {len(unexpected_keys)}")
    
    # Move to device
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to CUDA")
    else:
        device = 'cpu'
        print("Model on CPU")
    
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model ready - {total_params:,} parameters on {device}")
    
    return model, tokenizer, vocab_info, device

def generate_arithmetic(model, tokenizer, prompt: str, max_length: int = None, device: str = 'cuda', config=None):
    """Generate completion deterministically using configurable split token"""
    model.eval()
    
    # Use config defaults if not provided
    if max_length is None:
        max_length = config.get('inference.max_length', 20) if config else 20
    debug_top_k = config.get('inference.debug_top_k', 3) if config else 3
    debug_steps = config.get('inference.debug_steps', 5) if config else 5
    max_seq_len_check = config.get('training.evaluation.max_seq_len_check', 64) if config else 64
    
    # Get configurable split token
    split_token = config.get('data_loader.split_token', '=') if config else '='
    
    # Encode prompt
    encoded = tokenizer.encode(prompt)
    tokens = torch.tensor(encoded.ids, dtype=torch.long, device=device).unsqueeze(0)
    
    print(f"Input tokens: {encoded.tokens}")
    print(f"Input IDs: {encoded.ids}")
    
    with torch.no_grad():
        for i in range(max_length):
            # Forward pass
            logits = model(tokens)
            
            # Get next token logits
            next_logits = logits[0, -1, :]
            
            # Always use greedy decoding (deterministic)
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            
            # Show top predictions for debugging
            if i < debug_steps:
                top_k = debug_top_k
                top_logits, top_indices = torch.topk(next_logits, top_k)
                top_tokens = [tokenizer.decode([idx.item()]) for idx in top_indices]
                print(f"Step {i+1} top predictions: {list(zip(top_tokens, top_logits.tolist()))}")
                print(f"Chosen: {tokenizer.decode([next_token.item()])}")
            
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            
            # Decode to check if we should stop
            current_tokens = tokens[0].cpu().tolist()
            decoded = tokenizer.decode(current_tokens)
            
            # Stop at EOS token or when sequence gets too long
            if len(current_tokens) > 0 and (decoded.endswith('[EOS]') or len(current_tokens) >= max_seq_len_check):
                print("Stopping: EOS detected or max length reached")
                break
    
    # Decode generated sequence
    generated_tokens = tokens[0].cpu().tolist()
    generated_text = tokenizer.decode(generated_tokens)
    
    # Extract answer (everything after split token)
    answer = extract_answer(generated_text, split_token)
    
    return generated_text, generated_tokens, answer

def extract_answer(text: str, split_token: str = '=') -> str:
    """Extract answer portion from generated text (everything after split token)"""
    if split_token in text:
        parts = text.split(split_token, 1)  # Split only on first occurrence
        if len(parts) > 1:
            answer = parts[1].strip()
            # Remove [EOS] if present
            if answer.endswith('[EOS]'):
                answer = answer[:-5].strip()
            return answer
    
    # If no split token found, return everything after [BOS] if present
    if '[BOS]' in text:
        return text.split('[BOS]', 1)[1].strip()
    
    return text.strip()

def main():
    parser = argparse.ArgumentParser(description='Arithmetic inference for modded-nanogpt')
    parser.add_argument('checkpoint', help='Path to checkpoint file')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--prompt', default=None, help='Arithmetic prompt')
    parser.add_argument('--max-length', type=int, default=20, help='Max generation length')
    
    args = parser.parse_args()
    
    try:
        # Load config
        config = load_config()
        
        # Load model and tokenizer
        model, tokenizer, vocab_info, device = load_checkpoint_arithmetic(
            args.checkpoint, config, args.device
        )
        
        print(f"Vocabulary size: {vocab_info.get('vocab_size', 'Unknown')}")
        print(f"Special tokens: {vocab_info.get('special_tokens', {})}")
        
        # Generate arithmetic completion
        # Use default prompt from config if not provided
        prompt = args.prompt if args.prompt is not None else config.get('inference.default_prompt', '[BOS] 5 + 3 =')
        
        print(f"\nGenerating from prompt: '{prompt}'")
        print("-" * 50)
        
        generated_text, generated_tokens, answer = generate_arithmetic(
            model, tokenizer, prompt,
            max_length=args.max_length,
            device=device,
            config=config
        )
        
        print(f"Generated: {generated_text}")
        print(f"Token IDs: {generated_tokens}")
        print(f"Extracted Answer: '{answer}'")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
