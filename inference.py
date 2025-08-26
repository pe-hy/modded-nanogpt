#!/usr/bin/env python3
"""
Proper inference loader for modded-nanogpt that matches the exact training architecture.
Simplified for inference without distributed training, FlexAttention, or FP8 components.

uv run inference_proper.py logs/3ab5c844-9660-44cb-bcf5-30d4e24240b6/state_step001750.pt --prompt "The weather today is" --max-length 10 --temperature 0.95
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import tiktoken

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
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
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
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        
        # Merged QKV weights (matches train_gpt.py exactly)
        self.qkv_w = nn.Parameter(torch.randn(3, hdim, dim))  # Will be loaded from checkpoint
        self.rotary = Rotary(head_dim, max_seq_len)
        self.c_proj = CastedLinear(hdim, dim, bias=False)
        
        # Attention scale (matches train_gpt.py)
        self.attn_scale = 0.12

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
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = CastedLinear(dim, hdim, bias=False)
        self.c_proj = CastedLinear(hdim, dim, bias=False)

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = F.relu(x).square()  # ReLU^2 activation (matches train_gpt.py line 368)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    """Transformer block - matches train_gpt.py structure"""
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        # Skip attention of blocks.7 (matches train_gpt.py line 376)
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = MLP(dim)

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, lambdas: Tensor, sa_lambdas: Tensor):
        # Skip connection mixing (matches train_gpt.py line 380)
        x = lambdas[0] * x + lambdas[1] * x0
        
        if self.attn is not None:
            x = x + self.attn(norm(x), ve, sa_lambdas)
        x = x + self.mlp(norm(x))
        return x

class GPT(nn.Module):
    """Main GPT model - matches train_gpt.py architecture exactly"""
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int):
        super().__init__()
        vocab_size = next_multiple_of_n(vocab_size, n=128)
        
        # Embeddings (matches train_gpt.py lines 396-399)
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        
        # Transformer blocks
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, max_seq_len, i) for i in range(num_layers)])
        
        # Language modeling head (simplified, no FP8)
        self.lm_head = CastedLinear(model_dim, vocab_size, bias=False)
        
        # Scalars parameter (matches train_gpt.py structure + padding)
        # Original has padding for distributed training: (-num_layers * 5) % world_size
        # For checkpoint compatibility, we'll use the exact size from checkpoint (64)
        self.scalars = nn.Parameter(torch.ones(64))  # Will be loaded from checkpoint
        
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

    def forward(self, input_seq: Tensor, return_logits=True):
        assert input_seq.ndim == 2  # (batch_size, seq_len)
        B, T = input_seq.shape
        
        # Token value embeddings (matches train_gpt.py lines 465-468)
        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        # 012 ... 012 structure
        ve_pattern = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
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
        
        # Tanh softcapping (matches train_gpt.py line 494)
        logits = 30 * torch.tanh(logits / (7.5 * x.size(-1)**0.5))
        
        return logits

def load_checkpoint_proper(checkpoint_path: str, device='cuda'):
    """Load model with proper architecture matching train_gpt.py"""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"Checkpoint loaded - Step: {checkpoint.get('step', 'Unknown')}")
    
    # Standard modded-nanogpt parameters
    model = GPT(
        vocab_size=50257,  # Will be rounded to 50304
        num_layers=12,
        num_heads=6,
        model_dim=768,
        max_seq_len=2048  # Smaller for inference
    )
    
    # Load state dict with proper key mapping
    state_dict = checkpoint['model']
    new_state_dict = {}
    
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key[10:]  # Remove _orig_mod. prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # Load with strict=False to handle rotary embeddings which are computed, not loaded
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    
    # Filter out expected missing keys (rotary embeddings are computed)
    expected_missing = [k for k in missing_keys if 'rotary.inv_freq' in k or 'rotary.cos_cached' in k or 'rotary.sin_cached' in k]
    actual_missing = [k for k in missing_keys if k not in expected_missing]
    
    if actual_missing:
        print(f"Actually missing keys: {len(actual_missing)}")
        for key in actual_missing[:5]:
            print(f"  - {key}")
        if len(actual_missing) > 5:
            print(f"  ... and {len(actual_missing) - 5} more")
    
    if expected_missing:
        print(f"Expected missing keys (rotary embeddings): {len(expected_missing)}")
    
    if unexpected_keys:
        print(f"Unexpected keys: {len(unexpected_keys)}")
    
    # Move to device
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
        print(f"Model moved to CUDA")
    else:
        device = 'cpu'
        print(f"Model on CPU")
    
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model ready - {total_params:,} parameters on {device}")
    
    return model, device

def generate_text(model, tokenizer, prompt: str, max_length: int = 50, 
                 temperature: float = 0.8, top_p: float = 0.9, top_k: int = 50, 
                 repetition_penalty: float = 1.1, device: str = 'cuda'):
    """Generate text from prompt"""
    model.eval()
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # Add batch dim
    
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            logits = model(tokens)
            
            # Get next token logits
            next_logits = logits[0, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(tokens[0].tolist()):
                    next_logits[token_id] /= repetition_penalty
            
            # Apply temperature
            next_logits = next_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_actual = min(top_k, next_logits.size(-1))
                indices_to_remove = next_logits < torch.topk(next_logits, top_k_actual)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[indices_to_remove] = float('-inf')
            
            # Convert to probabilities and sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            
            # Stop at EOS token
            if next_token.item() == 50256:
                break
    
    # Decode generated text
    generated_tokens = tokens[0].cpu().tolist()
    generated_text = tokenizer.decode(generated_tokens)
    
    return generated_text

def main():
    parser = argparse.ArgumentParser(description='Proper inference for modded-nanogpt')
    parser.add_argument('checkpoint', help='Path to checkpoint file')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--prompt', default='The quick brown fox', help='Text prompt')
    parser.add_argument('--max-length', type=int, default=50, help='Max generation length')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=0.9, help='Top-p (nucleus) sampling')
    parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--repetition-penalty', type=float, default=1.1, help='Repetition penalty')
    
    args = parser.parse_args()
    
    try:
        # Load model
        model, device = load_checkpoint_proper(args.checkpoint, args.device)
        
        # Load tokenizer
        print("Loading GPT-2 tokenizer...")
        tokenizer = tiktoken.get_encoding("gpt2")
        
        # Generate text
        print(f"\nGenerating from prompt: '{args.prompt}'")
        print("-" * 50)
        
        generated = generate_text(
            model, tokenizer, args.prompt, 
            max_length=args.max_length, 
            temperature=args.temperature,
            top_p=getattr(args, 'top_p', 0.9),
            top_k=getattr(args, 'top_k', 50), 
            repetition_penalty=getattr(args, 'repetition_penalty', 1.1),
            device=device
        )
        
        print(generated)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
