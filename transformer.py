import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlp import FeedForward

class LayerNorm(nn.Module):
    """
    Layer normalization with optional bias.
    Equivalent to PyTorch's nn.LayerNorm, but allows disabling the bias term.
    """
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(
            x,
            normalized_shape=self.weight.shape,
            weight=self.weight,
            bias=self.bias,
            eps=1e-5
        )

class CausalSelfAttention(nn.Module):
    """
    Multi-head masked self-attention module.
    Optionally uses Flash Attention for improved efficiency (if available).
    """
    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads."

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.use_flash = hasattr(F, 'scaled_dot_product_attention')

        # Project input to queries, keys, and values (all heads at once)
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # Output projection after attention
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Dropout layers
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Mask for causal attention (only needed for non-flash fallback)
        if not self.use_flash:
            print("WARNING: Flash Attention not available. Using slower implementation.")
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(config.block_size, config.block_size))
                     .view(1, 1, config.block_size, config.block_size)
            )

    def forward(self, x):
        B, T, C = x.shape

        # Compute queries, keys, values
        q, k, v = self.qkv_proj(x).split(C, dim=2)

        # Reshape for multi-head attention: (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_flash:
            # Efficient attention using PyTorch's Flash Attention
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # Manual implementation of masked dot-product attention
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, n_head, T, T)
            scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
            weights = F.softmax(scores, dim=-1)
            weights = self.attn_dropout(weights)
            attn_out = weights @ v  # (B, n_head, T, head_dim)

        # Recombine attention heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)

        # Final output projection
        out = self.out_proj(attn_out)
        out = self.resid_dropout(out)
        return out

class Block(nn.Module):
    """
    Transformer block:
    - LayerNorm → Causal Self-Attention → Residual Add
    - LayerNorm → FeedForward → Residual Add
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.ffwd = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # Self-attention + residual
        x = x + self.ffwd(self.ln_2(x))  # Feedforward + residual
        return x
