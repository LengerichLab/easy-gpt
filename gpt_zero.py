import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Block
from gpt_config import GPTConfig

class GPT(nn.Module):
    """
    GPT Language Model:
    - Transformer decoder stack
    - Token + positional embeddings
    - Predicts next token using causal self-attention
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.block_size = config.block_size
        self.device = config.device

        # Token and positional embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)

        # Transformer blocks
        self.transformer = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        # Final normalization and output projection
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        """
        Forward pass through the GPT model.

        Inputs:
            idx: Tensor of shape (B, T) — token indices
            targets: Optional Tensor of shape (B, T) — target indices for loss

        Returns:
            logits: (B, T, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
        """
        B, T = idx.shape
        assert T <= self.block_size, "Sequence length exceeds block size"

        # Embed tokens and positions
        token_embeddings = self.token_embedding(idx)                   # (B, T, C)
        position_embeddings = self.position_embedding(torch.arange(T, device=self.device))  # (T, C)
        x = token_embeddings + position_embeddings                     # (B, T, C)

        # Transformer stack and final projection
        x = self.transformer(x)                                        # (B, T, C)
        x = self.ln_f(x)                                               # (B, T, C)
        logits = self.lm_head(x)                                       # (B, T, vocab_size)

        # Optional training loss
        loss = None
        if targets is not None:
            logits_flat = logits.view(B * T, -1)                       # (B*T, vocab_size)
            targets_flat = targets.view(B * T)                         # (B*T,)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Autoregressive generation.

        Inputs:
            idx: (B, T) initial context tokens
            max_new_tokens: number of tokens to generate

        Returns:
            idx: (B, T + max_new_tokens) full sequence
        """
        for _ in range(max_new_tokens):
            # Crop to last block_size tokens (for long contexts)
            idx_cond = idx[:, -self.block_size:]

            # Forward pass
            logits, _ = self(idx_cond)

            # Select logits for the last time step
            last_logits = logits[:, -1, :]                            # (B, vocab_size)

            # Convert to probabilities and sample
            probs = F.softmax(last_logits, dim=-1)                    # (B, vocab_size)
            next_token = torch.multinomial(probs, num_samples=1)     # (B, 1)

            # Append sampled token
            idx = torch.cat((idx, next_token), dim=1)                # (B, T+1)

        return idx
