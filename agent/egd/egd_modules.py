import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Assuming egd_registry is sibling to this directory or adjust path
from .egd_registry import register_module

# Define helper classes for the transformer model
@register_module('multi_head_attention')
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size, dropout=0.1):
        super().__init__()
        # Ensure dimensions are integers
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.block_size = int(block_size) # Cast block_size to int

        self.head_dim = self.d_model // self.num_heads
        assert self.head_dim * self.num_heads == self.d_model, "d_model must be divisible by num_heads"

        # Linear projections
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)

        self.dropout = nn.Dropout(float(dropout)) # Ensure dropout is float

        # Register buffer for causal attention mask (decoder-only)
        self.register_buffer(
            "causal_mask",
            # Now self.block_size is guaranteed to be an int
            torch.triu(torch.ones(self.block_size, self.block_size), diagonal=1).bool()
        )

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Linear projections and reshape
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask (mask out future positions)
        # Use the buffer directly, slicing up to the current sequence length
        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(mask, float('-inf'))

        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)

        # Final linear projection
        out = self.out_proj(attn_output)

        return out


@register_module('feed_forward')
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # Ensure dimensions are integers
        d_model = int(d_model)
        d_ff = int(d_ff)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(float(dropout)) # Ensure dropout is float

    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x) # Using GELU activation
        x = self.dropout(x)
        x = self.linear2(x)
        return x


@register_module('transformer_decoder_layer')
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, block_size, dropout=0.1):
        super().__init__()
        # Ensure dimensions are integers before passing them down
        d_model = int(d_model)
        num_heads = int(num_heads)
        d_ff = int(d_ff)
        block_size = int(block_size) # Cast block_size to int
        dropout = float(dropout) # Ensure dropout is float

        # Use the MultiHeadAttention class defined above
        self.self_attn = MultiHeadAttention(d_model, num_heads, block_size, dropout)
        # Use the FeedForward class defined above
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-Normalization (often more stable) could be an option here,
        # but following the original structure which looks like Post-Normalization.

        # Self-attention block
        attn_output = self.self_attn(x)
        x = x + self.dropout1(attn_output) # Residual connection
        x = self.norm1(x) # Post-normalization

        # Feed forward block
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output) # Residual connection
        x = self.norm2(x) # Post-normalization

        return x 