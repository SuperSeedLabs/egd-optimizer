import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
import numpy as np
import os
import math
from transformers import GPT2Tokenizer
from multiprocessing import freeze_support  # Add this for multiprocessing support
import time
from tokenizers import Tokenizer
from tqdm import tqdm

# Function definitions
def tokenize_function(examples):
    # Tokenize the texts
    tokenized = tokenizer(examples["content"], truncation=True)
    return tokenized

def get_batch(dataloader):
    data_iterator = iter(dataloader)
    for batch in data_iterator:
        yield batch["input_ids"].to(device), batch["labels"].to(device)

def compute_loss(logits, targets):
    # Reshape for cross entropy
    logits = logits.view(-1, vocab_size)
    targets = targets.view(-1)
    loss = F.cross_entropy(logits, targets)
    return loss

def evaluate(dataloader, model):
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for x, y in get_batch(dataloader):
            batch_size = x.size(0)
            logits = model(x)
            loss = compute_loss(logits, y)
            
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    
    return total_loss / total_samples

# Dataset class definition
class TextDataset(Dataset):
    def __init__(self, tokenized_dataset, block_size):
        self.examples = []
        
        for i in range(0, len(tokenized_dataset)):
            input_ids = tokenized_dataset[i]["input_ids"]
            
            # Create training examples by sliding window
            for j in range(0, len(input_ids) - block_size):
                # Get sequence of block_size tokens
                example = input_ids[j:j + block_size]
                
                # Labels are just the next tokens
                label = input_ids[j+1:j+1 + block_size]
                
                self.examples.append({"input_ids": example, "labels": label})
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.examples[idx]["input_ids"]),
            "labels": torch.tensor(self.examples[idx]["labels"])
        }

# Model class definitions
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Register buffer for causal attention mask (decoder-only)
        mask = torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)
    
    def forward(self, x):
        # Make sure x is on the same device as the model
        device = x.device
        
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Linear projections and reshape
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) 
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask (mask out future positions)
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

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Feed forward network with GELU activation
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter but should be part of the module's state)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention block with residual connection and layer norm
        # Create causal mask and move to same device as input
        seq_len = x.size(1)
        attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask, is_causal=True)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed-forward block with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, d_ff, max_seq_length, dropout=0.1):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        
        # Token embeddings + positional encoding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final linear layer for token prediction
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, x):
        # Make sure x is on the same device as the model
        device = next(self.parameters()).device
        x = x.to(device)
        
        # Get sequence length and batch size
        batch_size, seq_len = x.shape
        
        # Embed tokens and add positional encoding
        token_embed = self.token_embedding(x)  # [batch_size, seq_len, d_model]
        x = self.positional_encoding(token_embed)
        
        # Apply transformer decoder layers
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x)
            
        # Project to vocabulary
        logits = self.output_projection(x)  # [batch_size, seq_len, vocab_size]
        
        return logits
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # Get the device of the model
        device = next(self.parameters()).device
        
        # Move input to the same device as the model
        idx = idx.to(device)
        
        # Process the prompt (idx) and generate max_new_tokens
        for _ in range(max_new_tokens):
            # If idx exceeds max_seq_length, truncate it
            idx_cond = idx if idx.size(1) <= self.max_seq_length else idx[:, -self.max_seq_length:]
            
            # Get logits for the next token
            logits = self(idx_cond)
            
            # Focus on the last token's logits
            logits = logits[:, -1, :] / temperature  # (b, vocab_size)
            
            # Optional: Apply top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (b, vocab_size)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (b, 1)
            
            # Append the sampled token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (b, t+1)
        
        return idx

# Main execution code
if __name__ == '__main__':
    # Initialize multiprocessing support
    freeze_support()
    
    # 1. Load the BabyLM dataset
    print("Loading BabyLM dataset...")
    ds = load_dataset("AlgorithmicResearchGroup/babylm")

    # 2. Initialize tokenizer (still using GPT2 tokenizer as it's just for tokenization)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Parameters
    block_size = 128  # Context window size
    batch_size = 16
    vocab_size = tokenizer.vocab_size
    max_iters = 1000
    eval_interval = 500
    learning_rate = 3e-4

    # Set device first, before model creation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Process all splits
    print("Tokenizing dataset...")
    tokenized_datasets = {}
    for split in ds.keys():
        tokenized_datasets[split] = ds[split].map(
            tokenize_function,
            batched=True,
            remove_columns=["filename", "content"]
        )

    # Create datasets and dataloaders
    print("Creating training blocks...")
    train_dataset = TextDataset(tokenized_datasets["train"], block_size)
    val_dataset = TextDataset(tokenized_datasets["dev"], block_size)
    test_dataset = TextDataset(tokenized_datasets["test"], block_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 6. Initialize model
    print("Setting up model...")
    d_model = 384       # Embedding dimension
    num_heads = 6       # Number of attention heads
    num_layers = 6      # Number of transformer layers
    d_ff = 4 * d_model  # Feed-forward dimension

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_length=block_size,
        dropout=0.1
    )

    # Move model to device
    model = model.to(device)
    
    # Print model parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of model parameters: {num_params}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Initialize data iterators
    train_iterator = iter(get_batch(train_loader))
    val_iterator = iter(get_batch(val_loader))

    # Training loop
    print(f"Starting training with {max_iters} iterations")
    
    start_time = time.time()
    for iter in range(max_iters):
        # Training step
        model.train()
        try:
            x, y = next(train_iterator)
        except StopIteration:
            # Reinitialize iterator if it's exhausted
            train_iterator = iter(get_batch(train_loader))
            x, y = next(train_iterator)

        # Forward pass
        logits = model(x)
        loss = compute_loss(logits, y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluate every eval_interval
        if iter % eval_interval == 0 or iter == max_iters - 1:
            model.eval()
            with torch.no_grad():
                try:
                    val_x, val_y = next(val_iterator)
                except StopIteration:
                    val_iterator = iter(get_batch(val_loader))
                    val_x, val_y = next(val_iterator)
                
                val_logits = model(val_x)
                val_loss = compute_loss(val_logits, val_y)
                
                time_elapsed = time.time() - start_time
                print(f"Iter {iter}/{max_iters} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Time: {time_elapsed:.2f}s")
    
    # Generate text samples after training
    print("\nGenerating sample text:")
    model.eval()
    with torch.no_grad():
        # Sample a prompt from the test data
        test_iterator = iter(get_batch(test_loader))
        prompt, _ = next(test_iterator)
        
        # Generate text
        generated = model.generate(prompt, max_new_tokens=50, temperature=0.8, top_k=50)
        
        # Decode the generated tokens
        prompt_decoded = tokenizer.decode(prompt[0].cpu(), skip_special_tokens=True)
        generated_decoded = tokenizer.decode(generated[0].cpu(), skip_special_tokens=True)
        
        print(f"Prompt: {prompt_decoded}")
        print(f"Generated: {generated_decoded}")

    print("Done!") 