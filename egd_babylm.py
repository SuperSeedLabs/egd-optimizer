# test_egd_baby_lm.py
# imports
import os
import asyncio
import math
import random
import time
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

from agent.egd.egd import EGD
from agent.egd.egd_registry import register_module, ModuleConfig
from agent.egd.utils import print_elapsed_time

# Force using a single GPU for training to avoid NCCL errors
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # Use only the first GPU


# Define the TransformerLanguageModel class for BabyLM
@register_module('babylm_gpt2')
class TransformerLM(nn.Module):
    """Transformer-based decoder-only language model for BabyLM dataset.

    This module creates a transformer decoder-only language model with
    configurable parameters for embedding size, layers, and attention heads.

    Attributes:
        vocab_size (int): Size of vocabulary
        d_model (int): Embedding dimension
        num_heads (int): Number of attention heads
        num_layers (int): Number of transformer layers
        d_ff (int): Feed-forward layer dimension
        max_seq_len (int): Maximum sequence length
        dropout (float): Dropout probability
    """

    def __init__(self, **kwargs):
        super().__init__()

        # Get required parameters with defaults
        self.vocab_size = kwargs.get('vocab_size', 50257)  # Default GPT2 vocab size
        self.d_model = kwargs.get('d_model', 256)  # Embedding dimension (reduced from 384)
        self.num_heads = kwargs.get('num_heads', 4)  # Number of attention heads (reduced from 6)
        self.num_layers = kwargs.get('num_layers', 4)  # Number of transformer layers (reduced from 6)
        self.d_ff = kwargs.get('d_ff', 4 * self.d_model)  # Feed-forward dimension
        self.max_seq_len = kwargs.get('max_seq_len', 128)  # Context window size
        self.dropout_rate = kwargs.get('dropout', 0.1)  # Dropout probability

        # Store the block size
        self.block_size = self.max_seq_len

        # Initialize embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.max_seq_len, self.d_model))

        # Initialize transformer layers
        self.layers = nn.ModuleList([
            self._create_transformer_layer()
            for _ in range(self.num_layers)
        ])

        self.dropout = nn.Dropout(self.dropout_rate)
        self.norm = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size)

        # Track number of parameters
        self.n_params = sum(p.numel() for p in self.parameters())

        # Initialize parameters
        self.apply(self._init_weights)

    def _create_transformer_layer(self):
        """Create a single transformer decoder layer"""
        return TransformerDecoderLayer(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            block_size=self.max_seq_len,
            dropout=self.dropout_rate
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, x):
        # Ensure both the model and input are on the same device
        device = next(self.parameters()).device
        x = x.to(device)
        
        seq_len = x.size(1)

        # Token embeddings + positional embeddings
        token_embed = self.token_embedding(x)
        pos_embed = self.position_embedding[:, :seq_len, :].to(device)
        x = token_embed + pos_embed
        x = self.dropout(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits

    def generate(self, idx, max_new_tokens, temperature=1.0, top_p=0.9):
        """Generate new tokens beyond the context provided in idx."""
        self.eval()
        device = next(self.parameters()).device
        idx = idx.to(device)
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop idx to block_size tokens if needed
                idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]

                # Get predictions
                logits = self(idx_cond)

                # Focus on the last token
                logits = logits[:, -1, :] / temperature

                # Apply top-p (nucleus) sampling
                if top_p < 1.0:
                    # Sort logits in descending order
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    # Calculate cumulative probabilities
                    sorted_probs = torch.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    # Create a sparse mask to scatter the indices back to their original ordering
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('inf')

                # Get probabilities
                probs = F.softmax(logits, dim=-1)

                # Sample
                idx_next = torch.multinomial(probs, num_samples=1)

                # Append to the sequence
                idx = torch.cat((idx, idx_next), dim=1)

        return idx


# Define helper classes for the transformer model
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
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()
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
        scores = scores.masked_fill(self.causal_mask[:seq_len, :seq_len], float('-inf'))

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
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, block_size, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, block_size, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with residual connection
        attn_output = self.self_attn(x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feed forward with residual connection
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x


# Define a dataset class for BabyLM
class TextDataset(Dataset):
    def __init__(self, tokenized_dataset, block_size):
        self.examples = []

        print(f"Creating dataset with {len(tokenized_dataset)} documents, block size {block_size}")

        for i in range(len(tokenized_dataset)):
            input_ids = tokenized_dataset[i]["input_ids"]

            # Print debugging info for the first document
            if i == 0:
                print(f"Document {i} has {len(input_ids)} tokens")

            # Create training examples with non-overlapping blocks
            # Allow for shorter sequences at the end by padding
            for j in range(0, len(input_ids) - 1, block_size // 2):  # 50% overlap for more examples
                # Get input sequence (ensure it's exactly block_size tokens)
                end_idx = min(j + block_size, len(input_ids))
                if end_idx - j < 2:  # Skip if too short
                    continue

                example = input_ids[j:end_idx]

                # Pad if necessary to reach block_size
                if len(example) < block_size:
                    example = example + [0] * (block_size - len(example))

                # Get target (shifted by 1)
                label_end_idx = min(j + 1 + block_size, len(input_ids))
                label = input_ids[j + 1:label_end_idx]

                # Pad if necessary
                if len(label) < block_size:
                    label = label + [0] * (block_size - len(label))

                self.examples.append({"input_ids": example, "labels": label})

                # Limit examples per document to prevent memory issues
                if len(self.examples) % 1000 == 0:
                    print(f"Created {len(self.examples)} examples so far")

        print(f"Created dataset with {len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.examples[idx]["input_ids"]),
            "labels": torch.tensor(self.examples[idx]["labels"])
        }


def load_babylm(
        data_percent: float = 0.01,
        block_size: int = 1024,
        device: torch.device = None,
        seed: Optional[int] = None
) -> Tuple[List, List, List]:
    """Load and preprocess the BabyLM dataset."""
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize tokenizer (using GPT2 tokenizer)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the BabyLM dataset
    print("Loading BabyLM dataset...")
    ds = load_dataset("AlgorithmicResearchGroup/babylm")
    
    # Print some info about the dataset
    print(f"Dataset structure: {ds}")
    for split in ds.keys():
        print(f"{split} split has {len(ds[split])} items")
        if len(ds[split]) > 0:
            print(f"First item in {split}: {list(ds[split][0].keys())}")
            print(f"Content sample: {ds[split][0]['content'][:100]}...")
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples["content"], truncation=True, max_length=100000)  # Limit token length
    
    # Process splits
    print("Tokenizing dataset...")
    tokenized_datasets = {}
    
    # Process train split - just use 1 file to ensure we have data
    split = "train"
    print(f"Processing {split} split")
    subset = ds[split].select(range(1))  # Just use the first file
    tokenized_datasets[split] = subset.map(
        tokenize_function,
        batched=True,
        remove_columns=["filename", "content"]
    )
    print(f"Tokenized {split} has {len(tokenized_datasets[split])} items")
    
    # Process dev/validation split
    split = "dev"
    print(f"Processing {split} split")
    subset = ds[split].select(range(1))  # Just use the first file
    tokenized_datasets[split] = subset.map(
        tokenize_function,
        batched=True,
        remove_columns=["filename", "content"]
    )
    print(f"Tokenized {split} has {len(tokenized_datasets[split])} items")
    
    # Process test split
    split = "test"
    print(f"Processing {split} split")
    subset = ds[split].select(range(1))  # Just use the first file
    tokenized_datasets[split] = subset.map(
        tokenize_function,
        batched=True,
        remove_columns=["filename", "content"]
    )
    print(f"Tokenized {split} has {len(tokenized_datasets[split])} items")
    
    # Create datasets with smaller block size to ensure we have examples
    smaller_block_size = min(block_size, 32)  # Use very small blocks if needed
    print(f"Creating datasets with block size {smaller_block_size}")
    train_dataset = TextDataset(tokenized_datasets["train"], smaller_block_size)
    val_dataset = TextDataset(tokenized_datasets["dev"], smaller_block_size)
    test_dataset = TextDataset(tokenized_datasets["test"], smaller_block_size)
    
    # Convert to list of (input, target) pairs for EGD
    def dataset_to_pairs(dataset):
        pairs = []
        # Make sure we get at least some examples
        num_examples = max(5, min(len(dataset), int(len(dataset) * data_percent)))
        
        print(f"Converting {num_examples} examples to input/target pairs")
        for i in range(num_examples):
            example = dataset[i]
            input_tensor = example["input_ids"].to(device)
            target_tensor = example["labels"].to(device)
            pairs.append((input_tensor, target_tensor))
            
            # Print shape of first example for debugging
            if i == 0:
                print(f"Example shapes: input {input_tensor.shape}, target {target_tensor.shape}")
        
        return pairs
    
    training_data = dataset_to_pairs(train_dataset)
    validation_data = dataset_to_pairs(val_dataset)
    test_data = dataset_to_pairs(test_dataset)
    
    print(f'Loaded {len(training_data)} training examples, {len(validation_data)} validation examples, '
          f'and {len(test_data)} testing examples.')
    
    # Extra check to ensure we have data
    if len(training_data) == 0 or len(validation_data) == 0:
        print("WARNING: No data loaded, creating dummy data for testing")
        # Create dummy data to allow code to run
        dummy_input = torch.randint(0, 1000, (smaller_block_size,)).to(device)
        dummy_target = torch.randint(0, 1000, (smaller_block_size,)).to(device)
        
        training_data = [(dummy_input, dummy_target)] * 10
        validation_data = [(dummy_input, dummy_target)] * 5
        test_data = [(dummy_input, dummy_target)] * 5
        
        print(f"Created {len(training_data)} dummy training examples")
    
    return training_data, validation_data, test_data


# Add this custom loss class to your imports section
class LanguageModelLoss(nn.Module):
    """Custom loss function for language modeling that handles reshaping
    tensors before applying cross entropy loss.

    This class automatically reshapes the model outputs and target tensors
    to be compatible with CrossEntropyLoss for language modeling tasks.

    Attributes:
        criterion (nn.CrossEntropyLoss): The underlying cross entropy loss function
    """

    def __init__(self) -> None:
        """Initialize the language model loss with CrossEntropyLoss."""
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate loss after reshaping tensors appropriately.

        Args:
            outputs: Model prediction logits, typically shape [batch_size, seq_len, vocab_size]
            targets: Target token indices, typically shape [batch_size, seq_len]

        Returns:
            torch.Tensor: Scalar loss value
        """
        # Print shapes for debugging (useful during development)
        # if hasattr(outputs, 'shape') and hasattr(targets, 'shape'):
        #     print(f"Before reshaping - outputs: {outputs.shape}, targets: {targets.shape}")

        # Reshape outputs from [batch_size, seq_len, vocab_size] to [batch_size*seq_len, vocab_size]
        # This flattens the batch and sequence dimensions for classification over vocabulary
        if isinstance(outputs, torch.Tensor) and outputs.dim() > 2:
            outputs = outputs.reshape(-1, outputs.size(-1))

        # Reshape targets from [batch_size, seq_len] to [batch_size*seq_len]
        # This flattens the targets to match the reshaped outputs
        if isinstance(targets, torch.Tensor) and targets.dim() > 1:
            targets = targets.reshape(-1)

        # Print reshaped dimensions for verification
        # if hasattr(outputs, 'shape') and hasattr(targets, 'shape'):
        #     print(f"After reshaping - outputs: {outputs.shape}, targets: {targets.shape}")

        # Apply cross entropy loss on the reshaped tensors
        return self.criterion(outputs, targets)


def main():
    '''Main function to test EGD with BabyLM dataset'''

    # Skip the database section entirely
    print("\n=== Running babylm evaluation with EGD framework ===\n")

    # Use the updated config from above
    meta_config = {
        'LAYERS_CONFIG_INIT': [
            ModuleConfig(
                module_type='babylm_gpt2',
                kwargs={
                    'vocab_size': 50257,  # Same as babylm.py
                    'd_model': 768,  # Match EMBED_DIM from babylm.py
                    'num_heads': 12,  # Match N_HEAD from babylm.py 
                    'num_layers': 6,  # Match N_LAYER from babylm.py
                    'max_seq_len': 1024,  # Match CONTEXT_L from babylm.py
                    'dropout': 0.1
                }
            )
        ],
        'LAYERS_CONFIG_OPTIONS': [[
            # Varying embedding dimensions
            ModuleConfig(
                module_type='babylm_gpt2',
                kwargs={
                    'vocab_size': 50257,
                    'd_model': 512,  # Smaller model
                    'num_heads': 8,
                    'num_layers': 4,
                    'max_seq_len': 1024,
                    'dropout': 0.1
                }
            )],
            [
                ModuleConfig(
                    module_type='babylm_gpt2',
                    kwargs={
                        'vocab_size': 50257,
                        'd_model': 768,  # Standard model
                        'num_heads': 12,
                        'num_layers': 6,
                        'max_seq_len': 1024,
                        'dropout': 0.1
                    }
                )],
            [
                ModuleConfig(
                    module_type='babylm_gpt2',
                    kwargs={
                        'vocab_size': 50257,
                        'd_model': 1024,  # Larger model
                        'num_heads': 16,
                        'num_layers': 8,
                        'max_seq_len': 1024,
                        'dropout': 0.1
                    }
                )],
            [
                # Varying dropout rates
                ModuleConfig(
                    module_type='babylm_gpt2',
                    kwargs={
                        'vocab_size': 50257,
                        'd_model': 768,
                        'num_heads': 12,
                        'num_layers': 6,
                        'max_seq_len': 1024,
                        'dropout': 0.05  # Lower dropout
                    }
                )],
            [
                ModuleConfig(
                    module_type='babylm_gpt2',
                    kwargs={
                        'vocab_size': 50257,
                        'd_model': 768,
                        'num_heads': 12,
                        'num_layers': 6,
                        'max_seq_len': 1024,
                        'dropout': 0.2  # Higher dropout
                    }
                )],
            [
                # Different layer/head combinations
                ModuleConfig(
                    module_type='babylm_gpt2',
                    kwargs={
                        'vocab_size': 50257,
                        'd_model': 768,
                        'num_heads': 8,  # Fewer heads
                        'num_layers': 8,  # More layers
                        'max_seq_len': 1024,
                        'dropout': 0.1
                    }
                )],
            [
                ModuleConfig(
                    module_type='babylm_gpt2',
                    kwargs={
                        'vocab_size': 50257,
                        'd_model': 768,
                        'num_heads': 16,  # More heads
                        'num_layers': 4,  # Fewer layers
                        'max_seq_len': 1024,
                        'dropout': 0.1
                    }
                )],
        ],
        'POPULATION_SIZE': 25,  # Use 100 models in population
        'GENERATIONS': 10,  # Run for 50 generations
        'EPOCHS': 25,  # Train each model for 100 epochs per generation
        'DEBUG': True,
        'BATCH_SIZE_INIT': 16,  # Initial batch size
        'BATCH_SIZE_OPTIONS': [(4, 128), 8, 32],  # Range of batch sizes
        'LEARNING_RATE_INIT': 5e-5,  # Initial learning rate
        'LEARNING_RATE_RANGE': (1e-5, 5e-4),  # Min and max learning rate
        'WEIGHT_DECAY_INIT': 0.01,  # Initial weight decay
        'WEIGHT_DECAY_RANGE': (0.001, 0.1),  # Min and max weight decay
    }

    # TODO: remove the constraint on the float precision
    # Set default tensor type to 32-bit float
    torch.set_default_dtype(torch.float32)

    # Ensure CUDA uses float32 if available
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    # Load BabyLM data with reduced percentage for training
    block_size = 1024  # Match the model's max_seq_len
    train_data_percent = 0.01  # Use 10% of the data for training
    (training_data, validation_data, _) = load_babylm(
        block_size=block_size,
        data_percent=train_data_percent
    )

    # Initialize EGD
    egd = EGD(
        training_data=training_data,
        validation_data=validation_data,
        loss_fn=LanguageModelLoss(),
        meta_config=meta_config,
        seed=42,
    )

    # Set log path
    egd.log_path = f'logs/babylm_evaluation.csv'

    print('\nRunning the evaluation training\n')

    # Start timing
    start_time = time.time()

    # Train with EGD
    best_net, most_acc = asyncio.run(egd.train())

    # End timing
    end_time = time.time()
    training_time = end_time - start_time

    print_elapsed_time(training_time)

    # Load full validation and test data for complete evaluation
    test_data_percent = 0.2 # Use 100% of data for testing
    print(f'\nLoading {test_data_percent*100}% of validation and test data for final evaluation\n')
    (_, full_validation_data, full_test_data) = load_babylm(
        block_size=block_size,
        data_percent=test_data_percent
    )

    # Test on the full validation data
    print(f'\nEvaluating the trained model on {test_data_percent*100}% validation data...\n')
    val_loss, _ = egd.test_net(best_net, full_validation_data)
    val_perplexity = math.exp(val_loss)
    print(f'\nValidation loss: {val_loss:.4f}, Perplexity: {val_perplexity:.2f}\n')
    
    # Test on the full test data
    print(f'\nEvaluating the trained model on {test_data_percent*100}% test data...\n')
    test_loss, _ = egd.test_net(best_net, full_test_data)
    test_perplexity = math.exp(test_loss)
    print(f'\nTest loss: {test_loss:.4f}, Perplexity: {test_perplexity:.2f}\n')
    print(f'Number of parameters: {best_net.n_params}\n')

    # Generate some sample text with the model
    print("\n=== Sample Text Generation ===\n")
    device = next(best_net.parameters()).device

    # Sample prompts
    prompts = [
        "Once upon a time",
        "The quick brown fox",
        "In a galaxy far away",
    ]

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for prompt in prompts:
        print(f"\n{'=' * 50}")
        print(f"PROMPT: {prompt}")
        print(f"{'=' * 50}")

        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Generate continuation with direct values
        output_ids = best_net.generate(
            input_ids,
            max_new_tokens=100,  # MAX_GEN_LENGTH value
            temperature=0.8,     # GEN_TEMPERATURE value
            top_p=0.9,           # GEN_TOP_P value
        )

        # Calculate the length of input to separate the generated part
        input_length = len(tokenizer.encode(prompt))
        continuation = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)

        print(f"\nGENERATED CONTINUATION:")
        print(f"{'-' * 50}")
        print(continuation)
        print(f"{'-' * 50}")


if __name__ == '__main__':
    main()
