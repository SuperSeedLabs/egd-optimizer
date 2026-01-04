# test_egd_baby_lm.py
# imports
import asyncio
import random
import time
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
import math
import os

from agent.egd.egd import EGD
from agent.egd.egd_registry import register_module, ModuleConfig, ModuleRegistryDB
from agent.egd.utils import print_elapsed_time


# Define the TransformerLanguageModel class for BabyLM
@register_module('baby_lm_transformer')
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
        seq_len = x.size(1)
        
        # Token embeddings + positional embeddings
        token_embed = self.token_embedding(x)
        pos_embed = self.position_embedding[:, :seq_len, :]
        x = token_embed + pos_embed
        x = self.dropout(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate new tokens beyond the context provided in idx."""
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop idx to block_size tokens if needed
                idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
                
                # Get predictions
                logits = self(idx_cond)
                
                # Focus on the last token
                logits = logits[:, -1, :] / temperature
                
                # Optional top-k sampling
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')
                
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
                label = input_ids[j+1:label_end_idx]
                
                # Pad if necessary
                if len(label) < block_size:
                    label = label + [0] * (block_size - len(label))
                
                self.examples.append({"input_ids": example, "labels": label})
                
                # Limit examples per document to prevent memory issues
                if len(self.examples) % 1000 == 0:
                    print(f"Created {len(self.examples)} examples so far")
                
                if len(self.examples) >= 5000:  # Cap per document
                    break
            
            if len(self.examples) >= 10000:  # Cap total
                break
        
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
        block_size: int = 128,
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
    tensors before applying cross entropy loss."""
    
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, outputs, targets):
        # Print shapes for debugging
        # if hasattr(outputs, 'shape') and hasattr(targets, 'shape'):
        #     print(f"Before reshaping - outputs: {outputs.shape}, targets: {targets.shape}")
        
        # Reshape outputs to [batch_size*seq_len, vocab_size]
        if isinstance(outputs, torch.Tensor) and outputs.dim() > 2:
            outputs = outputs.reshape(-1, outputs.size(-1))
        
        # Reshape targets to [batch_size*seq_len]
        if isinstance(targets, torch.Tensor) and targets.dim() > 1:
            targets = targets.reshape(-1)
        
        # # Print reshaped dimensions
        # if hasattr(outputs, 'shape') and hasattr(targets, 'shape'):
        #     print(f"After reshaping - outputs: {outputs.shape}, targets: {targets.shape}")
        
        # Apply cross entropy loss
        return self.criterion(outputs, targets)


def main():
    '''Main function to test EGD with BabyLM dataset'''

    # Define a unique module name for database test
    db_module_name = "babylm_transformer_db_test"

    # Initialize database connection
    registry = ModuleRegistryDB()

    if registry.connected:
        print(f"\n=== Registering '{db_module_name}' module to database ===\n")

        # First, check if module exists and delete it to start fresh
        existing = registry.modules_collection.find_one({"name": db_module_name})
        if existing:
            print(f"Found existing module with name '{db_module_name}', deleting it first")
            registry.delete_module(str(existing["_id"]))

        # Save the module with serializable parameters
        module_id = registry.save_module(
            module_class=TransformerLM,
            name=db_module_name,
            description="Transformer decoder-only language model for BabyLM dataset (DB version)",
            category="transformer",
            tags=["babylm", "transformer", "language_model", "test"],
            example_code=f"model = {db_module_name}(vocab_size=50257, d_model=256, num_heads=4, num_layers=4)",
            example_inputs={
                "vocab_size": 50257,  # GPT2 vocab size
                "d_model": 128,       # Very small embedding size for testing
                "num_heads": 2,        # Small number of heads
                "num_layers": 2,       # Small number of layers
                "max_seq_len": 64,     # Small context window
                "dropout": 0.1
            }
        )

        if module_id:
            print(f"\nSaved '{db_module_name}' module to database with ID: {module_id}")

            # Create a very small transformer config to minimize memory usage
            db_lm_config = [
                ModuleConfig(
                    module_type=db_module_name,
                    kwargs={
                        "vocab_size": 50257,
                        "d_model": 128,
                        "num_heads": 2,
                        "num_layers": 2,
                        "max_seq_len": 64,
                        "dropout": 0.1
                    }
                )
            ]

            # Close connection before EGD initialization
            registry.close()

            # Use very small data size and minimal training for testing
            meta_config = {
                'POPULATION_SIZE': 100,    # Minimum population
                'GENERATIONS': 1,        # Just one generation for testing
                'EPOCHS': 2,             # Minimum epochs for testing
                'DEBUG': True,
                'LAYERS_CONFIG_INIT': db_lm_config,
                'LAYERS_CONFIG_OPTIONS': [db_lm_config],
                'BATCH_SIZE_INIT': 4,    # Very small batches to reduce memory
            }
            
            # Load tiny fraction of BabyLM data
            block_size = 64  # Match the model's max_seq_len
            (training_data, validation_data, test_data) = load_babylm(
                data_percent=0.01,
                block_size=block_size
            )

            try:
                # Initialize EGD with our config
                egd = EGD(
                    training_data=training_data,
                    validation_data=validation_data,
                    loss_fn=LanguageModelLoss(),
                    meta_config=meta_config,
                    seed=42
                )

                print('\nRunning the population based training with database module\n')

                # Train the model
                best_net, most_acc = asyncio.run(egd.train())

                # Test results with small test set
                print('\nTesting the model loaded from database...\n')
                # Use only the first few test samples
                small_test_data = test_data[:10]
                loss, _ = egd.test_net(best_net, small_test_data)
                perplexity = math.exp(loss)
                print(f'\nTest loss: {loss:.4f}, Perplexity: {perplexity:.2f}\n')

                print("Database module test completed successfully!")

            except Exception as e:
                print(f"Error during EGD with database module: {e}")
                import traceback
                traceback.print_exc()

            # Clean up the test module
            print("\n=== Cleaning up database ===")
            registry = ModuleRegistryDB()
            if registry.connected:
                if registry.delete_module(module_id):
                    print(f"Successfully deleted test module from database")
                registry.close()
        else:
            print("Failed to save module to database. Skipping database test.")
            registry.close()
    else:
        print("Could not connect to MongoDB. Running the standard test only.")

    # Run the standard test with local modules and reduced memory usage
    print("\n=== Running standard test with local modules ===\n")

    # Define different transformer configurations for EGD to explore
    lm_config_small = [
        ModuleConfig(
            module_type='baby_lm_transformer',
            kwargs={
                'vocab_size': 50257,
                'd_model': 128,
                'num_heads': 2,
                'num_layers': 2,
                'max_seq_len': 64,
                'dropout': 0.1
            }
        )
    ]
    
    lm_config_medium = [
        ModuleConfig(
            module_type='baby_lm_transformer',
            kwargs={
                'vocab_size': 50257,
                'd_model': 192,
                'num_heads': 3,
                'num_layers': 3,
                'max_seq_len': 64,
                'dropout': 0.1
            }
        )
    ]

    # Configuration dictionary for local test with reduced resources
    meta_config = {
        'POPULATION_SIZE': 20,        # Small population
        'GENERATIONS': 5,            # Few generations
        'EPOCHS': 10,                 # Few epochs
        'DEBUG': True,
        'LAYERS_CONFIG_INIT': lm_config_small,
        'LAYERS_CONFIG_OPTIONS': [
            lm_config_small,
            lm_config_medium,
        ],
        'BATCH_SIZE_INIT': 32,
        'BATCH_SIZE_OPTIONS': [16, 32, 64],
        'LEARNING_RATE_INIT': 5e-4,
        'LEARNING_RATE_OPTIONS': [1e-4, 5e-4, 1e-3],
    }
    
    # Load BabyLM data with very small percent
    block_size = 64  # Match the model's max_seq_len
    (training_data, validation_data, test_data) = load_babylm(
        data_percent=0.2,
        block_size=block_size
    )
    
    # Initialize EGD
    egd = EGD(
        training_data=training_data,
        validation_data=validation_data,
        loss_fn=LanguageModelLoss(),
        meta_config=meta_config,
        seed=42
    )

    # Set log path
    egd.log_path = f'logs/babylm.csv'

    print('\nRunning the population based training\n')

    # Start timing
    start_time = time.time()

    # Train with EGD
    best_net, most_acc = asyncio.run(egd.train())

    # End timing
    end_time = time.time()
    training_time = end_time - start_time

    print_elapsed_time(training_time)

    print(f'Configuration: Population size={meta_config["POPULATION_SIZE"]}, '
          f'Generations={meta_config["GENERATIONS"]}, '
          f'Epochs={meta_config["EPOCHS"]}')

    # Test the best performing network
    print('\nTesting the best performing network...\n')
    loss, _ = egd.test_net(best_net, test_data[:20])
    perplexity = math.exp(loss)
    print('\nTesting complete\n')
    print(f'\nTest loss: {loss:.4f}, Perplexity: {perplexity:.2f}\n')
    print(f'Number of parameters: {best_net.n_params}\n')
    print(f'Network architecture: {best_net}\n')

    # Test the most accurate network
    print('\nTesting the most accurate network...\n')
    loss, _ = egd.test_net(most_acc, test_data[:20])
    perplexity = math.exp(loss)
    print('\nTesting complete\n')
    print(f'\nTest loss: {loss:.4f}, Perplexity: {perplexity:.2f}\n')
    print(f'Number of parameters: {most_acc.n_params}\n')
    print(f'Network architecture: {most_acc}\n')


if __name__ == '__main__':
    main()
