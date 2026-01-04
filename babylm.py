# Set environment variables for CUDA and bitsandbytes support
import os
# Add ~/.cuda/lib to LD_LIBRARY_PATH
cuda_lib_path = os.path.expanduser("~/.cuda/lib")
if os.path.exists(cuda_lib_path):
    os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

# Let bitsandbytes know we're using CUDA 12.4
os.environ["BNB_CUDA_VERSION"] = "124"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Force using a single GPU for training to avoid NCCL errors
os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # Use only the first GPU

import torch
from typing import Dict, Tuple, Any, List, Optional
from datasets import load_dataset, DatasetDict
import random
import numpy as np
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)

"""
BabyLM Model Training Script

This script trains a lightweight GPT-2 language model from scratch on the BabyLM dataset.
BabyLM is a dataset designed for child language acquisition research, containing filtered, 
age-appropriate content for training language models with more natural language patterns.

The script:
1. Initializes a GPT-2 model architecture with smaller capacity than the full model
2. Loads and preprocesses the BabyLM dataset
3. Trains the model using causal language modeling (predicting the next token)
4. Saves the trained model and tokenizer for later use

The hyperparameters are configurable through the hyperparams dictionary, making it easy
to experiment with different model sizes and training configurations.

Output: A trained language model saved to the specified OUTPUT_DIR that can be used
for text generation, fine-tuning, or analysis of learned linguistic patterns.


NOTE: LLMs (claude 3.7 sonnet thinking) have trouble implementing the distributed training

"""

# Configuration dictionary with all hyperparameters
hyperparams = {
    # Model configuration
    "MODEL_NAME": "gpt2",        # Architecture to base our model on
    "VOCAB_SIZE": 50257,         # GPT-2's vocabulary size
    "OUTPUT_DIR": "babylm_model",
    
    # Model architecture
    "CONTEXT_L": 1024,         # Maximum sequence length
    "EMBED_DIM": 768,              # Embedding dimension
    "N_LAYER": 6,                # Number of transformer layers
    "N_HEAD": 12,                # Number of attention heads
    
    # Training configuration
    "NUM_EPOCHS": 1000,
    "TRAIN_BATCH_SIZE": 8192,    # Increased from 64 to 8192
    "EVAL_BATCH_SIZE": 8192,     # Increased from 64 to 8192
    "EVAL_STEPS": 500,
    "SAVE_STEPS": 1000,
    "WARMUP_STEPS": 500,
    "LEARNING_RATE": 5e-5,
    "WEIGHT_DECAY": 0.01,
    "LOGGING_STEPS": 100,
    "MAX_SEQ_LENGTH": 1024,
    "USE_MP": False, # disable mixed precision training
    
    # Experiment configuration
    "DEFAULT_SEED": 42,
    
    # Generation configuration
    "MAX_GEN_LENGTH": 100,
    "GEN_TEMPERATURE": 0.8,
    "GEN_TOP_P": 0.9,
}

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: The random seed to set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set seed for HuggingFace transformers
    os.environ["PYTHONHASHSEED"] = str(seed)

def create_model_from_scratch() -> GPT2LMHeadModel:
    """
    Initialize a GPT-2 language model with custom configuration.
    
    Returns:
        GPT2LMHeadModel: A newly initialized GPT-2 model
    """
    config = GPT2Config(
        vocab_size=hyperparams["VOCAB_SIZE"],
        n_positions=hyperparams["CONTEXT_L"],
        n_embd=hyperparams["EMBED_DIM"],
        n_layer=hyperparams["N_LAYER"],
        n_head=hyperparams["N_HEAD"],
    )
    model = GPT2LMHeadModel(config)
    model.init_weights()
    return model

def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_and_preprocess_data() -> Tuple[DatasetDict, PreTrainedTokenizer]:
    """
    Load BabyLM dataset and preprocess it for language modeling.
    
    Returns:
        Tuple[DatasetDict, PreTrainedTokenizer]: Processed dataset and tokenizer
    """
    # Load the dataset
    dataset = load_dataset("AlgorithmicResearchGroup/babylm")
    
    # Print dataset information
    print("\n========== Dataset Information ==========")
    print(f"Dataset structure: {dataset}")
    
    # Print statistics for each split
    for split in dataset.keys():
        print(f"\n{split} split:")
        print(f"  • Number of examples: {len(dataset[split])}")
        if len(dataset[split]) > 0:
            # Print some info about the first example
            print(f"  • Example fields: {list(dataset[split][0].keys())}")
            # Print a snippet of the first example content
            content = dataset[split][0]["content"]
            print(f"  • First example content (truncated): {content[:100]}...")
            # Print length statistics
            content_lengths = [len(ex["content"]) for ex in dataset[split].select(range(min(100, len(dataset[split]))))]
            if content_lengths:
                print(f"  • Average content length (first 100 examples): {sum(content_lengths)/len(content_lengths):.1f} chars")
    print("==========================================\n")
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hyperparams["MODEL_NAME"])
    
    # Set the padding token to be the same as the EOS token
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize text examples in a batched way."""
        return tokenizer(
            examples["content"], 
            truncation=True, 
            max_length=hyperparams["MAX_SEQ_LENGTH"], 
            padding="max_length"
        )
    
    # Apply tokenization to the dataset, removing raw text columns
    tokenized_datasets = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["filename", "content"]
    )
    
    # Print tokenized dataset information
    print("\n========== Tokenized Dataset Information ==========")
    print(f"Tokenized dataset structure: {tokenized_datasets}")
    for split in tokenized_datasets.keys():
        print(f"\n{split} split:")
        print(f"  • Number of examples: {len(tokenized_datasets[split])}")
        if len(tokenized_datasets[split]) > 0:
            print(f"  • Example fields: {list(tokenized_datasets[split][0].keys())}")
            # Print token statistics
            token_lengths = [len([t for t in ex["input_ids"] if t != tokenizer.pad_token_id]) 
                             for ex in tokenized_datasets[split].select(range(min(100, len(tokenized_datasets[split]))))]
            if token_lengths:
                print(f"  • Average sequence length (first 100 examples): {sum(token_lengths)/len(token_lengths):.1f} tokens")
    print("====================================================\n")
    
    return tokenized_datasets, tokenizer

def create_training_args(seed: int, output_dir: Optional[str] = None) -> TrainingArguments:
    """
    Configure the training arguments for the Hugging Face Trainer.
    
    Args:
        seed: Random seed for reproducibility
        output_dir: Optional override for output directory
    
    Returns:
        TrainingArguments: Configuration for the training process
    """
    experiment_output_dir = output_dir if output_dir else hyperparams["OUTPUT_DIR"]
    return TrainingArguments(
        output_dir=experiment_output_dir,
        overwrite_output_dir=True,
        num_train_epochs=hyperparams["NUM_EPOCHS"],
        per_device_train_batch_size=hyperparams["TRAIN_BATCH_SIZE"],
        per_device_eval_batch_size=hyperparams["EVAL_BATCH_SIZE"],
        eval_steps=hyperparams["EVAL_STEPS"],
        save_steps=hyperparams["SAVE_STEPS"],
        warmup_steps=hyperparams["WARMUP_STEPS"],
        learning_rate=hyperparams["LEARNING_RATE"],
        weight_decay=hyperparams["WEIGHT_DECAY"],
        logging_dir=f"{experiment_output_dir}/logs",
        logging_steps=hyperparams["LOGGING_STEPS"],
        eval_strategy="steps",
        fp16=hyperparams["USE_MP"],  # Disable mixed precision training
        seed=seed,  # Set the seed in the training arguments
        deepspeed=None,  # Explicitly disable DeepSpeed
        no_cuda=False,  # Use CUDA
        local_rank=-1,  # Disable distributed training
        dataloader_num_workers=4
    )

def run_experiment(seed: int, experiment_name: str) -> None:
    """
    Run a single experiment with a specific seed.
    
    Args:
        seed: Random seed for this experiment
        experiment_name: Name for this experiment (used for output directory)
    """
    # Set seed for reproducibility
    set_seed(seed)
    
    # Create output directory for this experiment
    output_dir = f"{hyperparams['OUTPUT_DIR']}_{experiment_name}_seed{seed}"
    
    print(f"Starting experiment '{experiment_name}' with seed {seed}")
    print(f"Experiment output will be saved to {output_dir}")
    
    # Initialize the model
    model = create_model_from_scratch()
    
    # Print model statistics
    total_params = count_parameters(model)
    print(f"\n========== Model Statistics ==========")
    print(f"Model architecture: GPT-2 with {hyperparams['N_LAYER']} layers and {hyperparams['N_HEAD']} attention heads")
    print(f"Embedding dimension: {hyperparams['EMBED_DIM']}")
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Context length: {hyperparams['CONTEXT_L']} tokens")
    print("======================================\n")
    
    # Load and preprocess the data
    tokenized_datasets, tokenizer = load_and_preprocess_data()
    
    # Set the padding token ID for the model
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Configure training arguments with this seed
    training_args = create_training_args(seed, output_dir)
    
    # Store metrics history
    metrics_history = []
    
    # Add callback to print evaluation metrics during training
    class MetricsCallback(TrainerCallback):
        """Callback to display and store evaluation metrics during model training."""
        
        def on_evaluate(self, args: TrainingArguments, state: TrainerState, 
                        control: TrainerControl, metrics: Dict[str, float], **kwargs) -> None:
            """Called after model evaluation to display metrics."""
            # Store metrics for later analysis
            metrics_history.append((state.global_step, metrics.copy()))
            
            # Format and print the current step and evaluation metrics
            print(f"Step {state.global_step}: Eval results: {metrics}")
    
    # Initialize the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["dev"],
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=False  # Use causal language modeling (not masked)
        ),
    )
    
    # Register the callback with the trainer
    trainer.add_callback(MetricsCallback())
    
    # Run the training process
    train_result = trainer.train()
    
    # Print final training metrics
    print("\n========== Final Training Metrics ==========")
    print(f"Training loss: {train_result.training_loss:.4f}")
    
    # Run final evaluation
    eval_results = trainer.evaluate()
    print("\n========== Final Evaluation Metrics ==========")
    print(f"Evaluation loss: {eval_results['eval_loss']:.4f}")
    print(f"Perplexity: {np.exp(eval_results['eval_loss']):.2f}")
    print(f"Model size: {count_parameters(model):,} parameters")
    print("==============================================\n")
    
    # Save the model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Generate sample text
    print("\n========== Sample Generation ==========")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Sample prompts to generate from
    prompts = [
        "Once upon a time",
        "The quick brown fox",
        "In a galaxy far away",
    ]
    
    for prompt in prompts:
        print(f"\n{'='*50}")
        print(f"PROMPT: {prompt}")
        print(f"{'='*50}")
        
        # Tokenize with explicit attention mask
        encoded_input = tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            return_attention_mask=True
        )
        
        # Move inputs to device
        input_ids = encoded_input["input_ids"].to(device)
        attention_mask = encoded_input["attention_mask"].to(device)
        
        # Generate text
        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=hyperparams["MAX_GEN_LENGTH"],
                temperature=hyperparams["GEN_TEMPERATURE"],
                top_p=hyperparams["GEN_TOP_P"],
                no_repeat_ngram_size=3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Calculate the length of input to separate the generated part
        input_length = len(tokenizer.encode(prompt))
        generated_text = tokenizer.decode(output[0][input_length-1:], skip_special_tokens=True)
        
        print(f"\nGENERATED CONTINUATION:")
        print(f"{'-'*50}")
        print(generated_text)
        print(f"{'-'*50}")
    
    print("=====================================\n")
    
    print(f"Experiment '{experiment_name}' completed. Model saved to {output_dir}")

def run_multiple_experiments(seeds: List[int], experiment_base_name: str = "experiment") -> None:
    """
    Run multiple experiments with different seeds.
    
    Args:
        seeds: List of seeds to use for different experiment runs
        experiment_base_name: Base name for experiments
    """
    print(f"Running {len(seeds)} experiments with seeds: {seeds}")
    
    for i, seed in enumerate(seeds):
        experiment_name = f"{experiment_base_name}_{i+1}"
        run_experiment(seed, experiment_name)
    
    print(f"All {len(seeds)} experiments completed!")

def main() -> None:
    """
    Main function to train a language model on the BabyLM dataset.
    
    Options:
    1. Run a single experiment with default seed
    2. Run multiple experiments with different seeds
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train BabyLM models with specified seeds')
    parser.add_argument('--seeds', type=int, nargs='+', default=[hyperparams["DEFAULT_SEED"]], 
                        help='List of seeds for multiple experiments')
    parser.add_argument('--experiment_name', type=str, default="experiment", 
                        help='Base name for the experiments')
    args = parser.parse_args()
    
    if len(args.seeds) == 1:
        # Run a single experiment with the provided seed
        run_experiment(args.seeds[0], args.experiment_name)
    else:
        # Run multiple experiments with different seeds
        run_multiple_experiments(args.seeds, args.experiment_name)
    
if __name__ == "__main__":
    # Set the environment variable to disable tokenizers parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()