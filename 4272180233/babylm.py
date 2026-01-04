from datasets import load_dataset
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
import deepspeed

# Load BabyLM dataset
dataset = load_dataset("AlgorithmicResearchGroup/babylm")
train_dataset = dataset['train']
dev_dataset = dataset['dev']

# Load pre-trained model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Config - adjust based on available resources
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    weight_decay=0.01,
    fp16=True,
    deepspeed="./ds_config.json",  # Configuration file for DeepSpeed setup
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
)

# Training
trainer.train()