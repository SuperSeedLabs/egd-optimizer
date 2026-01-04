import torch
import datasets
import transformers
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import TrainingArguments, Trainer
import deepspeed

# Load dataset
babylm_dataset = datasets.load_dataset('AlgorithmicResearchGroup/babylm','plain_text')
train_dataset = babylm_dataset['train']
eval_dataset = babylm_dataset['dev']

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def tokenize_function(examples):
    return tokenizer(examples['content'], padding='max_length', max_length=1024, truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

train_data_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
eval_data_loader = DataLoader(eval_dataset, batch_size=2)

model = GPT2LMHeadModel.from_pretrained('gpt2')

# Training arguments
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir='./logs',
    output_dir='./results',
    logging_steps=500,
    evaluation_strategy='steps',
    eval_steps=1000,
    save_steps=1000,
    load_best_model_at_end=True,
    fp16=True,
    deepspeed='./deepspeed_config.json'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data_loader,
    eval_dataset=eval_data_loader,
)

# Start training
trainer.train()

# Evaluate the model
print('Evaluating the model...')
eval_loss = trainer.evaluate()['eval_loss']
print('Eval loss:', eval_loss)

# Perplexity
eval_perplexity = torch.exp(torch.tensor(eval_loss))
print('Perplexity:', eval_perplexity)

# Generate text
print('Generating text...')
generation_config = {
    'do_sample': True,
    'max_length': 100,
    'top_p': 0.9,
    'temperature': 0.8
}
prompts = ['Once upon a time', 'The quick brown fox', 'In a galaxy far away']
for prompt in prompts:
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    outputs = model.generate(input_ids, **generation_config)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))