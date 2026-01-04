from agent.supervisor import Supervisor
import time
import random
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.columns import Columns
import re
import json
import click
# from agent_eval import agent

console = Console()

def print_markdown_table(results):
    header = "| Metric                      | Value       |\n"
    separator = "|-----------------------------|-------------|\n"
    rows = "\n".join([f"| {metric:<27} | {value:<11} |" for metric, value in results])
    table = f"{header}{separator}{rows}"
    print(table)
    
    
def parse_json(input_string):
    # Use a regular expression to find the JSON part of the string
    json_match = re.search(r'\{.*\}', input_string)
    if json_match:
        json_part = json_match.group(0)
        # Replace single quotes with double quotes to make it a valid JSON
        json_part = json_part.replace("'", '"')
        # Parse the JSON
        parsed_json = json.loads(json_part)
        return parsed_json
    else:
        return None

def pretty_task(content):
        return f"[yellow] {content}"
    
    
def run_task(prompt, provider="openai"):
    user_id = 1
    run_id = random.getrandbits(32)
    
    user_renderables = [
        Panel(pretty_task(prompt), expand=True),
    ]
    console.print(Panel(Columns(user_renderables)))
                
    
    supervisor = Supervisor()
    supervisor_result = supervisor.run(user_id, run_id, prompt, provider)

    return supervisor_result


"""
Example:

python3 run.py --prompt "write an article on the history of python" --provider openai
"""

default_prompt = """
Train a GPT-2 model on the BabyLM dataset. You have access to 8 NVIDIA H100 GPUs, each with 80GB of HBM3 memory.

Dataset details:
- The BabyLM dataset can be accessed via Hugging Face: "AlgorithmicResearchGroup/babylm"
- It contains "train" and "dev" splits that should be used for training and validation respectively
- Each example in the dataset has "filename" and "content" fields, where "content" contains the text

Training configuration:
- Start with a lightweight GPT-2 architecture but scale it up to utilize the available GPU resources
- Use causal language modeling (predicting the next token) as the training objective
- The existing code uses hyperparameters like below but you can change them to get the best perplexity on the dev set:
  * Context length: 1024 tokens
  * Default embedding dimension: 768
  * Default layers: 6
  * Default attention heads: 12
  * These can be increased given the available hardware. The goal is to get the best perplexity on the dev set.

Distributed training:
- Modify the existing babylm.py script to leverage all 8 H100 GPUs
- Implement distributed training with PyTorch since it's already installed
- Consider using DeepSpeed, Megatron-LM, or other frameworks for model parallelism
- Adjust batch sizes and learning rates appropriately for multi-GPU training

Evaluation and generation:
- Evaluate the model using perplexity on the "dev" split
- Generate sample text using prompts like "Once upon a time", "The quick brown fox", and "In a galaxy far away"
- Use parameters like temperature=0.8, top_p=0.9, and max_generation_length=100

Your goal is to create a distributed training script that efficiently utilizes all 8 H100 GPUs to train the best GPT-2 model on this dataset while maintaining stability and reproducibility. Consider both data and model parallelism approaches to maximize throughput and model capacity.
Report the best loss and perplexity on the dev set you can achieve.
"""

@click.command()
@click.option('--prompt', type=str, help='The prompt to run', default=default_prompt)
@click.option('--provider', type=click.Choice(['openai', 'anthropic']), default='openai', help='The provider to use')
def main(prompt, provider):
    start = time.time()

    supervisor_result = run_task(prompt, provider)
    
    end = time.time()
    
    print(supervisor_result['result'])
    
    result = parse_json(str(supervisor_result['result']))

    try:
        print(f"Plan: {supervisor_result['plan']}")
        
        table = Table(title="Task Complete!!!")
        table.add_column("Mertic", justify="right", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Run ID", str(supervisor_result['run_number']))
        table.add_row("Submission", str(result['subtask_result']['submission']))
        table.add_row("Model Path", str(result['subtask_result']['model_path']))
        table.add_row("Total Tokens", str(supervisor_result['total_tokens']))
        table.add_row("Total Turns", str(supervisor_result['total_turns']))
        table.add_row("Time Taken in Seconds", str(end - start))
        table.add_row("Time Taken in Minutes", str((end - start) / 60))
        table.add_row("Time Taken in Hours", str((end - start) / 3600))

        console = Console()
        console.print(table)

    except Exception as e:
        print(f"An error occurred: {e}")
        
        
        
    print_markdown_table([
        ("Run ID", supervisor_result['run_number']),
        ("Submission", str(result['subtask_result']['submission'])),
        ("Model Path", str(result['subtask_result']['model_path'])),
        ("Total Tokens", supervisor_result['total_tokens']),
        ("Total Turns", supervisor_result['total_turns']),
        ("Time Taken in Seconds", end - start),
        ("Time Taken in Minutes", (end - start) / 60),
        ("Time Taken in Hours", (end - start) / 3600),
    ])
    

    print("Task complete")


if __name__ == "__main__":
    main()