from datasets import load_dataset

# Load the BabyLM dataset
dataset = load_dataset('AlgorithmicResearchGroup/babylm')

# Explore the structure of the dataset
train_split = dataset['train']
dev_split = dataset['dev']

# Print some sample data to understand the structure
print('Sample from training set:', train_split[0])
print('Sample from dev set:', dev_split[0])