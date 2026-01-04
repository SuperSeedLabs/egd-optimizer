from datasets import load_dataset

def load_babylm_dataset():
    dataset = load_dataset("AlgorithmicResearchGroup/babylm")
    train_dataset = dataset['train']
    dev_dataset = dataset['dev']
    
    print(f'Training examples: {len(train_dataset)}')
    print(f'Development examples: {len(dev_dataset)}')
    
    return train_dataset, dev_dataset

if __name__ == "__main__":
    load_babylm_dataset()