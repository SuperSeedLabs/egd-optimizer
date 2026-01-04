from datasets import load_dataset

def load_babylm_dataset():
    dataset = load_dataset('AlgorithmicResearchGroup/babylm')
    train_dataset = dataset['train']
    dev_dataset = dataset['dev']
    return train_dataset, dev_dataset

if __name__ == '__main__':
    train_dataset, dev_dataset = load_babylm_dataset()
    print(f"Train Dataset: {len(train_dataset)} samples")
    print(f"Dev Dataset: {len(dev_dataset)} samples")