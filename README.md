# egd-agent
repo for build and eval of the egd agent on various benchmarks

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python runner.py
```


# Results from babylm 
- manual babylm results: 
Training dataset: babylm (100% of train data)
Evaluation dataset: babylm (100% of dev data)
Model: gpt2
Evaluation loss: 8.4876
Perplexity: 4854.34
Model size: 81,912,576 parameters

- agent only babylm results:
Evaluation loss: 
Perplexity: 
Model size: 

- babylm results from egd:
Training dataset: babylm (1% of train data)
Evaluation dataset: babylm (75% of dev data)
Model: gpt2
Evaluation loss: 7.6240
Perplexity: 2046.66
Model size: 64,648,273 parameters

- babylm results from egd with agent:
Evaluation loss: 
Perplexity: 
Model size: 



