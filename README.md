# Pink Elephant LLM

Training infrastructure and evaluation suite for the [Pink Elephant LLM](https://huggingface.co/pinkelephantlimited) model series.

## Models

| Model | Parameters | HF Hub |
|-------|-----------|--------|
| Pink Elephant LLM 1.3B | 1.3B | [Link](https://huggingface.co/pinkelephantlimited/pink-elephant-llm-1.3b) |
| Pink Elephant LLM 6.7B | 6.7B | [Link](https://huggingface.co/pinkelephantlimited/pink-elephant-llm-6.7b) |
| Pink Elephant LLM 33B | 33B | [Link](https://huggingface.co/pinkelephantlimited/pink-elephant-llm-33b) |

## Datasets

| Dataset | Examples | HF Hub |
|---------|----------|--------|
| Pink Elephant Dataset 1.3B | 500 | [Link](https://huggingface.co/datasets/pinkelephantlimited/pink-elephant-dataset-1.3b) |
| Pink Elephant Dataset 6.7B | 1,000 | [Link](https://huggingface.co/datasets/pinkelephantlimited/pink-elephant-dataset-6.7b) |
| Pink Elephant Dataset 33B | 2,000 | [Link](https://huggingface.co/datasets/pinkelephantlimited/pink-elephant-dataset-33b) |

## Repository Structure

```
├── configs/          # Training configuration files
├── scripts/          # Training and evaluation scripts
├── pe_llama/         # Custom model implementation
└── requirements.txt  # Python dependencies
```

## Training

```bash
# Install dependencies
pip install -r requirements.txt

# Launch training with DeepSpeed
torchrun --nproc_per_node=8 scripts/train.py --config configs/train_33b.json
```

## Evaluation

```bash
python scripts/evaluate.py --model pinkelephantlimited/pink-elephant-llm-33b --tasks mmlu humaneval
```

## License

Apache 2.0
