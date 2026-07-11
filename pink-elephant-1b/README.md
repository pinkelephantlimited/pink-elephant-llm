# Pink Elephant:1B

A 1 billion parameter language model built from scratch using modern transformer architecture.

## Architecture

Pink Elephant:1B uses a LLaMA-like architecture with:
- **RMSNorm** (Root Mean Square Layer Normalization)
- **RoPE** (Rotary Position Embeddings)
- **SwiGLU** activation in feed-forward layers
- **Grouped-Query Attention** (GQA)
- **Pre-norm** residual architecture
- **Tied embeddings** for parameter efficiency

### Model Specifications

| Parameter | Value |
|-----------|-------|
| Parameters | ~1.1B |
| Hidden size | 2048 |
| Layers | 20 |
| Attention heads | 16 |
| Key/value heads | 16 |
| Intermediate size | 8192 (SwiGLU) |
| Max sequence length | 4096 |
| Vocabulary size | 50257 |
| Activation | SwiGLU (SiLU) |
| Position encoding | RoPE |
| Norm | RMSNorm |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train on TinyStories
python scripts/train.py --data data/tinystories.txt --epochs 1 --batch-size 2 --device cpu

# Generate text
python scripts/inference.py --prompt "Once upon a time" --max-tokens 200

# Interactive chat
python scripts/inference.py --interactive

# Download training data
python scripts/download_data.py --dataset tinystories
```

## Project Structure

```
pink-elephant-1b/
├── config/          - Configuration
├── model/           - Transformer implementation
├── tokenizer/       - BPE tokenizer
├── training/        - Training pipeline
├── inference/       - Inference engine
├── scripts/         - Training & inference scripts
└── tests/           - Unit tests
```

## Requirements

- Python 3.10+
- PyTorch 2.0+

## Running Tests

```bash
python tests/test_model.py
python tests/test_tokenizer.py
```
