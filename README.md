<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/framework-transformers-orange.svg" alt="Transformers">
</p>

<h1 align="center">🐘 Pink Elephant LLM</h1>
<p align="center"><em>Open-source language models trained entirely from scratch — no fine-tuning, no transfer learning, 100% original weights.</em></p>

---

## Overview

Pink Elephant LLM is a family of decoder-only transformer models trained from scratch on permissively licensed data. Each model uses a LLaMA-style architecture with RoPE position encoding, SwiGLU activations, and Pre-RMSNorm.

All models are trained on freely available GPUs (Google Colab T4, molab RTX Pro 6000) — demonstrating that high-quality models can be developed without massive compute budgets.

---

## Model Family

| Model | Params | Description | Status | Hardware |
|-------|--------|-------------|--------|----------|
| [Micro Coder](https://huggingface.co/pinkelephantlimited/pink-elephant-micro-coder) | **22M** | Lightweight code completion model with custom BPE tokenizer (vocab=8,192) | ✅ Released | Colab T4 |
| [33M General](https://huggingface.co/pinkelephantlimited/pink-elephant-33m) | **33M** | General-purpose language model for text generation, trained on FineWeb-Edu | ✅ Released | molab RTX Pro 6000 |
| [90M General](https://huggingface.co/pinkelephantlimited/real_model_90m) | **95M** | General-purpose language model for text generation | ✅ Released | Colab T4 |
| [12B General](https://huggingface.co/pinkelephantlimited/pink-elephant-12b) | **12.3B** | Large-scale general-purpose model with diversified training data | 🔄 Training | molab RTX Pro 6000 |

---

## Model Details

### Micro Coder (22M)

A lightweight code completion model that demonstrates meaningful code generation can be achieved with minimal compute. Trained on a T4 GPU in under 2 minutes.

- **Architecture**: LLaMA, 8 layers, 384 hidden, 12 heads
- **Tokenizer**: BPE (vocab=8,192), trained from scratch
- **Context**: 1,024 tokens
- **Training**: Code + natural language mix
- **Usage**: `pipeline("text-generation", model="pinkelephantlimited/pink-elephant-micro-coder")`

### 33M General

A compact general-purpose language model trained on FineWeb-Edu (educational web text). Designed for efficient inference on CPU and low-resource environments.

- **Architecture**: LLaMA, 8 layers, 512 hidden, 8 heads
- **Tokenizer**: BPE (vocab=4,096), trained from scratch
- **Context**: 2,048 tokens
- **Training**: 150K examples from FineWeb-Edu sample-10BT
- **Usage**: `pipeline("text-generation", model="pinkelephantlimited/pink-elephant-33m")`

### 90M General

A general-purpose language model trained on narrative text. Good for story generation and creative writing tasks.

- **Architecture**: LLaMA, 8 layers, 768 hidden, 12 heads
- **Tokenizer**: GPT-2 tokenizer (vocab=50,257)
- **Context**: 1,024 tokens
- **Usage**: `pipeline("text-generation", model="pinkelephantlimited/real_model_90m")`

---

## Training Philosophy

All Pink Elephant models are trained **from scratch** — not fine-tuned from existing models. This ensures:

1. **100% original weights** — no derivative model concerns
2. **Full control over data** — only permissively licensed sources
3. **Complete transparency** — every training step is reproducible from our notebooks

### Training Notebooks

| Model | Notebook | Hardware |
|-------|----------|----------|
| 33M General | [train_33m_general.ipynb](https://huggingface.co/pinkelephantlimited/train-micro-coder/blob/main/train_33m_general.ipynb) | molab RTX Pro 6000 |
| 12B General | [train_12b_diversified.py](https://huggingface.co/pinkelephantlimited/train-micro-coder/blob/main/train_12b_diversified.py) | molab RTX Pro 6000 |

All notebooks are available in the [train-micro-coder](https://huggingface.co/pinkelephantlimited/train-micro-coder) repository.

---

## Quick Start

```python
from transformers import pipeline

# Micro Coder — code completion
pipe = pipeline("text-generation", model="pinkelephantlimited/pink-elephant-micro-coder")
print(pipe("def fibonacci(n):", max_new_tokens=40)[0]["generated_text"])

# 33M General — text generation
pipe = pipeline("text-generation", model="pinkelephantlimited/pink-elephant-33m")
print(pipe("The future of AI is", max_new_tokens=80)[0]["generated_text"])
```

---

## License

All models and code are released under the **MIT License**.

---

<p align="center">
  <a href="https://huggingface.co/pinkelephantlimited">🤗 Hugging Face</a> •
  <a href="https://github.com/pinkelephantlimited/pink-elephant-llm">🐙 GitHub</a>
</p>
