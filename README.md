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
| [22M](https://huggingface.co/pinkelephantlimited/pink-elephant-22m) | **22M** | Code completion + text generation (HumanEval + LAMBADA) | ✅ Released | Colab T4 |
| [33M](https://huggingface.co/pinkelephantlimited/pink-elephant-33m) | **33M** | General-purpose text generation (FineWeb-Edu) | ✅ Released | molab RTX Pro 6000 |
| [90M](https://huggingface.co/pinkelephantlimited/pink-elephant-90m) | **89M** | General-purpose text generation with GQA (FineWeb-Edu) | ✅ Released | Colab T4 |
| [1B](https://huggingface.co/pinkelephantlimited/pink-elephant-1b) | **1.1B** | Multi-domain text gen (7 datasets: web, math, code, legal, finance) | 🔄 Training | molab RTX Pro 6000 |
| [10B](https://huggingface.co/pinkelephantlimited/pink-elephant-10b) | **9.85B** | Multi-domain text gen (7 datasets: web, math, code, legal, finance) | ⏳ Ready | molab RTX Pro 6000 |

---

## Model Details

### 22M — Code Completion & Text Generation

A lightweight model demonstrating meaningful language generation with minimal compute. Trained on a T4 GPU in ~2 minutes.

- **Architecture**: LLaMA, 8 layers, 384 hidden, 12 heads
- **Tokenizer**: BPE (vocab=8,192), trained from scratch
- **Context**: 1,024 tokens
- **Data**: HumanEval (code) + LAMBADA (narrative text)
- **Usage**: `pipeline("text-generation", model="pinkelephantlimited/pink-elephant-22m")`

### 33M — General-Purpose

A compact model optimized for CPU and low-resource environments.

- **Architecture**: LLaMA, 8 layers, 512 hidden, 8 heads
- **Tokenizer**: BPE (vocab=4,096), trained from scratch
- **Context**: 2,048 tokens
- **Data**: 150K examples from FineWeb-Edu sample-10BT
- **Usage**: `pipeline("text-generation", model="pinkelephantlimited/pink-elephant-33m")`

### 90M — General-Purpose with GQA

A general-purpose model with Grouped-Query Attention (4 KV heads, 12 query heads) for efficient inference.

- **Architecture**: LLaMA with GQA, 8 layers, 768 hidden, 12 heads, 4 KV heads
- **Tokenizer**: GPT-2 (vocab=50,257)
- **Context**: 1,024 tokens
- **Data**: 150K examples from FineWeb-Edu sample-10BT
- **Usage**: `pipeline("text-generation", model="pinkelephantlimited/pink-elephant-90m")`

### 1B — Multi-Domain General-Purpose

A diverse model trained on 7 verified datasets spanning web text, mathematics, books, code, legal, and finance.

- **Architecture**: LLaMA, 16 layers, 2,048 hidden, 16 heads
- **Tokenizer**: BPE (vocab=16,384), trained from scratch
- **Context**: 2,048 tokens
- **Data**: FineWeb-Edu + FineWeb + OpenWebMath + SmolLM cosmopedia-v2 + CodeParrot + Nemotron-Legal + Investopedia (~410K examples)
- **Training**: Batch 256, bf16, 8-bit Adam, checkpoints every 1K steps
- **Usage**: `pipeline("text-generation", model="pinkelephantlimited/pink-elephant-1b")`

### 10B — Large-Scale General-Purpose (9.85B params)

The largest Pink Elephant model — 31 layers, 5120 hidden, 4096 context. Trained on 7 diverse datasets.

- **Architecture**: LLaMA, 31 layers, 5,120 hidden, 40 heads
- **Tokenizer**: BPE (vocab=4,096), trained from scratch
- **Context**: 4,096 tokens
- **Data**: Same 7-dataset mix (~410K examples)
- **Training Config**: Batch 4, grad accum 8, bf16, 8-bit Adam, VRAM ~75 GB
- **Usage**: `pipeline("text-generation", model="pinkelephantlimited/pink-elephant-10b")`

---

## Training Philosophy

All Pink Elephant models are trained **from scratch** — not fine-tuned from existing models. This ensures:

1. **100% original weights** — no derivative model concerns
2. **Full control over data** — only permissively licensed sources
3. **Complete transparency** — every training step is reproducible from our notebooks

### Training Notebooks

| Model | Notebook | Hardware |
|-------|----------|----------|
| 22M | [train_micro_coder.ipynb](https://huggingface.co/pinkelephantlimited/pink-elephant-22m/blob/main/train_micro_coder.ipynb) | Colab T4 |
| 33M | [train_33m_general.ipynb](https://huggingface.co/pinkelephantlimited/pink-elephant-33m/blob/main/train_33m_general.ipynb) | molab RTX Pro 6000 |
| 90M | [train_90m_general.ipynb](https://huggingface.co/pinkelephantlimited/pink-elephant-90m/blob/main/train_90m_general.ipynb) | Colab T4 |
| 1B | [train_1b_general.py](https://github.com/pinkelephantlimited/pink-elephant-llm/blob/master/train_1b_general.py) | molab RTX Pro 6000 |
| 10B | [train_10b_general.py](https://github.com/pinkelephantlimited/pink-elephant-llm/blob/master/train_10b_general.py) | molab RTX Pro 6000 |

All scripts are in this GitHub repository.

---

## Quick Start

```python
from transformers import pipeline

# 22M — code completion
pipe = pipeline("text-generation", model="pinkelephantlimited/pink-elephant-22m")
print(pipe("def fibonacci(n):", max_new_tokens=40)[0]["generated_text"])

# 33M — general text generation
pipe = pipeline("text-generation", model="pinkelephantlimited/pink-elephant-33m")
print(pipe("The future of AI is", max_new_tokens=80)[0]["generated_text"])

# 90M — general text generation
pipe = pipeline("text-generation", model="pinkelephantlimited/pink-elephant-90m")
print(pipe("Once upon a time,", max_new_tokens=80)[0]["generated_text"])

# 1B — multi-domain text generation
pipe = pipeline("text-generation", model="pinkelephantlimited/pink-elephant-1b")
print(pipe("The definition of artificial intelligence is", max_new_tokens=80)[0]["generated_text"])
```

---

## Real-World Applications

### 22M — Embedded Code Autocomplete

Runs on Raspberry Pi, WebAssembly browsers, or IoT devices. Sub-100ms on CPU.

```python
pipe("def fibonacci(n):", max_new_tokens=40)
pipe("import pandas as pd\ndf = pd.read_csv('data.csv')\ndf.", max_new_tokens=40)
```

### 33M — Edge Text Generation

Fits in 200MB RAM, runs on $5/month VPS. Educational Q&A, note autocomplete.

```python
pipe("The water cycle consists of", max_new_tokens=80)
pipe("Photosynthesis is the process by which", max_new_tokens=80)
```

### 90M — Mobile Content Drafting

GQA provides 3x faster inference on phone GPUs. Social media drafts, blog intros.

```python
pipe("5 tips for better sleep: 1.", max_new_tokens=100)
pipe("The rise of remote work has fundamentally", max_new_tokens=100)
```

### 1B — Multi-Domain Assistant

Customer support, legal clause drafting, finance summarization, code gen — all in one model.

```python
pipe("According to our return policy,", max_new_tokens=80)
pipe("In accordance with IFRS standards,", max_new_tokens=80)
pipe("The auditor shall review all", max_new_tokens=80)
pipe("def train_model(model, data, epochs):", max_new_tokens=80)
```

### 10B — Enterprise Document Intelligence

Largest in the family. Handles multi-paragraph legal/finance/code generation.

```python
pipe("The court finds that the defendant's actions", max_new_tokens=120)
pipe("Based on the audited financial statements,", max_new_tokens=120)
pipe("class DatabaseConnection:", max_new_tokens=120)
```

---

## License

All models and code are released under the **MIT License**.

---

<p align="center">
  <a href="https://huggingface.co/pinkelephantlimited">🤗 Hugging Face</a> •
  <a href="https://github.com/pinkelephantlimited/pink-elephant-llm">🐙 GitHub</a>
</p>
