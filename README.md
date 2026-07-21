<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/framework-transformers-orange.svg" alt="Transformers">
  <img src="https://img.shields.io/badge/params-22M%20%E2%80%93%2033B-brightgreen.svg" alt="Params">
</p>

<h1 align="center">🐘 Pink Elephant LLM</h1>
<p align="center"><em>Open-source large language models trained from scratch on permissively licensed data</em></p>

---

## Table of Contents

- [Overview](#overview)
- [Model Family](#model-family)
- [Getting Started](#getting-started)
- [Detailed Model Architecture](#detailed-model-architecture)
- [Training Details](#training-details)
- [Datasets](#datasets)
- [Benchmarks](#benchmarks)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Pink Elephant LLM is a family of open-source transformer-based language models developed by Pink Elephant Limited. The models range from a lightweight 22M parameter code completion model to a large-scale 33B general-purpose language model. All models are trained from scratch using publicly available, permissively licensed datasets.

| Feature | Description |
|---------|-------------|
| **Architecture** | Decoder-only transformer (LLaMA-style) |
| **Position encoding** | Rotary Position Embeddings (RoPE) |
| **Activation** | SiLU (Swish) |
| **Normalization** | RMSNorm |
| **Framework** | Hugging Face Transformers |
| **License** | MIT |

---

## Model Family

| Model | Params | Hidden | Layers | Heads | LR | Tokens | Type |
|-------|--------|--------|--------|-------|----|--------|------|
| [Micro Coder](https://huggingface.co/pinkelephantlimited/pink-elephant-micro-coder) | 22M | 384 | 8 | 12 | 3e-4 | 26M | Code completion |
| [Legal 100M](https://huggingface.co/pinkelephantlimited/pink-elephant-legal-100m) | 126M | 768 | 12 | 12 | 3e-4 | — | Legal text (training) |
| [1.3B](https://huggingface.co/pinkelephantlimited/pink-elephant-llm-1.3b) | 1.3B | 2048 | 24 | 16 | — | — | General purpose |
| [6.7B](https://huggingface.co/pinkelephantlimited/pink-elephant-llm-6.7b) | 6.7B | 4096 | 32 | 32 | — | — | General purpose |
| [33B](https://huggingface.co/pinkelephantlimited/pink-elephant-llm-33b) | 33B | 7168 | 48 | 56 | — | — | General purpose |

---

## Getting Started

### Installation

```bash
pip install transformers torch
```

### Code Completion (Micro Coder)

```python
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="pinkelephantlimited/pink-elephant-micro-coder",
    device_map="auto"
)

prompt = "def fibonacci(n):\n    if n <= 1:"
result = pipe(prompt, max_new_tokens=40, temperature=0.5, do_sample=True)[0]["generated_text"]
print(result)
```

### Text Generation (1.3B)

```python
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="pinkelephantlimited/pink-elephant-llm-1.3b",
    device_map="auto"
)

result = pipe("The future of AI is", max_new_tokens=100, temperature=0.7)[0]["generated_text"]
print(result)
```

---

## Detailed Model Architecture

All models follow the LLaMA decoder-only architecture with the following components:

### Core Design

- **Decoder-only transformer**: Autoregressive language modeling with causal attention masking
- **Pre-normalization**: RMSNorm applied before each sublayer (Pre-LN)
- **Residual connections**: Standard skip connections around attention and FFN sublayers
- **Activation function**: SiLU (also known as Swish) for the feed-forward network
- **Position encoding**: Rotary Position Embeddings (RoPE) applied to query and key vectors
- **Weight tying**: Embedding and LM head weights are tied where applicable

### Feed-Forward Network

Each transformer layer uses a SwiGLU variant:

```
FFN(x) = (SiLU(xW_gate) ⊙ xW_up) W_down
```

where ⊙ denotes element-wise multiplication.

### Micro Coder Specifics

| Component | Configuration |
|-----------|--------------|
| Vocab size | 8,192 (BPE tokenizer) |
| Hidden dim | 384 |
| Intermediate dim | 1,536 |
| Num layers | 8 |
| Num attention heads | 12 |
| Num KV heads | 12 (multi-head attention) |
| Head dim | 32 |
| Max seq length | 1,024 |
| RoPE theta | 10,000.0 |
| Dropout | 0.0 |
| Bias | False (all linear layers) |

---

## Training Details

### Micro Coder Training

The Micro Coder was trained on a single **NVIDIA T4 GPU (16GB VRAM)** via **Google Colab free tier** — demonstrating that meaningful models can be developed without dedicated hardware.

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW (β₁=0.9, β₂=0.999, ε=1e-8) |
| Peak learning rate | 3 × 10⁻⁴ |
| LR schedule | Linear warmup + cosine decay |
| Warmup steps | 200 |
| Weight decay | 0.01 |
| Per-device batch size | 8 |
| Gradient accumulation | 4 |
| Effective batch size | 32 |
| Gradient clipping | 1.0 |
| Precision | FP16 (mixed precision) |
| Training epochs | 20 |
| Training steps | ~875 |
| Training time | ~87 seconds |

### Tokenizer

A Byte-Pair Encoding (BPE) tokenizer was trained from scratch on the combined training corpus:

- **Algorithm**: BPE with byte-level preprocessing
- **Vocab size**: 8,192 tokens
- **Special tokens**: `<unk>`, `<s>` (BOS), `</s>` (EOS), `<pad>`, `<mask>`
- **Pre-tokenizer**: ByteLevel (adds prefix space = false)

---

## Datasets

The Micro Coder model was trained on the following open-access datasets:

| Dataset | Type | Examples | License | Source |
|---------|------|----------|---------|--------|
| [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval) | Python code | 164 problems | MIT | OpenAI |
| [Lambada (OpenAI)](https://huggingface.co/datasets/EleutherAI/lambada_openai) | Narrative text | ~5,000 passages | Public domain | EleutherAI |

**Data processing pipeline:**

1. Code solutions from HumanEval are formatted as prompt + canonical solution pairs
2. Lambada passages are used as-is for language modeling
3. All texts are tokenized by the BPE tokenizer (max length 512, truncation)
4. Sequences under 10 tokens are filtered out
5. The combined dataset yields approximately 50,000 training sequences

---

## Benchmarks

*Coming soon — benchmarks will be added as evaluation is completed.*

---

## Installation

### From PyPI

```bash
pip install transformers torch
```

### From source

```bash
git clone https://github.com/pinkelephantlimited/pink-elephant-llm.git
cd pink-elephant-llm
pip install -r requirements.txt
```

---

## Contributing

We welcome contributions! Please open an issue or submit a pull request on [GitHub](https://github.com/pinkelephantlimited/pink-elephant-llm).

---

## License

All models and code in this repository are released under the **MIT License**. See the `LICENSE` file in each model repository for details.

```
MIT License

Copyright (c) 2026 Pink Elephant Limited

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<p align="center">
  <a href="https://huggingface.co/pinkelephantlimited">🤗 Hugging Face</a> •
  <a href="https://github.com/pinkelephantlimited/pink-elephant-llm">🐙 GitHub</a>
</p>
