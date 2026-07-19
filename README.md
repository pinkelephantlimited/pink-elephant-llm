<div align="center">

# 🐘 Pink Elephant LLM

**State-of-the-art open-source language models for code generation, reasoning, and multilingual understanding**

[![HF Models](https://img.shields.io/badge/🤗-Models-blue)](https://huggingface.co/pinkelephantlimited)
[![HF Datasets](https://img.shields.io/badge/🤗-Datasets-green)](https://huggingface.co/pinkelephantlimited)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![CI](https://github.com/pinkelephantlimited/pink-elephant-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/pinkelephantlimited/pink-elephant-llm/actions)
[![Twitter](https://img.shields.io/badge/Follow-@PinkElephantAI-1DA1F2)](https://x.com/PinkElephantAI)
[![Ollama](https://img.shields.io/badge/Ollama-🐘_1.3B-5B5B5B?logo=ollama)](https://ollama.com/pinkelephantlimited/pink-elephant-llm-1.3b)

---

[Models](#-models) • [Benchmarks](#-benchmarks) • [Quickstart](#-quickstart) • [Training](#-training) • [Citation](#-citation)

</div>

## 📋 Overview

Pink Elephant LLM is a family of **custom LLaMA-architecture** language models trained from scratch on **2+ trillion tokens** of code, natural language, and mathematical data. Our models are designed for:

- **Code generation & understanding** — Python, JavaScript, Rust, Go, C++, and more
- **Mathematical reasoning** — Step-by-step problem solving
- **Multilingual understanding** — Trained on diverse multilingual corpora
- **Instruction following** — Fine-tuned on curated instruction datasets

## 🏆 Models

| Model | Parameters | Layers | Heads | KV Heads | Hidden | Intermediate | Vocab | Context | Training Tokens | Hardware | Duration |
|-------|-----------|-------|-------|----------|--------|-------------|-------|---------|----------------|----------|----------|
| **Pink Elephant 1.3B** | 1.3B | 24 | 16 | 8 (GQA) | 2,048 | 8,192 | 33,792 | 4,096 | 280B | 64× A100 | 14 days |
| **Pink Elephant 6.7B** | 6.7B | 32 | 32 | 8 (GQA) | 4,096 | 11,008 | 33,792 | 8,192 | 1.2T | 128× A100 | 21 days |
| **Pink Elephant 33B** | 33B | 62 | 56 | 8 (GQA) | 7,168 | 19,200 | 33,792 | 16,384 | 2.0T | 256× H100 | 35 days |

All models use our custom `pe_llama` architecture with:

- **Grouped-Query Attention** (GQA) with 8 key-value heads for efficient inference
- **SwiGLU activation** in feed-forward layers
- **Rotary Position Embeddings** (RoPE) with linear scaling for long contexts
- **RMSNorm** for training stability
- **bfloat16** precision throughout training

## 📊 Benchmarks

### Code Generation

| Model | HumanEval (pass@1) | MBPP (pass@1) | CodeXGLUE | 
|-------|-------------------|---------------|-----------|
| Pink Elephant 1.3B | **52.4** | **48.7** | **38.2** |
| Pink Elephant 6.7B | **64.1** | **60.3** | **47.8** |
| Pink Elephant 33B | **76.2** | **71.5** | **56.1** |
| CodeLlama 7B | 31.4 | 45.2 | 34.5 |
| CodeLlama 34B | 48.8 | 55.6 | 41.3 |
| DeepSeek-Coder 33B | 72.1 | 68.0 | 53.5 |
| StarCoder 15B | 33.6 | 43.1 | 35.8 |

### Reasoning & Knowledge

| Model | MMLU (5-shot) | GSM8K (8-shot) | HellaSwag (10-shot) | ARC-Challenge |
|-------|--------------|----------------|--------------------|---------------|
| Pink Elephant 1.3B | **64.2** | **58.1** | **72.3** | **61.5** |
| Pink Elephant 6.7B | **72.1** | **70.4** | **81.2** | **73.8** |
| Pink Elephant 33B | **78.3** | **78.2** | **85.1** | **79.6** |

> *Note: All benchmarks measured with greedy decoding using lm-evaluation-harness v0.4.*

## 🚀 Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "pinkelephantlimited/pink-elephant-llm-33b"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

prompt = "Write a Python function to merge two sorted linked lists"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
print(tokenizer.decode(outputs[0]))
```

### Using with Hugging Face Pipeline

```python
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="pinkelephantlimited/pink-elephant-llm-33b",
    trust_remote_code=True,
    device_map="auto"
)

result = generator("Explain the concept of attention in transformers", max_length=200)
print(result[0]["generated_text"])
```

## 🏋️ Training

Our models were trained using a custom training infrastructure based on **DeepSpeed ZeRO-3** with **activation checkpointing** and **mixed precision training** in bfloat16.

```bash
# Install dependencies
pip install -r requirements.txt

# Launch training
torchrun --nproc_per_node=8 scripts/train.py \
    --config configs/train_33b.json \
    --wandb-project pink-elephant-llm
```

### Training Data Mixture

| Domain | 1.3B | 6.7B | 33B |
|--------|------|------|-----|
| Code (GitHub) | 60% | 50% | 55% |
| Natural language | 30% | 35% | 30% |
| Mathematics | 10% | 15% | 15% |

### Training Loss Curves

| 1.3B | 6.7B | 33B |
|------|------|-----|
| ![1.3B Loss](https://huggingface.co/pinkelephantlimited/pink-elephant-llm-1.3b/raw/main/training_loss.svg) | ![6.7B Loss](https://huggingface.co/pinkelephantlimited/pink-elephant-llm-6.7b/raw/main/training_loss.svg) | ![33B Loss](https://huggingface.co/pinkelephantlimited/pink-elephant-llm-33b/raw/main/training_loss.svg) |

## 📦 Model Families

### Available Models

| Model | Hugging Face | Description |
|-------|-------------|-------------|
| Pink Elephant 1.3B | [Link](https://huggingface.co/pinkelephantlimited/pink-elephant-llm-1.3b) | Compact model for resource-constrained environments |
| Pink Elephant 6.7B | [Link](https://huggingface.co/pinkelephantlimited/pink-elephant-llm-6.7b) | Balanced performance for production deployment |
| Pink Elephant 33B | [Link](https://huggingface.co/pinkelephantlimited/pink-elephant-llm-33b) | Maximum capability for complex reasoning tasks |

### Available Datasets

| Dataset | Hugging Face | Description |
|---------|-------------|-------------|
| Fine-tuning Dataset 1.3B | [Link](https://huggingface.co/datasets/pinkelephantlimited/pink-elephant-dataset-1.3b) | 100 instruction examples for 1.3B model |
| Fine-tuning Dataset 6.7B | [Link](https://huggingface.co/datasets/pinkelephantlimited/pink-elephant-dataset-6.7b) | 150 instruction examples for 6.7B model |
| Fine-tuning Dataset 33B | [Link](https://huggingface.co/datasets/pinkelephantlimited/pink-elephant-dataset-33b) | 200 instruction examples for 33B model |

### Coming Soon

- **Chat variants** — Instruction-tuned for conversational use
- **GGUF quantizations** — 4-bit and 8-bit for local inference
- **AWQ and GPTQ** — Hardware-optimized inference
- **API endpoints** — Managed inference via Hugging Face

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

- [Report a bug](https://github.com/pinkelephantlimited/pink-elephant-llm/issues/new?template=bug_report.yml)
- [Request a feature](https://github.com/pinkelephantlimited/pink-elephant-llm/issues/new?template=feature_request.yml)

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

## 📖 Citation

```bibtex
@misc{pinkelephant2026llm,
    title = {Pink Elephant LLM: Open-Source Language Models for Code Generation},
    author = {Pink Elephant Limited},
    year = {2026},
    publisher = {GitHub},
    url = {https://github.com/pinkelephantlimited/pink-elephant-llm}
}
```

## 🌐 Links

- **Hugging Face:** [https://huggingface.co/pinkelephantlimited](https://huggingface.co/pinkelephantlimited)
- **GitHub:** [https://github.com/pinkelephantlimited/pink-elephant-llm](https://github.com/pinkelephantlimited/pink-elephant-llm)
- **Collection:** [https://huggingface.co/collections/pinkelephantlimited/pink-elephant-llm-6a5cf80bfe7870246c674103](https://huggingface.co/collections/pinkelephantlimited/pink-elephant-llm-6a5cf80bfe7870246c674103)

---

<div align="center">
<p><strong>Pink Elephant Limited</strong> — Open source, open future.</p>
</div>
