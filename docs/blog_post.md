---
title: "Introducing Pink Elephant LLM: Open-Source Language Models for Code Generation"
published: false
tags: [ai, opensource, llm, code-generation]
---

# Introducing Pink Elephant LLM: Open-Source Language Models for Code Generation

Today we're thrilled to announce the Pink Elephant LLM family — a collection of open-source language models purpose-built for code generation, now available under the MIT license. We believe that powerful code generation models should be accessible to everyone: from independent developers experimenting with new ideas to enterprises building production-grade tooling. Pink Elephant delivers state-of-the-art performance across code, natural language, and mathematical reasoning tasks, and we can't wait to see what the community builds with it.

## The Model Family

Pink Elephant comes in three sizes, designed to fit a range of deployment scenarios:

| Model | Parameters | Use Case |
|-------|-----------|----------|
| Pink Elephant 1.3B | 1.3 billion | Lightweight inference, edge devices, rapid prototyping |
| Pink Elephant 6.7B | 6.7 billion | Local development, fine-tuning on commodity hardware |
| Pink Elephant 33B | 33 billion | Highest accuracy, server-scale deployments |

All three models share a unified decoder-only transformer architecture with modern design choices:

- **Grouped-Query Attention (GQA)** — Reduces memory bandwidth requirements during inference with minimal quality loss, enabling faster generation on both GPUs and CPUs.
- **SwiGLU Activation** — Improves training stability and downstream task performance compared to standard ReLU or GeLU activations.
- **Rotary Position Embeddings (RoPE)** — Provides relative position encoding that generalizes well to sequence lengths beyond those seen during training.

These architectural decisions together yield models that are both performant and practical to deploy.

## Training Data

Pink Elephant was trained on a carefully curated mixture spanning three domains:

- **Code** (~60%) — A diverse corpus of permissively licensed source code from GitHub, including Python, JavaScript, TypeScript, Rust, Go, C++, Java, and many more.
- **Natural Language** (~25%) — High-quality web text, technical documentation, tutorials, and stack exchange data.
- **Math & Reasoning** (~15%) — Mathematical problem sets, theorem proofs, and step-by-step reasoning chains.

All data was deduplicated using MinHash, filtered for quality heuristics, and tokenized with a custom BPE tokenizer trained for code-heavy workloads.

## Training Infrastructure

Training the Pink Elephant family required substantial compute:

- **33B model** — Trained on 256 NVIDIA H100 80GB GPUs using DeepSpeed ZeRO-3, completing 2.0 trillion tokens over approximately 35 days.
- **6.7B model** — Trained on 128 NVIDIA A100 80GB GPUs, completing 1.2 trillion tokens.
- **1.3B model** — Trained on 64 NVIDIA A100 80GB GPUs, completing 280 billion tokens.

Mixed precision training (bf16) and activation checkpointing kept memory usage manageable while maintaining throughput. Our training pipeline achieved approximately 42% Model Flops Utilization (MFU) on H100 clusters.

## Benchmark Results

We evaluated Pink Elephant against several leading open-source code models:

| Model | HumanEval | MMLU (5-shot) | GSM8K (8-shot) |
|-------|-----------|---------------|----------------|
| Pink Elephant 1.3B | 52.4% | 64.2% | 58.1% |
| Pink Elephant 6.7B | 64.1% | 72.1% | 70.4% |
| **Pink Elephant 33B** | **76.2%** | **78.3%** | **78.2%** |
| CodeLlama 34B | 73.8% | 75.4% | 72.6% |
| DeepSeek-Coder 33B | 75.4% | 76.9% | 76.1% |

The 33B Pink Elephant model sets a new bar for open-weight code LLMs, outperforming both CodeLlama 34B and DeepSeek-Coder 33B across all three evaluations.

## Open-Source Commitment

Pink Elephant is released under the **MIT license** — no restrictions on use, no royalty fees, no commercial limitations.

- **Model weights and tokenizer**: Available on [Hugging Face](https://huggingface.co/pinkelephantlimited)
- **Training and inference code**: Available on [GitHub](https://github.com/pinkelephantlimited/pink-elephant-llm)
- **Documentation and examples**: Included in the repository to help you get started in minutes

## Future Plans

- **Chat Variants** — Instruction-tuned and RLHF-aligned versions for conversational coding assistants.
- **GGUF Quantizations** — Community-friendly quantized models for running on consumer hardware.
- **API Endpoints** — Managed inference through a simple REST API.
- **Extended Context** — Longer-context versions supporting 32K+ tokens.

We're actively listening to the community. If there's a size, variant, or feature you'd like to see, open an issue on our GitHub.

## Get Started

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "pinkelephantlimited/pink-elephant-llm-33b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True, device_map="auto"
)

prompt = "Write a Python function to merge two sorted linked lists"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

We encourage you to:
- **Try the models** on [Hugging Face](https://huggingface.co/pinkelephantlimited)
- **Star the repo** on [GitHub](https://github.com/pinkelephantlimited/pink-elephant-llm)
- **Share your feedback** — your input shapes our roadmap

Let's make code generation open for everyone.
