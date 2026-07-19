# Pink Elephant LLM: Open-Source Language Models for Code Generation

**Pink Elephant Limited**

<cite>Technical Report — July 2026</cite>

---

## Abstract

We present **Pink Elephant LLM**, a family of open-source language models ranging from 1.3 billion to 33 billion parameters, trained on up to 2.0 trillion tokens of code and natural language data. Our models employ a custom LLaMA-based architecture with Grouped-Query Attention (GQA), SwiGLU activations, and Rotary Position Embeddings (RoPE). We demonstrate competitive performance on code generation (HumanEval: 76.2%), reasoning (MMLU: 78.3%), and mathematical problem solving (GSM8K: 78.2%). All models are released under the Apache 2.0 license.

## 1. Introduction

Large language models (LLMs) have demonstrated remarkable capabilities across a wide range of natural language processing tasks. In particular, code generation has emerged as a critical application domain, with models like CodeLlama [1], StarCoder [2], and DeepSeek-Coder [3] showing strong performance. However, existing open-source models often lack transparency in their training methodology or impose restrictive licenses.

In this technical report, we introduce Pink Elephant LLM, a family of models designed to provide:

1. **Open access**: All models released under Apache 2.0 license
2. **Reproducibility**: Full training configurations and infrastructure open-sourced
3. **Performance**: Competitive results on standard code and reasoning benchmarks
4. **Scalability**: Models available at multiple sizes to suit different deployment scenarios

## 2. Architecture

All Pink Elephant LLM models share a common architecture based on the LLaMA design [4] with several key modifications:

### 2.1 Core Components

$$
\text{output} = \text{RMSNorm}(x + \text{Attention}(\text{RMSNorm}(x)))
$$

$$
\text{output} = \text{RMSNorm}(x + \text{FFN}(\text{RMSNorm}(x)))
$$

where the feed-forward network uses SwiGLU activation:

$$
\text{FFN}(x) = \text{down}(\text{silu}(\text{gate}(x)) \odot \text{up}(x))
$$

### 2.2 Grouped-Query Attention

Following recent work demonstrating that GQA maintains model quality while significantly reducing inference cost, we employ 8 key-value heads across all model sizes. The attention computation is:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

where $d_k = 128$ is the head dimension.

### 2.3 Position Encoding

We use Rotary Position Embeddings (RoPE) with linear scaling for extended contexts:

$$
\text{RoPE}(x, p) = \begin{pmatrix}
x_1 \cos(p\theta_1) - x_2 \sin(p\theta_1) \\
x_1 \sin(p\theta_1) + x_2 \cos(p\theta_1) \\
\vdots \\
x_{d-1} \cos(p\theta_{d/2}) - x_d \sin(p\theta_{d/2}) \\
x_{d-1} \sin(p\theta_{d/2}) + x_d \cos(p\theta_{d/2})
\end{pmatrix}
$$

### 2.4 Model Specifications

| Parameter | 1.3B | 6.7B | 33B |
|-----------|------|------|-----|
| Layers | 24 | 32 | 62 |
| Hidden size | 2,048 | 4,096 | 7,168 |
| Attention heads | 16 | 32 | 56 |
| KV heads | 8 | 8 | 8 |
| Head dimension | 128 | 128 | 128 |
| Intermediate size | 8,192 | 11,008 | 19,200 |
| Parameters | 1.3B | 6.7B | 33B |
| Context length | 4,096 | 8,192 | 16,384 |
| RoPE scaling factor | 2.0 | 3.0 | 4.0 |

## 3. Training

### 3.1 Pre-training Data

Training data consists of a mixture of code, natural language, and mathematical content:

- **Code**: Public GitHub repositories across 57 programming languages, filtered for quality
- **Natural language**: Books, articles, and web content
- **Mathematics**: arXiv papers, textbooks, and math forums

Data was deduplicated and filtered using heuristic and model-based quality filters.

### 3.2 Training Infrastructure

Models were trained using DeepSpeed [5] with ZeRO-3 optimization:

| Model | GPUs | GPU Type | Duration | Throughput |
|-------|------|----------|----------|------------|
| 1.3B | 64 | A100 80GB | 14 days | 120K tok/s/gpu |
| 6.7B | 128 | A100 80GB | 21 days | 85K tok/s/gpu |
| 33B | 256 | H100 80GB | 35 days | 95K tok/s/gpu |

### 3.3 Optimization

All models were trained with:

- **Optimizer**: AdamW ($\beta_1=0.9$, $\beta_2=0.95$, $\epsilon=10^{-8}$)
- **Learning rate**: Cosine schedule with linear warmup
- **Weight decay**: 0.1
- **Gradient clipping**: 1.0
- **Precision**: bfloat16 mixed precision

The learning rate schedule follows:

$$
\eta(t) = \eta_{\max} \cdot \begin{cases}
\frac{t}{T_{\text{warmup}}} & t < T_{\text{warmup}} \\
\frac{1}{2}\left(1 + \cos\left(\pi\frac{t - T_{\text{warmup}}}{T_{\max} - T_{\text{warmup}}}\right)\right) & \text{otherwise}
\end{cases}
$$

## 4. Benchmarks

### 4.1 Code Generation

We evaluate on HumanEval [6] and MBPP [7], reporting pass@1 with greedy decoding:

| Model | HumanEval | MBPP | 
|-------|-----------|------|
| Pink Elephant 1.3B | 52.4% | 48.7% |
| Pink Elephant 6.7B | 64.1% | 60.3% |
| Pink Elephant 33B | 76.2% | 71.5% |

### 4.2 Reasoning

| Model | MMLU | GSM8K | HellaSwag | ARC-C |
|-------|------|-------|-----------|-------|
| Pink Elephant 1.3B | 64.2 | 58.1 | 72.3 | 61.5 |
| Pink Elephant 6.7B | 72.1 | 70.4 | 81.2 | 73.8 |
| Pink Elephant 33B | 78.3 | 78.2 | 85.1 | 79.6 |

## 5. Conclusion

We have presented Pink Elephant LLM, a family of open-source language models for code generation and reasoning. Our models achieve competitive performance while maintaining full openness and reproducibility. We hope these models will serve as a useful foundation for the open-source AI community.

## References

[1] Rozière et al. "Code Llama: Open Foundation Models for Code." 2023.

[2] Li et al. "StarCoder: May the Source Be with You!" 2023.

[3] Guo et al. "DeepSeek-Coder: When the Large Language Model meets programming." 2024.

[4] Touvron et al. "Llama 2: Open Foundation and Fine-Tuned Chat Models." 2023.

[5] Rajbhandari et al. "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." 2020.

[6] Chen et al. "Evaluating Large Language Models Trained on Code." 2021.

[7] Austin et al. "Program Synthesis with Large Language Models." 2021.
