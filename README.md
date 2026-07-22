<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/framework-transformers-orange.svg" alt="Transformers">
</p>

<h1 align="center">🐘 Pink Elephant LLM</h1>
<p align="center"><em>Open-source language models trained from scratch on permissively licensed data</em></p>

---

## Model Family

| Model | Params | Hidden | Layers | Heads | Type |
|-------|--------|--------|--------|-------|------|
| [Micro Coder](https://huggingface.co/pinkelephantlimited/pink-elephant-micro-coder) | 22M | 384 | 8 | 12 | Code completion |
| [33M General](https://huggingface.co/pinkelephantlimited/pink-elephant-33m) | 33M | 512 | 8 | 8 | General purpose |
| [90M General](https://huggingface.co/pinkelephantlimited/real_model_90m) | 95M | 768 | 8 | 12 | General purpose |
| [Legal 100M](https://huggingface.co/pinkelephantlimited/pink-elephant-legal-100m) | 126M | 768 | 12 | 12 | Legal text |
| [Legal/Finance 500M](https://huggingface.co/pinkelephantlimited/pink-elephant-legalfinance-500m) | 528M | 1152 | 24 | 18 | Legal, finance, accounting |
| [12B General](https://huggingface.co/pinkelephantlimited/pink-elephant-12b) | 12.3B | 4608 | 36 | 32 | General purpose (training) |

---

## Getting Started

### Micro Coder

```python
from transformers import pipeline
pipe = pipeline("text-generation", model="pinkelephantlimited/pink-elephant-micro-coder", device_map="auto")
print(pipe("def fibonacci(n):\n    if n <= 1:", max_new_tokens=40, temperature=0.5)[0]["generated_text"])
```

### Legal / Finance / Accounting (500M)

```python
from transformers import pipeline
pipe = pipeline("text-generation", model="pinkelephantlimited/pink-elephant-legalfinance-500m", device_map="auto")
print(pipe("In accordance with IFRS standards,", max_new_tokens=100, temperature=0.5)[0]["generated_text"])
```

---

## Training Notebooks

| Model | Notebook |
|-------|----------|
| 33M General | [train_33m_general.ipynb](https://huggingface.co/pinkelephantlimited/train-micro-coder/blob/main/train_33m_general.ipynb) |
| 500M Legal/Finance | [train_500m_legalfinance.ipynb](https://huggingface.co/pinkelephantlimited/train-micro-coder/blob/main/train_500m_legalfinance.ipynb) |
| 12B General | [train_12b_diversified.py](https://huggingface.co/pinkelephantlimited/train-micro-coder/blob/main/train_12b_diversified.py) |
| 100M Legal | [train_legal_100m.ipynb](https://huggingface.co/pinkelephantlimited/train-micro-coder/blob/main/train_legal_100m.ipynb) |

---

## License

All models and code are released under the **MIT License**.

---

<p align="center">
  <a href="https://huggingface.co/pinkelephantlimited">🤗 Hugging Face</a> •
  <a href="https://github.com/pinkelephantlimited/pink-elephant-llm">🐙 GitHub</a>
</p>
