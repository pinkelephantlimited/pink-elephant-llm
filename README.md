# Pink Elephant LLM

Open-source large language models by Pink Elephant Limited. All models are trained from scratch using open datasets and permissively licensed.

## Models

| Model | Parameters | Description | Training Data |
|-------|-----------|-------------|---------------|
| [Pink Elephant 33B](https://huggingface.co/pinkelephantlimited/pink-elephant-llm-33b) | 33B | Large-scale general-purpose LLM | — |
| [Pink Elephant 6.7B](https://huggingface.co/pinkelephantlimited/pink-elephant-llm-6.7b) | 6.7B | Mid-size model for broader capability | — |
| [Pink Elephant 1.3B](https://huggingface.co/pinkelephantlimited/pink-elephant-llm-1.3b) | 1.3B | Lightweight general-purpose LLM | — |
| [Pink Elephant Micro Coder](https://huggingface.co/pinkelephantlimited/pink-elephant-micro-coder) | 22M | Code completion model trained on free Colab T4 | HumanEval + Lambada |

## Usage

All models can be used via Hugging Face `transformers`:

```python
from transformers import pipeline

# For code completion:
pipe = pipeline("text-generation", model="pinkelephantlimited/pink-elephant-micro-coder")
result = pipe("def hello():\n    print", max_new_tokens=30)[0]["generated_text"]

# For general text:
pipe = pipeline("text-generation", model="pinkelephantlimited/pink-elephant-llm-1.3b")
result = pipe("Hello, I'm", max_new_tokens=50)[0]["generated_text"]
```

## Training Data

The **Micro Coder** model was trained on two open-access datasets:

- **[HumanEval](https://huggingface.co/datasets/openai/openai_humaneval)** — Hand-written Python programming problems with function signatures and canonical solutions
- **[Lambada](https://huggingface.co/datasets/EleutherAI/lambada_openai)** — Narrative text passages for language modeling

Detailed training procedures and hyperparameters are available on each model's Hugging Face page.

## License

All models are released under the MIT License. See individual model pages for license details.
