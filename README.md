# Pink Elephant LLM

Open-source large language models by Pink Elephant Limited.

## Models

| Model | Parameters | Description |
|-------|-----------|-------------|
| [Pink Elephant 1.3B](https://huggingface.co/pinkelephantlimited/pink-elephant-llm-1.3b) | 1.3B | Lightweight general-purpose LLM |
| [Pink Elephant 6.7B](https://huggingface.co/pinkelephantlimited/pink-elephant-llm-6.7b) | 6.7B | Mid-size model for broader capability |
| [Pink Elephant 33B](https://huggingface.co/pinkelephantlimited/pink-elephant-llm-33b) | 33B | Large-scale model for complex tasks |
| [Pink Elephant Micro Coder](https://huggingface.co/pinkelephantlimited/pink-elephant-micro-coder) | 22M | Tiny code completion model (training in progress) |

## Usage

All models can be used via Hugging Face `transformers`:

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="pinkelephantlimited/pink-elephant-llm-1.3b")
result = pipe("Hello, I'm", max_new_tokens=50)[0]["generated_text"]
print(result)
```

## License

MIT License — see `LICENSE` file for details.
