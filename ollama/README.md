# Ollama Deployment

These Modelfiles allow running Pink Elephant LLM models via [Ollama](https://ollama.ai).

## Usage

```bash
# Create the model
ollama create pink-elephant-llm-1.3b -f ollama/Modelfile

# Run inference
ollama run pink-elephant-llm-1.3b
```

## Available Models

| Model | Modelfile |
|-------|-----------|
| 1.3B | [Modelfile](Modelfile) |
