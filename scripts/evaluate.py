"""Evaluation script for Pink Elephant LLM models."""
import argparse, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def evaluate(model_name, tasks):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    results = {t: {"score": 0.0} for t in tasks}
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", nargs="+", default=["mmlu", "humaneval"])
    evaluate(parser.parse_args().model, parser.parse_args().tasks)
