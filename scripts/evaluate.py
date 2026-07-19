import argparse, json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BENCHMARKS = {
    "mmlu": {"name": "mmlu", "shots": 5, "type": "multiple_choice"},
    "humaneval": {"name": "humaneval", "shots": 0, "type": "code_generation"},
    "gsm8k": {"name": "gsm8k", "shots": 8, "type": "math"},
    "hellaswag": {"name": "hellaswag", "shots": 10, "type": "multiple_choice"},
}

def evaluate(model_name: str, tasks: list[str], output: str = "eval_results.json"):
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    results = {}
    for task in tasks:
        print(f"Evaluating {task}...")
        results[task] = {"score": 0.0}

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Results saved to {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", nargs="+", default=list(BENCHMARKS.keys()))
    parser.add_argument("--output", default="eval_results.json")
    args = parser.parse_args()
    evaluate(args.model, args.tasks, args.output)
