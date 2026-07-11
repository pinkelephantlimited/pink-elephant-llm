"""
Inference script for Pink Elephant:1B.

Example usage:
    python scripts/inference.py --prompt "Once upon a time" --max-tokens 200
    python scripts/inference.py --checkpoint checkpoints/pink_elephant_best.pt --interactive
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import PinkElephant1BConfig
from model import PinkElephantForCausalLM
from tokenizer import PinkElephantTokenizer
from inference import InferenceEngine, chat


def main():
    parser = argparse.ArgumentParser(description="Run inference with Pink Elephant:1B")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--vocab-file", type=str, default="vocab.json", help="Path to vocab file")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    parser.add_argument("--stream", action="store_true", help="Stream output token by token")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Torch dtype")

    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)

    print(f"Loading Pink Elephant:1B on {device}...")

    config = PinkElephant1BConfig()
    model = PinkElephantForCausalLM(config)

    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint found. Using randomly initialized weights (untrained).")

    tokenizer = PinkElephantTokenizer(vocab_file=args.vocab_file)

    engine = InferenceEngine(model, tokenizer, device=device, torch_dtype=torch_dtype)

    if args.interactive:
        chat(engine)
        return

    if args.prompt is None:
        args.prompt = input("Enter prompt: ")

    if args.stream:
        print("Pink Elephant: ", end="", flush=True)
        for token in engine.stream_generate(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        ):
            if token != args.prompt:
                print(token, end="", flush=True)
        print()
    else:
        output = engine.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        print(f"Prompt: {args.prompt}")
        print(f"\nGenerated: {output}")


if __name__ == "__main__":
    main()
