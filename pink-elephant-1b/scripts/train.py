"""
Training script for Pink Elephant:1B — optimized for low-memory training.

On MacBook Air M1 (8GB RAM):
    python scripts/train.py --data data/tinystories.txt --max-steps 5000

On GPU (CUDA):
    python scripts/train.py --data data/tinystories.txt --optimizer adamw --batch-size 4 --max-steps 5000
"""
import argparse
import os
import sys
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from config import PinkElephant400MConfig, PinkElephant1BConfig, PinkElephant50MConfig
from model import PinkElephantForCausalLM
from tokenizer import PinkElephantTokenizer, create_base_vocab
from training import (
    TextDataset,
    DataCollator,
    Trainer,
    create_optimizer,
    create_sgd_optimizer,
    create_gradient_free_sgd,
    create_scheduler,
    create_sgd_scheduler,
    create_gradient_free_scheduler,
)


def get_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def configure_memory(device: str):
    if device == "cpu":
        torch.set_num_threads(4)
    elif device == "mps":
        torch.set_num_threads(2)


def main():
    parser = argparse.ArgumentParser(description="Train Pink Elephant:1B (low-memory mode)")
    parser.add_argument("--data", type=str, required=True, help="Path to training data file")
    parser.add_argument("--eval-data", type=str, default=None, help="Path to eval data file")
    parser.add_argument("--vocab-file", type=str, default="vocab.json", help="Path to vocab file")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--max-steps", type=int, default=5000, help="Max training steps")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (keep 1 for low RAM)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--model-size", type=str, default="50m", choices=["50m", "400m", "1b"], help="Model size: 50m (~50M params), 400m (~350M params) or 1b (~1.45B params)")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw", "gradient_free"], help="Optimizer: sgd, adamw, or gradient_free (~3GB)")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False, help="Enable gradient checkpointing (saves memory, slower)")
    parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length (lower = less RAM)")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)")
    parser.add_argument("--save-steps", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--log-steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--eval-steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--dtype", type=str, default="auto", help="Torch dtype: auto, float32, bfloat16")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-subsample", type=int, default=None, help="Use only N lines from data (for testing)")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    configure_memory(device)

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if args.dtype == "auto":
        if device in ("cuda", "mps"):
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32
    else:
        torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)

    print(f"Device: {device}  |  PyTorch: {torch.__version__}")
    print(f"Dtype: {torch_dtype}  |  Optimizer: {args.optimizer}  |  Seq: {args.max_length}  |  Batch: {args.batch_size}")
    print(f"Gradient checkpointing: {args.gradient_checkpointing} | Max steps: {args.max_steps}")

    print("Creating tokenizer...")
    if not os.path.exists(args.vocab_file):
        print("Generating vocab file...")
        create_base_vocab(args.vocab_file)
    tokenizer = PinkElephantTokenizer(vocab_file=args.vocab_file)

    print("Creating model...")
    if args.model_size == "1b":
        config = PinkElephant1BConfig()
    elif args.model_size == "400m":
        config = PinkElephant400MConfig()
    else:
        config = PinkElephant50MConfig()

    bytes_per_param = 2 if torch_dtype == torch.bfloat16 else 4

    # Estimate params before creation
    est = config.num_params_estimate
    weight_gb = est * bytes_per_param / 1e9
    print(f"Estimated: {est/1e6:.0f}M params | Weights: {weight_gb:.2f}GB + Gradients: {weight_gb:.2f}GB = ~{weight_gb*2:.1f}GB")

    with torch.device(device):
        model = PinkElephantForCausalLM(config)

    if torch_dtype != torch.float32:
        for param in model.parameters():
            param.data = param.data.to(torch_dtype)

    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    actual = sum(p.numel() for p in model.parameters())
    print(f"Actual params: {actual/1e6:.2f}M on {device}")

    param_device = next(model.parameters()).device
    param_dtype = next(model.parameters()).dtype
    print(f"Model on {param_device} dtype={param_dtype}")

    if args.gradient_checkpointing:
        model.model.gradient_checkpointing = True
        print("Gradient checkpointing enabled")

    print("Loading data...")
    with open(args.data) as f:
        texts = [line.strip() for line in f if line.strip()]

    if args.data_subsample is not None:
        texts = texts[:args.data_subsample]
        print(f"Subsampled to {len(texts)} lines")

    if args.eval_data:
        with open(args.eval_data) as f:
            eval_texts = [line.strip() for line in f if line.strip()]
    else:
        split_idx = max(1, int(len(texts) * 0.95))
        eval_texts = texts[split_idx:]
        texts = texts[:split_idx]

    print(f"Train samples: {len(texts)}, Eval samples: {len(eval_texts)}")

    train_dataset = TextDataset(texts, tokenizer, max_length=args.max_length)
    eval_dataset = TextDataset(eval_texts, tokenizer, max_length=args.max_length)

    collator = DataCollator()
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=0
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=0
    )

    print("Creating optimizer...")
    warmup_steps = max(1, min(200, args.max_steps // 10))

    if args.optimizer == "sgd":
        optimizer = create_sgd_optimizer(model, learning_rate=args.lr)
        scheduler = create_sgd_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=args.max_steps)
        print(f"SGD optimizer | LR={args.lr} | momentum=0.0 | warmup={warmup_steps}")
    elif args.optimizer == "gradient_free":
        optimizer = create_gradient_free_sgd(model, learning_rate=args.lr)
        scheduler = create_gradient_free_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=args.max_steps)
        print(f"GradientFreeSGD | LR={args.lr} | peak mem ~3.1GB | warmup={warmup_steps}")
    else:
        optimizer = create_optimizer(model, learning_rate=args.lr)
        scheduler = create_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=args.max_steps)
        print(f"AdamW optimizer | LR={args.lr} | warmup={warmup_steps}")

    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        eval_dataloader=eval_dataloader,
        max_epochs=100,
        max_steps=args.max_steps,
        save_dir=args.output_dir,
        save_steps=args.save_steps,
        log_steps=args.log_steps,
        eval_steps=args.eval_steps,
        device=device,
    )

    print("Starting training...")
    try:
        trainer.train()
    except torch.OutOfMemoryError as e:
        print(f"\nOUT OF MEMORY: {e}")
        print("Try: python scripts/train.py --data data/tinystories.txt --max-steps 5000 --device cpu --dtype float32")
        sys.exit(1)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "mps" in str(e).lower():
            print(f"\nMEMORY ERROR: {e}")
            print("Try: python scripts/train.py --data data/tinystories.txt --max-steps 5000 --device cpu --dtype float32")
        else:
            raise


if __name__ == "__main__":
    main()
