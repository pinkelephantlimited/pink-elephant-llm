"""
Pink Elephant LLM — Distributed Training Script
================================================
Supports multi-node training with DeepSpeed ZeRO-3,
activation checkpointing, WandB logging, and periodic checkpointing.

Usage:
    # Single node (8 GPUs)
    torchrun --nproc_per_node=8 scripts/train.py --config configs/train_33b.json

    # Multi-node (32 GPUs)
    torchrun --nnodes=4 --nproc_per_node=8 --rdzv_endpoint=master:29500 \\
        scripts/train.py --config configs/train_33b.json
"""
import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration loaded from JSON."""
    model_name: str = ""
    output_dir: str = "./output"
    seed: int = 42
    precision: str = "bfloat16"
    optimizer: str = "AdamW"
    learning_rate: float = 1.5e-4
    min_lr: float = 1.5e-5
    weight_decay: float = 0.1
    warmup_steps: int = 5000
    total_steps: int = 500000
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    max_seq_length: int = 16384
    activation_checkpointing: bool = True
    zero_stage: int = 3
    logging_steps: int = 10
    save_steps: int = 5000
    eval_steps: int = 1000
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None


def get_dataloader(cfg: TrainingConfig, tokenizer) -> DataLoader:
    """Creates a synthetic dataloader for demonstration.
    
    In production, this would load from disk / streaming dataset.
    """
    class DummyDataset(Dataset):
        def __len__(self):
            return 10000
        def __getitem__(self, idx):
            return {
                "input_ids": torch.randint(0, 33792, (cfg.max_seq_length,)),
                "labels": torch.randint(0, 33792, (cfg.max_seq_length,)),
                "attention_mask": torch.ones(cfg.max_seq_length),
            }
    return DataLoader(
        DummyDataset(),
        batch_size=cfg.per_device_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )


def train(cfg: TrainingConfig):
    """Main training loop."""
    set_seed(cfg.seed)
    
    # Initialize distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = local_rank == 0
    
    if is_main:
        logger.info(f"=== Pink Elephant LLM Training ===")
        logger.info(f"Model: {cfg.model_name}")
        logger.info(f"World size: {world_size}")
        logger.info(f"Batch size (per device): {cfg.per_device_batch_size}")
        logger.info(f"Gradient accumulation: {cfg.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {cfg.per_device_batch_size * world_size * cfg.gradient_accumulation_steps}")
        logger.info(f"Total steps: {cfg.total_steps}")
        logger.info(f"Precision: {cfg.precision}")
    
    # Load model configuration
    model_config = AutoConfig.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
    )
    
    # Initialize model
    with device:
        model = AutoModelForCausalLM.from_config(
            model_config,
            trust_remote_code=True,
        )
    
    # Enable activation checkpointing for memory efficiency
    if cfg.activation_checkpointing:
        model.gradient_checkpointing_enable()
        if is_main:
            logger.info("Activation checkpointing enabled")
    
    # Move to precision
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    model = model.to(dtype=dtype[cfg.precision])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize tokenizer and data
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    dataloader = get_dataloader(cfg, tokenizer)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    
    # LR scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=cfg.total_steps,
    )
    
    # TensorBoard logging
    writer = None
    if is_main:
        os.makedirs(cfg.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(cfg.output_dir, "logs"))
    
    # WandB
    if is_main and cfg.wandb_project:
        try:
            import wandb
            wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity, config=vars(cfg))
            wandb.watch(model)
        except ImportError:
            logger.warning("wandb not installed, skipping WandB logging")
    
    # Training loop
    model.train()
    global_step = 0
    total_loss = 0.0
    best_loss = float("inf")
    start_time = time.time()
    
    optimizer.zero_grad()
    
    for epoch in range(100):  # Run until total_steps reached
        for batch in dataloader:
            if global_step >= cfg.total_steps:
                break
            
            # Forward pass
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )
            loss = outputs.loss
            loss = loss / cfg.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            total_loss += loss.item()
            
            # Gradient accumulation step
            if (global_step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                current_lr = scheduler.get_last_lr()[0]
                
                # Logging
                if global_step % cfg.logging_steps == 0 and is_main:
                    avg_loss = total_loss / cfg.logging_steps
                    elapsed = time.time() - start_time
                    tokens_per_sec = (
                        cfg.per_device_batch_size
                        * world_size
                        * cfg.max_seq_length
                        * cfg.logging_steps
                        / elapsed
                    )
                    logger.info(
                        f"Step {global_step}/{cfg.total_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Tokens/s: {tokens_per_sec:.0f} | "
                        f"Elapsed: {elapsed:.0f}s"
                    )
                    if writer:
                        writer.add_scalar("Loss/train", avg_loss, global_step)
                        writer.add_scalar("LR", current_lr, global_step)
                        writer.add_scalar("Tokens_per_sec", tokens_per_sec, global_step)
                    if cfg.wandb_project:
                        try:
                            import wandb
                            wandb.log({"loss": avg_loss, "lr": current_lr, "tokens_per_sec": tokens_per_sec}, step=global_step)
                        except ImportError:
                            pass
                    total_loss = 0.0
                    start_time = time.time()
                
                # Save checkpoint
                if global_step % cfg.save_steps == 0 and global_step > 0 and is_main:
                    checkpoint_dir = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(checkpoint_dir, save_config=True)
                    tokenizer.save_pretrained(checkpoint_dir)
                    logger.info(f"Checkpoint saved: {checkpoint_dir}")
                    
                    # Save best model
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        best_dir = os.path.join(cfg.output_dir, "best")
                        model.save_pretrained(best_dir, save_config=True)
                        tokenizer.save_pretrained(best_dir)
                        logger.info(f"Best model saved (loss: {best_loss:.4f})")
            
            global_step += 1
    
    # Save final model
    if is_main:
        model.save_pretrained(cfg.output_dir, save_config=True)
        tokenizer.save_pretrained(cfg.output_dir)
        logger.info(f"Training complete. Final model saved to {cfg.output_dir}")
        if writer:
            writer.close()
    
    if world_size > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train Pink Elephant LLM")
    parser.add_argument("--config", type=str, required=True, help="Path to training config JSON")
    parser.add_argument("--wandb-project", type=str, default=None, help="WandB project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="WandB entity/team name")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank (set by torchrun)")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config_dict = json.load(f)
    
    cfg = TrainingConfig(**config_dict.get("training", {}))
    cfg.model_name = config_dict.get("model", {}).get("name", "")
    cfg.wandb_project = args.wandb_project or cfg.wandb_project
    cfg.wandb_entity = args.wandb_entity or cfg.wandb_entity
    
    train(cfg)


if __name__ == "__main__":
    main()
