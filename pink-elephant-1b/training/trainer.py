import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional

from .optimizer import GradientFreeSGD, GradientFreeScheduler


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        optimizer,
        scheduler=None,
        eval_dataloader: DataLoader | None = None,
        max_epochs: int = 1,
        max_steps: int = -1,
        save_dir: str = "checkpoints",
        save_steps: int = 1000,
        log_steps: int = 10,
        eval_steps: int = 500,
        device: str = "cpu",
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.eval_dataloader = eval_dataloader
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.save_dir = save_dir
        self.save_steps = save_steps
        self.log_steps = log_steps
        self.eval_steps = eval_steps
        self.device = device
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        self._use_gradient_free = isinstance(optimizer, GradientFreeSGD)

        if self._use_gradient_free:
            optimizer.attach()

        os.makedirs(save_dir, exist_ok=True)

    def train(self):
        self.model.train()
        total_start = time.time()

        for epoch in range(self.max_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            epoch_start = time.time()

            for batch_idx, batch in enumerate(self.train_dataloader):
                loss = self._train_step(batch)
                epoch_loss += loss

                if self.global_step % self.log_steps == 0:
                    lr = self.optimizer.lr if self._use_gradient_free else self.optimizer.param_groups[0]["lr"]
                    print(
                        f"Epoch {epoch} | Step {self.global_step} | Loss {loss:.4f} | "
                        f"LR {lr:.2e}"
                    )

                if self.eval_dataloader and self.global_step % self.eval_steps == 0:
                    eval_loss = self.evaluate()
                    print(f"  Eval: {eval_loss:.4f}")
                    if eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                        self.save_checkpoint("best")
                    self.model.train()

                if self.save_steps > 0 and self.global_step % self.save_steps == 0:
                    self.save_checkpoint(f"step_{self.global_step}")

                if 0 < self.max_steps <= self.global_step:
                    total_time = time.time() - total_start
                    print(f"Reached max steps {self.max_steps}. Training complete.")
                    print(f"Total training time: {total_time / 60:.2f} minutes")
                    self.save_checkpoint("final")
                    return

            epoch_time = time.time() - epoch_start
            avg_epoch_loss = epoch_loss / (batch_idx + 1)
            print(f"Epoch {epoch} complete | Avg Loss {avg_epoch_loss:.4f} | Time {epoch_time / 60:.2f}m")

        self.save_checkpoint("final")
        total_time = time.time() - total_start
        print(f"Training complete! Total time: {total_time / 60:.2f} minutes")

    def _train_step(self, batch: dict) -> float:
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        if not self._use_gradient_free:
            self.optimizer.zero_grad()

        outputs = self.model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  WARNING: NaN/Inf loss at step {self.global_step}. Skipping.")
            self.global_step += 1
            return 0.0

        loss.backward()

        if self._use_gradient_free:
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.model.zero_grad()
        else:
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

        self.global_step += 1
        return loss.item()

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.eval_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            outputs = self.model(input_ids=input_ids, labels=labels)
            total_loss += outputs["loss"].item()
            num_batches += 1

        return total_loss / num_batches

    def save_checkpoint(self, name: str):
        path = os.path.join(self.save_dir, f"pink_elephant_{name}.pt")
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if not self._use_gradient_free else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "gradient_free_lr": self.optimizer.lr if self._use_gradient_free else None,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if self._use_gradient_free and "gradient_free_lr" in checkpoint:
            self.optimizer.lr = checkpoint["gradient_free_lr"]
        elif not self._use_gradient_free and checkpoint.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.global_step = checkpoint.get("global_step", 0)
        self.epoch = checkpoint.get("epoch", 0)
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        print(f"Checkpoint loaded from {path}")
