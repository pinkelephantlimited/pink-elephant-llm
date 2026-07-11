import math
import torch
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR, ConstantLR


class GradientFreeSGD:
    """SGD with optional momentum that applies updates during backward (via hooks).

    Stores velocity buffers per parameter (momentum). Does NOT store gradients.
    Peak memory when momentum=0: weights (2.89GB for 1B) + largest gradient (~200MB)
    Peak memory when momentum=0.9: weights + velocity = 2x weights
    """

    def __init__(self, model, lr: float = 3.0, momentum: float = 0.0):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self._hooks = []
        self._step_count = 0
        self._velocity = {}

    def attach(self):
        seen = set()
        for p in self.model.parameters():
            if p.requires_grad and p.data_ptr() not in seen:
                seen.add(p.data_ptr())
                if self.momentum > 0:
                    self._velocity[p.data_ptr()] = torch.zeros_like(p.data)
                hook = p.register_hook(self._make_hook(p))
                self._hooks.append(hook)

    def _make_hook(self, p):
        mom = self.momentum
        vel = self._velocity.get(p.data_ptr(), None)

        def hook(grad):
            with torch.no_grad():
                g = grad.to(dtype=p.dtype)
                if mom > 0 and vel is not None:
                    vel.mul_(mom).add_(g, alpha=self.lr)
                    p.data.sub_(vel)
                else:
                    p.data -= self.lr * g
            return None
        return hook

    def zero_grad(self):
        pass

    def step(self):
        self._step_count += 1

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._velocity.clear()

    def state_dict(self):
        return {
            "lr": self.lr,
            "momentum": self.momentum,
            "_step_count": self._step_count,
            "velocity": {str(k): v.clone() for k, v in self._velocity.items()},
        }

    def load_state_dict(self, state_dict):
        self.lr = state_dict["lr"]
        self.momentum = state_dict.get("momentum", 0.0)
        self._step_count = state_dict["_step_count"]
        if "velocity" in state_dict:
            for k_str, v in state_dict["velocity"].items():
                self._velocity[int(k_str)] = v


class GradientFreeScheduler:
    """Minimal scheduler for GradientFreeSGD."""

    def __init__(self, optimizer, warmup_steps: int = 100, total_steps: int = 100000):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = optimizer.lr
        self._step_count = 0

    def step(self):
        self._step_count += 1
        if self._step_count <= self.warmup_steps:
            factor = 0.01 + 0.99 * self._step_count / max(1, self.warmup_steps)
        elif self._step_count >= self.total_steps:
            factor = 0.0
        else:
            progress = (self._step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            factor = 0.5 * (1.0 + math.cos(progress * math.pi))
        self.optimizer.lr = self.base_lr * factor

    def state_dict(self):
        return {
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "base_lr": self.base_lr,
            "_step_count": self._step_count,
        }

    def load_state_dict(self, state_dict):
        self.warmup_steps = state_dict["warmup_steps"]
        self.total_steps = state_dict["total_steps"]
        self.base_lr = state_dict["base_lr"]
        self._step_count = state_dict["_step_count"]


def create_optimizer(model, learning_rate: float = 3e-4, weight_decay: float = 0.1, beta1: float = 0.9, beta2: float = 0.95, eps: float = 1e-8):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps,
    )


def create_sgd_optimizer(model, learning_rate: float = 3.0, momentum: float = 0.0, weight_decay: float = 0.0):
    return SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )


def create_gradient_free_sgd(model, learning_rate: float = 3.0, momentum: float = 0.9):
    return GradientFreeSGD(model, lr=learning_rate, momentum=momentum)


def create_gradient_free_scheduler(optimizer, warmup_steps: int = 100, total_steps: int = 100000):
    return GradientFreeScheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)


def create_scheduler(optimizer, warmup_steps: int = 1000, total_steps: int = 100000):
    warmup = LinearLR(optimizer, start_factor=0.0, end_factor=1.0, total_iters=warmup_steps)
    decay = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    return SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_steps])


def create_sgd_scheduler(optimizer, warmup_steps: int = 100, total_steps: int = 100000):
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    decay = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    return SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_steps])
