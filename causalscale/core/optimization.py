"""
Advanced Optimization Engine
==============================
Mixed-precision training, cosine annealing, gradient accumulation, and
scale-specific learning rate scheduling.

Key features:
  1. Mixed Precision (FP16/BF16): Automatic mixed precision via torch.cuda.amp
  2. Cosine Annealing: smooth LR decay for better convergence
  3. Warmup: gradual LR increase at start to avoid early instability
  4. Gradient Accumulation: simulate large batches on limited GPU memory
  5. Adaptive Gradient Clipping: percentile-based clipping threshold
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict, Tuple
import math


class CosineScheduler:
    """
    Cosine annealing with linear warmup.

    LR schedule:
        warmup:   lr = base_lr * (step / warmup_steps)
        cosine:   lr = base_lr * 0.5 * (1 + cos(pi * (step - warmup) / (total - warmup)))
        plateau:  lr = base_lr * final_factor

    Args:
        base_lr: peak learning rate
        total_steps: total training steps (epochs * steps_per_epoch)
        warmup_steps: number of warmup steps (default: 10% of total)
        final_factor: final LR as fraction of base_lr (default: 0.01)
    """

    def __init__(
        self,
        base_lr: float,
        total_steps: int,
        warmup_steps: Optional[int] = None,
        final_factor: float = 0.01,
        min_warmup_ratio: float = 0.1  # start warmup at 10% of base_lr, not 1/warmup_steps
    ):
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps or max(1, total_steps // 10)
        self.final_factor = final_factor
        self.min_warmup_ratio = min_warmup_ratio
        self.current_step = 0

    def step(self) -> float:
        """Get LR for current step and advance."""
        lr = self.get_lr(self.current_step)
        self.current_step += 1
        return lr

    def get_lr(self, step: int) -> float:
        """Get LR for given step without advancing."""
        if step < self.warmup_steps:
            # Linear warmup from min_warmup_ratio to base_lr
            progress = step / max(self.warmup_steps - 1, 1)
            return self.base_lr * (self.min_warmup_ratio + (1 - self.min_warmup_ratio) * progress)
        elif step >= self.total_steps:
            # Plateau
            return self.base_lr * self.final_factor
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.base_lr * self.final_factor + \
                   self.base_lr * (1 - self.final_factor) * 0.5 * (1 + math.cos(math.pi * progress))


class MixedPrecisionTrainer:
    """
    Mixed-precision training wrapper with gradient accumulation.

    Automatically uses FP16 for forward/backward (via torch.cuda.amp)
    and FP32 for parameter updates. Falls back to FP32 on CPU.

    Usage:
        trainer = MixedPrecisionTrainer(model, optimizer, accumulation_steps=4)
        for epoch in range(epochs):
            loss = trainer.training_step(loss_fn, X, D)
            if loss is not None:  # step completed
                print(f"Loss: {loss}")
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        accumulation_steps: int = 1,
        use_amp: bool = True,
        max_grad_norm: float = 1.0,
        grad_clip_mode: str = 'value'  # 'value' or 'percentile'
    ):
        """
        Args:
            model: PyTorch model
            optimizer: optimizer instance
            accumulation_steps: number of forward passes before optimizer.step()
            use_amp: enable automatic mixed precision (GPU only)
            max_grad_norm: gradient clipping threshold
            grad_clip_mode: 'value' (fixed threshold) or 'percentile' (adaptive)
        """
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.grad_clip_mode = grad_clip_mode

        self.use_amp = use_amp and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        self._step_count = 0
        self._total_loss = 0.0

    def training_step(
        self,
        loss_fn: callable,
        *args,
        **kwargs
    ) -> Optional[float]:
        """
        Execute one training step with optional gradient accumulation.

        Args:
            loss_fn: callable that returns (loss, components_dict)

        Returns:
            Average loss if accumulation step completed, None otherwise
        """
        if self.use_amp:
            with torch.amp.autocast('cuda'):
                loss, components = loss_fn(*args, **kwargs)
        else:
            loss, components = loss_fn(*args, **kwargs)

        # Scale loss for gradient accumulation
        scaled_loss = loss / self.accumulation_steps
        if self.use_amp:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        self._total_loss += float(loss.item())

        self._step_count += 1

        if self._step_count % self.accumulation_steps == 0:
            self._optimizer_step()
            avg_loss = self._total_loss / self.accumulation_steps
            self._total_loss = 0.0
            return avg_loss

        return None

    def _optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        if self.use_amp:
            # Unscale before clipping
            self.scaler.unscale_(self.optimizer)

        if self.grad_clip_mode == 'percentile':
            # Adaptive: clip at 95th percentile of gradient norms
            norms = []
            for p in self.model.parameters():
                if p.grad is not None:
                    norms.append(p.grad.norm().item())
            if norms:
                p95 = np.percentile(norms, 95) if len(norms) > 1 else norms[0]
                clip_val = max(self.max_grad_norm, p95 * 1.5)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), clip_val
                )
        else:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )

        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()

    def state_dict(self) -> Dict:
        if self.use_amp:
            return {
                'scaler': self.scaler.state_dict(),
                'step_count': self._step_count,
            }
        return {'step_count': self._step_count}

    def load_state_dict(self, state: Dict):
        if self.use_amp and 'scaler' in state:
            self.scaler.load_state_dict(state['scaler'])
        self._step_count = state['step_count']


def create_optimizer_with_scheduler(
    model: nn.Module,
    base_lr: float = 0.001,
    total_steps: int = 1000,
    warmup_steps: Optional[int] = None,
    weight_decay: float = 0.0,
    multi_scale: bool = False,
    scale_lr_multipliers: Optional[List[float]] = None
) -> Tuple[torch.optim.Optimizer, CosineScheduler]:
    """
    Create optimizer with cosine annealing scheduler.

    For multi-scale models, creates parameter groups with scale-specific LR.
    """
    if multi_scale and hasattr(model, 'get_parameter_groups'):
        param_groups = model.get_parameter_groups(base_lr)
        optimizer = torch.optim.AdamW(
            param_groups, weight_decay=weight_decay
        )
    elif multi_scale and scale_lr_multipliers:
        # Manual parameter groups
        param_groups = []
        all_params = list(model.parameters())
        for i, params in enumerate(all_params):
            lr = base_lr * scale_lr_multipliers[min(i, len(scale_lr_multipliers) - 1)]
            param_groups.append({'params': [params], 'lr': lr})
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=base_lr, weight_decay=weight_decay
        )

    scheduler = CosineScheduler(
        base_lr=base_lr,
        total_steps=total_steps,
        warmup_steps=warmup_steps
    )

    return optimizer, scheduler
