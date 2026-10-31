"""Small PyTorch training helpers corresponding to the SAM training code.

These functions operate on an already-created model, optimizer and iterable
of ``(images, labels)`` batches. Dataset construction and model architecture
remain in their existing modules. SAM's parameter perturbation/restoration is
delegated to :class:`sam_pytorch.optimizers.SAM`.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from sam_pytorch.optimizers import SAM


def _masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Average per-example values, safely returning zero for all-padding input."""
    weights = torch.ones_like(values) if mask is None else mask.to(values)
    return (values * weights).sum() / weights.sum().clamp_min(1)


def error_rate_metric(
    logits: torch.Tensor, labels: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Fraction of valid examples whose top-1 class is incorrect.

    ``labels`` may be one-hot encoded (matching ``logits``) or class indices.
    The mask is flattened to one weight per example, matching the source
    metric's batch-dimension semantics. An all-zero mask returns zero.
    """
    targets = labels.argmax(dim=-1) if labels.ndim == logits.ndim else labels.long()
    errors = (logits.argmax(dim=-1) != targets).to(logits.dtype)
    weights = (torch.ones(logits.shape[0], device=logits.device, dtype=logits.dtype)
               if mask is None else mask.to(device=logits.device, dtype=logits.dtype))
    weights = weights.reshape(logits.shape[0])
    error_rate = (errors * weights).sum() / weights.sum()
    return torch.nan_to_num(error_rate)


def top_k_error_rate_metric(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int = 5,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fraction of valid examples whose target is absent from the top ``k``.

    Matches the source metric's ``argsort(...)[..., -k:]`` selection and
    averages hits using one mask weight per example. ``labels`` may be
    one-hot encoded or class indices.
    """
    targets = labels.argmax(dim=-1) if labels.ndim == logits.ndim else labels.long()
    targets = targets.reshape(-1, 1)
    top_k_preds = torch.argsort(logits, dim=-1)[..., -k:]
    hits = top_k_preds.eq(targets).any(dim=-1).to(logits.dtype)
    weights = (torch.ones(logits.shape[0], device=logits.device, dtype=logits.dtype)
               if mask is None else mask.to(device=logits.device, dtype=logits.dtype))
    weights = weights.reshape(logits.shape[0])
    error_rate = 1 - (hits * weights).sum() / weights.sum()
    return torch.nan_to_num(error_rate)


def make_learning_rate_scheduler(
    optimizer: torch.optim.Optimizer,
    kind: str,
    num_epochs: int,
    num_training_obs: int,
    batch_size: int,
    *,
    warmup_epochs: float = 0,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """Build a per-update scheduler from PyTorch's scheduler implementations.

    The number of updates per epoch follows the source's floor division. A
    constant learning rate needs no scheduler and returns ``None``.
    """
    if batch_size <= 0 or num_epochs <= 0 or num_training_obs <= 0:
        raise ValueError("num_epochs, num_training_obs and batch_size must be positive")
    if kind in ("constant", "none"):
        return None
    if kind not in ("cosine", "exponential"):
        raise ValueError(f"Unsupported learning-rate schedule: {kind}")

    # SAM delegates its real update to the wrapped optimizer. Attach PyTorch's
    # scheduler there so its step-order tracking observes that base update.
    scheduler_optimizer = optimizer.base_optimizer if isinstance(optimizer, SAM) else optimizer
    # The source uses floor division (and, for its single-host setup, host_count
    # is one). A zero-update epoch cannot define either schedule's epoch scale.
    steps_per_epoch = num_training_obs // batch_size
    if steps_per_epoch == 0:
        raise ValueError("num_training_obs must be at least batch_size")
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = min(total_steps - 1, max(0, int(warmup_epochs * steps_per_epoch)))
    decay_steps = max(1, total_steps - warmup_steps)
    if kind == "cosine":
        decay = torch.optim.lr_scheduler.CosineAnnealingLR(
            scheduler_optimizer, T_max=decay_steps, eta_min=0.0
        )
    else:
        # ExponentialLR reaches 1.2% of the starting LR over decay_steps.
        decay = torch.optim.lr_scheduler.ExponentialLR(
            scheduler_optimizer, gamma=0.012 ** (1 / decay_steps)
        )
    if not warmup_steps:
        return decay
    warmup = torch.optim.lr_scheduler.LinearLR(
        scheduler_optimizer, start_factor=0.0, end_factor=1.0, total_iters=warmup_steps
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, decay], milestones=[warmup_steps]
    )


def _unpack_batch(batch):
    """Accept the repo's dict batches or the usual ``(images, labels)`` pair."""
    if isinstance(batch, dict):
        images = batch.get("image", batch.get("images"))
        labels = batch.get("label", batch.get("labels"))
        mask = batch.get("mask")
    else:
        images, labels = batch[:2]
        mask = batch[2] if len(batch) > 2 else None
    if images is None or labels is None:
        raise ValueError("Batch must contain image(s) and label(s)")
    return images, labels, mask


def _move_batch_to_model_device(model, images, labels, mask):
    """Move this single-process batch onto the model's device (GPU or CPU)."""
    device = next(model.parameters()).device
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True) if mask is not None else None
    return images, labels, mask


def _classification_loss(logits, labels, mask, label_smoothing):
    targets = labels.to(logits.dtype) if labels.ndim == logits.ndim else labels.long()
    per_sample_loss = F.cross_entropy(
        logits, targets, reduction="none", label_smoothing=label_smoothing
    )
    return _masked_mean(per_sample_loss, mask)


def _loss(model, images, labels, mask, label_smoothing, l2_reg, *, include_l2=True):
    logits = model(images)
    loss = _classification_loss(logits, labels, mask, label_smoothing)
    if include_l2 and l2_reg:
        # The reference adds lambda/2 * sum(w^2) to loss (including bias and
        # normalization parameters unless the caller chooses a filtered model).
        penalty = sum(p.square().sum() for p in model.parameters() if p.requires_grad)
        loss = loss + 0.5 * l2_reg * penalty
    return loss, logits


def _get_rng_state():
    """Capture default CPU and CUDA RNG states for replaying a SAM pass."""
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return torch.get_rng_state(), cuda_states


def _set_rng_state(state) -> None:
    """Restore default CPU and CUDA RNG states."""
    cpu_state, cuda_states = state
    torch.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch,
    *,
    learning_rate: Optional[float] = None,
    label_smoothing: float = 0.0,
    l2_reg: Optional[float] = None,
    gradient_clip_norm: Optional[float] = None,
    preserve_first_pass_buffers: bool = True,
) -> dict[str, float]:
    """Run one SGD update or the two forward/backward passes required by SAM."""
    images, labels, mask = _unpack_batch(batch)
    images, labels, mask = _move_batch_to_model_device(model, images, labels, mask)
    is_sam = isinstance(optimizer, SAM)
    if is_sam and l2_reg is not None:
        for group in optimizer.param_groups:
            group["l2_reg"] = l2_reg
    elif l2_reg is None:
        l2_reg = float(optimizer.param_groups[0].get("l2_reg", 0.0)) if is_sam else 0.0
    if learning_rate is not None:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

    optimizer.zero_grad(set_to_none=True)
    # The JAX reference uses the same PRNG key for both SAM loss evaluations.
    # Replay the first pass's RNG state for the perturbed pass, then leave the
    # global generators advanced by one pass, as they would be without replay.
    first_pass_rng = _get_rng_state() if is_sam else None
    loss, logits = _loss(model, images, labels, mask, label_smoothing,
                         0.0 if is_sam else l2_reg)
    loss.backward()

    if is_sam:
        rng_after_first_pass = _get_rng_state()
        optimizer.first_step(zero_grad=True)
        buffers = ([buffer.detach().clone() for buffer in model.buffers()]
                   if preserve_first_pass_buffers else None)
        _set_rng_state(first_pass_rng)
        loss, logits = _loss(model, images, labels, mask, label_smoothing, 0.0)
        loss.backward()
        _set_rng_state(rng_after_first_pass)
        if buffers is not None:
            with torch.no_grad():
                for buffer, saved in zip(model.buffers(), buffers):
                    buffer.copy_(saved)
        optimizer.add_l2_gradient()
    clip_limit = gradient_clip_norm if gradient_clip_norm and gradient_clip_norm > 0 else float("inf")
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_limit)
    grad_norms = [p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None]
    grad_norm = float(torch.linalg.vector_norm(torch.stack(grad_norms))) if grad_norms else 0.0
    if is_sam:
        optimizer.second_step(l2_already_added=True)
    else:
        optimizer.step()

    train_loss = float(_classification_loss(
        logits.detach(), labels, mask, label_smoothing
    ))
    train_error = float(error_rate_metric(logits.detach(), labels, mask))
    param_norms = [p.detach().norm(2) for p in model.parameters()]
    param_norm = float(torch.linalg.vector_norm(torch.stack(param_norms))) if param_norms else 0.0
    lr = float(optimizer.param_groups[0]["lr"])
    return {"train_loss": train_loss, "train_error_rate": train_error,
            "gradient_norm": grad_norm, "param_norm": param_norm, "learning_rate": lr}


@torch.no_grad()
def eval_step(model: nn.Module, batch, *, label_smoothing: float = 0.0,
              top_k: Optional[int] = None) -> dict[str, float]:
    """Evaluate one batch and return metric sums over its valid samples."""
    was_training = model.training
    model.eval()
    try:
        images, labels, mask = _unpack_batch(batch)
        images, labels, mask = _move_batch_to_model_device(model, images, labels, mask)
        logits = model(images)
        count = images.shape[0] if mask is None else float(mask.sum())
        result = {
            "loss": float(_classification_loss(logits, labels, mask, label_smoothing) * count),
            "error_rate": float(error_rate_metric(logits, labels, mask) * count),
        }
        if top_k is not None:
            key = "top_5_error_rate" if top_k == 5 else f"top_{top_k}_error_rate"
            result[key] = float(top_k_error_rate_metric(logits, labels, top_k, mask) * count)
        return result
    finally:
        model.train(was_training)


@torch.no_grad()
def eval_on_dataset(model: nn.Module, batches, *, label_smoothing: float = 0.0,
                    top_k: Optional[int] = None) -> dict[str, float]:
    """Aggregate batch metric sums by valid sample count, including short batches."""
    was_training = model.training
    model.eval()
    totals = {"loss": 0.0, "error_rate": 0.0}
    if top_k is not None:
        key = "top_5_error_rate" if top_k == 5 else f"top_{top_k}_error_rate"
        totals[key] = 0.0
    count = 0.0
    try:
        for batch in batches:
            images, _, mask = _unpack_batch(batch)
            count += images.shape[0] if mask is None else float(mask.sum())
            for name, value in eval_step(model, batch, label_smoothing=label_smoothing,
                                         top_k=top_k).items():
                totals[name] = totals.get(name, 0.0) + value
    finally:
        model.train(was_training)
    return {name: value / max(count, 1.0) for name, value in totals.items()}


def train_for_one_epoch(model, optimizer, batches, *, step: int = 0,
                        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
                        summary_writer=None,
                        **step_kwargs):
    """Train for one epoch, average batch metrics, and optionally log them.

    ``step`` is the current global update count. The scheduler advances once
    after each optimizer update; the returned step is the updated count.
    """
    model.train()
    start_time = time.time()
    totals: dict[str, float] = {}
    count = 0
    for batch in batches:
        metrics = train_step(model, optimizer, batch, **step_kwargs)
        if scheduler is not None:
            scheduler.step()
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
        step += 1
    averages = {key: value / max(count, 1) for key, value in totals.items()}
    if summary_writer is not None:
        for name, value in averages.items():
            summary_writer.add_scalar(name, value, step)
        summary_writer.flush()
    logging.info("Whole training epoch done in %.2f seconds (%d steps)",
                 time.time() - start_time, count)
    return averages, step
