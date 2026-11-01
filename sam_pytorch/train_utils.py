"""Small PyTorch training helpers corresponding to the SAM training code.

These functions operate on an already-created model, optimizer and iterable
of ``(images, labels)`` batches. Dataset construction and model architecture
remain in their existing modules. SAM's parameter perturbation/restoration is
delegated to :class:`sam_pytorch.optimizers.SAM`.
"""

from __future__ import annotations

import copy
import csv
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Optional

import yaml
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from sklearn.metrics import classification_report
from torch import nn

from sam_pytorch.optimizers import SAM, ExponentialMovingAverage, RMSProp


def load_config(path: str | Path) -> dict:
    """Load an experiment YAML file."""
    with Path(path).open() as config_file:
        return yaml.safe_load(config_file)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Average per-example values, safely returning zero for all-padding input."""
    weights = (torch.ones_like(values) if mask is None else
               mask.to(values).reshape(values.shape[0]))
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
    weights = _metric_weights(logits, mask)
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
    weights = _metric_weights(logits, mask)
    error_rate = 1 - (hits * weights).sum() / weights.sum()
    return torch.nan_to_num(error_rate)


def _metric_weights(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Return one metric weight per example, matching the batch dimension."""
    if mask is None:
        weights = torch.ones(logits.shape[0], device=logits.device, dtype=logits.dtype)
    else:
        weights = mask.to(device=logits.device, dtype=logits.dtype)
    return weights.reshape(logits.shape[0])


def _top_k_metric_name(top_k: int) -> str:
    return "top_5_error_rate" if top_k == 5 else f"top_{top_k}_error_rate"


def make_learning_rate_scheduler(
    optimizer: torch.optim.Optimizer,
    kind: str,
    num_epochs: int,
    num_training_obs: int,
    batch_size: int,
    *,
    warmup_epochs: float = 0,
    steps_per_epoch_override: Optional[int] = None,
    total_steps_override: Optional[int] = None,
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
    steps_per_epoch = (steps_per_epoch_override if steps_per_epoch_override is not None
                       else num_training_obs // batch_size)
    if steps_per_epoch == 0:
        raise ValueError("num_training_obs must be at least batch_size")
    total_steps = (total_steps_override if total_steps_override is not None
                   else num_epochs * steps_per_epoch)
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
        scheduler_optimizer, start_factor=1e-8, end_factor=1.0,
        total_iters=warmup_steps
    )
    return torch.optim.lr_scheduler.SequentialLR(
        scheduler_optimizer, schedulers=[warmup, decay], milestones=[warmup_steps]
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


def _loss(model, images, labels, mask, label_smoothing):
    """Classification loss only; the L2 penalty is added to the gradient."""
    logits = model(images)
    return _classification_loss(logits, labels, mask, label_smoothing), logits


@torch.no_grad()
def _add_l2_gradient(model, l2_reg, skip_1d=False):
    """Add d/dw [0.5 * l2 * sum(w^2)] = l2 * w to the gradients (one fused op).

    Equal to putting the penalty in the loss, as the JAX reference does, but
    without building ``square().sum()`` graph nodes for every parameter.
    ``skip_1d`` mirrors JAX no_weight_decay_on_bn (only tensors with ndim > 1).
    """
    if not l2_reg:
        return
    pairs = [(p.grad, p) for p in model.parameters()
             if p.grad is not None and (p.ndim > 1 or not skip_1d)]
    if pairs:
        grads, params = zip(*pairs)
        torch._foreach_add_(list(grads), list(params), alpha=l2_reg)


def _get_rng_state():
    """Capture device RNG states for replaying a SAM pass."""
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mps_state = (torch.mps.get_rng_state()
                 if torch.backends.mps.is_available() else None)
    return torch.get_rng_state(), cuda_states, mps_state


def _set_rng_state(state) -> None:
    """Restore default CPU, CUDA, and MPS RNG states."""
    cpu_state, cuda_states, mps_state = state
    torch.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
    if mps_state is not None:
        torch.mps.set_rng_state(mps_state)


def _get_mps_rng_state():
    return torch.mps.get_rng_state() if torch.backends.mps.is_available() else None


def _set_mps_rng_state(state) -> None:
    if state is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state.cpu())


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batches,
    *,
    learning_rate: Optional[float] = None,
    label_smoothing: float = 0.0,
    l2_reg: Optional[float] = None,
    gradient_clip_norm: Optional[float] = None,
    preserve_first_pass_buffers: bool = True,
    no_weight_decay_on_bn: bool = False,
) -> dict[str, float]:
    """Run one optimizer update over ``batches`` (a list of microbatches).

    Gradients of all microbatches are accumulated (weighted by their valid
    sample counts) before the update. For SAM the perturbation is computed
    from the full accumulated gradient, then every microbatch is replayed at
    the perturbed weights, so the update equals one large-batch SAM step.
    """
    moved = [_move_batch_to_model_device(model, *_unpack_batch(b)) for b in batches]
    sizes = [int(m.sum()) if m is not None else int(i.shape[0]) for i, _, m in moved]
    weights = [n / max(sum(sizes), 1) for n in sizes]
    is_sam = isinstance(optimizer, SAM)
    if is_sam and l2_reg is not None:
        for group in optimizer.param_groups:
            group["l2_reg"] = l2_reg
            group["l2_skip_1d"] = no_weight_decay_on_bn
    elif l2_reg is None:
        l2_reg = float(optimizer.param_groups[0].get("l2_reg", 0.0)) if is_sam else 0.0
    if learning_rate is not None:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

    def accumulate(rng_states=None):
        """Backward over every microbatch; returns weighted loss/error."""
        loss_sum = error_sum = 0.0
        states = []
        for i, ((images, labels, mask), weight) in enumerate(zip(moved, weights)):
            if rng_states is None:
                states.append(_get_rng_state() if is_sam else None)
            else:
                _set_rng_state(rng_states[i])
            loss, logits = _loss(model, images, labels, mask, label_smoothing)
            (loss * weight).backward()
            loss_sum = loss_sum + loss.detach() * weight
            error_sum = error_sum + error_rate_metric(logits.detach(), labels, mask) * weight
        return loss_sum, error_sum, states

    optimizer.zero_grad(set_to_none=True)
    # The JAX reference uses the same PRNG key for both SAM loss evaluations.
    # Replay each microbatch's first-pass RNG state for the perturbed pass.
    loss, error_rate, first_pass_rng = accumulate()

    if is_sam:
        optimizer.first_step(zero_grad=True)  # adds the L2 gradient itself
        buffers = (
            [buffer.detach().clone() for buffer in model.buffers()]
            if preserve_first_pass_buffers
            else None
        )
        loss, error_rate, _ = accumulate(first_pass_rng)
        if buffers is not None:
            with torch.no_grad():
                for buffer, saved in zip(model.buffers(), buffers):
                    buffer.copy_(saved)
        optimizer.add_l2_gradient()
    else:
        _add_l2_gradient(model, l2_reg, no_weight_decay_on_bn)
    clip_limit = gradient_clip_norm if gradient_clip_norm and gradient_clip_norm > 0 else float("inf")
    # clip_grad_norm_ returns the pre-clip norm; the logged (post-clip) norm
    # follows from the scale it applied, so no second pass over the gradients.
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_limit)
    grad_norm = total_norm * (clip_limit / (total_norm + 1e-6)).clamp(max=1.0)
    if is_sam:
        optimizer.second_step(l2_already_added=True)
    else:
        optimizer.step()

    # Metrics stay on the device (no per-step host sync); the epoch loop
    # converts them once. ``loss`` is the classification loss on the last
    # pass (the perturbed one for SAM), as in the JAX metrics.
    with torch.no_grad():
        params = [p.detach() for p in model.parameters()]
        param_norm = torch.linalg.vector_norm(
            torch.stack(torch._foreach_norm(params))) if params else torch.zeros(())
        return {
            "train_loss": loss,
            "train_error_rate": error_rate,
            "gradient_norm": grad_norm,
            "param_norm": param_norm,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }


def _eval_sums(model, batch, label_smoothing, top_k):
    """Metric sums and valid-sample count for one batch (device tensors)."""
    images, labels, mask = _unpack_batch(batch)
    images, labels, mask = _move_batch_to_model_device(model, images, labels, mask)
    logits = model(images)
    count = (torch.tensor(float(images.shape[0]), device=logits.device)
             if mask is None else mask.sum().to(logits.dtype))
    sums = {
        "loss": _classification_loss(logits, labels, mask, label_smoothing) * count,
        "error_rate": error_rate_metric(logits, labels, mask) * count,
    }
    if top_k is not None:
        sums[_top_k_metric_name(top_k)] = (
            top_k_error_rate_metric(logits, labels, top_k, mask) * count)
    return sums, count


@torch.no_grad()
def eval_step(
    model: nn.Module,
    batch,
    *,
    label_smoothing: float = 0.0,
    top_k: Optional[int] = None,
) -> dict[str, float]:
    """Evaluate one batch and return metric sums over its valid samples."""
    was_training = model.training
    model.eval()
    try:
        sums, _ = _eval_sums(model, batch, label_smoothing, top_k)
        return {name: float(value) for name, value in sums.items()}
    finally:
        model.train(was_training)


@torch.no_grad()
def eval_on_dataset(
    model: nn.Module,
    batches,
    *,
    label_smoothing: float = 0.0,
    top_k: Optional[int] = None,
) -> dict[str, float]:
    """Aggregate batch metric sums by valid sample count, including short batches."""
    was_training = model.training
    model.eval()
    totals: dict[str, torch.Tensor] = {}
    count = 0.0
    try:
        for batch in batches:
            sums, batch_count = _eval_sums(model, batch, label_smoothing, top_k)
            count = count + batch_count
            for name, value in sums.items():
                totals[name] = totals.get(name, 0.0) + value
    finally:
        model.train(was_training)
    denominator = max(float(count), 1.0)  # single host sync for the whole pass
    return {name: float(value) / denominator for name, value in totals.items()}


def train_for_one_epoch(
    model,
    optimizer,
    batches,
    *,
    step: int = 0,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    summary_writer=None,
    max_steps: Optional[int] = None,
    accumulation_steps: int = 1,
    freeze_batch_norm_statistics: bool = False,
    ema: Optional[ExponentialMovingAverage] = None,
    **step_kwargs,
):
    """Train for one epoch, average batch metrics, and optionally log them.

    ``step`` is the current global update count and ``max_steps`` limits
    updates, not microbatches. Each update consumes ``accumulation_steps``
    microbatches (the last one of an epoch may be shorter). The scheduler
    advances once per update; the returned step is the updated count.
    """
    model.train()
    if freeze_batch_norm_statistics:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    start_time = time.time()
    totals: dict[str, float] = {}
    count = 0
    last_learning_rate = None
    batch_iter = iter(batches)
    updates = 0
    while max_steps is None or updates < max_steps:
        group = list(islice(batch_iter, accumulation_steps))
        if not group:
            break
        updates += 1
        metrics = train_step(model, optimizer, group, **step_kwargs)
        last_learning_rate = metrics["learning_rate"]
        if scheduler is not None:
            scheduler.step()
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
        step += 1
        if ema is not None:
            ema.update(model, step)
    # Per-step metrics are device tensors; this is the epoch's only host sync.
    averages = {key: float(value) / max(count, 1) for key, value in totals.items()}
    if last_learning_rate is not None:
        averages["learning_rate"] = last_learning_rate
    if summary_writer is not None:
        for name, value in averages.items():
            summary_writer.add_scalar(name, value, step)
        summary_writer.flush()
    logging.info(
        "Whole training epoch done in %.2f seconds (%d steps)",
        time.time() - start_time,
        count,
    )
    return averages, step


def _eval_state(model, ema):
    """Weights evaluated and selected: EMA weights if used, else the model's."""
    source = ema.shadow if ema is not None else model.state_dict()
    return {k: v.detach().clone() for k, v in source.items()}


def _plain_train_loader(data, config):
    """Complete, non-augmented training split (no dropped remainder)."""
    seed = int(config.get("seed", 1))
    if hasattr(data, "_train_augmented"):
        return data.get_train_loader(use_augmentations=False, seed=seed)
    return data.get_train_loader(seed=seed, drop_last=False)


def _atomic_save(state, path: Path) -> None:
    """Save through a temporary file so a stopped write cannot corrupt a checkpoint."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def _classification_results(model, batches, class_names, label_smoothing=0.0):
    was_training = model.training
    model.eval()
    targets, predictions, loss_sum = [], [], 0.0
    try:
        for batch in batches:
            images, labels, mask = _unpack_batch(batch)
            images, labels, mask = _move_batch_to_model_device(model, images, labels, mask)
            logits = model(images)
            truth = labels.argmax(-1) if labels.ndim == logits.ndim else labels.long()
            valid = (torch.ones_like(truth, dtype=torch.bool) if mask is None
                     else mask.bool().reshape(-1))
            n = int(valid.sum())
            loss_sum += float(_classification_loss(logits, labels, mask, label_smoothing)) * n
            targets.extend(truth[valid].cpu().tolist())
            predictions.extend(logits.argmax(-1)[valid].cpu().tolist())
    finally:
        model.train(was_training)
    names = list(class_names)
    accuracy = sum(a == b for a, b in zip(targets, predictions)) / max(len(targets), 1)
    report = classification_report(targets, predictions, labels=list(range(len(names))),
                                   target_names=names, zero_division=0)
    return {"loss": loss_sum / max(len(targets), 1), "accuracy": accuracy,
            "error": 1.0 - accuracy}, report


def _make_optimizer(model, optimizer_name: str, config: dict, training: dict):
    """Create the configured optimizer with the reference's L2 handling."""
    from sam_pytorch.optimizers import make_sgd

    if training.get("use_rmsprop", False):
        # EfficientNet-paper settings: beta2=0.9, eps=0.001.
        rms_args = {"lr": float(training["learning_rate"]),
                    "beta": float(training.get("momentum", 0.9)),
                    "beta2": 0.9, "eps": 0.001}
        rho = float(config.get("sam", {}).get("rho", 0.05))
        if optimizer_name == "sam" and rho > 0:
            return SAM(model.parameters(), base_optimizer=RMSProp, rho=rho,
                       l2_skip_1d=bool(training.get("no_weight_decay_on_bn", False)),
                       **rms_args)
        if optimizer_name in ("sam", "sgd"):
            return RMSProp(model.parameters(), **rms_args)
    optimizer_args = {
        "lr": float(training["learning_rate"]),
        "momentum": float(training.get("momentum", 0.9)),
        # Weight decay is applied as an L2 loss penalty in train_step.
        "weight_decay": 0.0,
        "nesterov": bool(training.get("nesterov", True)),
    }
    if optimizer_name == "sam":
        rho = float(config.get("sam", {}).get("rho", 0.05))
        if rho == 0:
            return make_sgd(model.parameters(), **optimizer_args)
        return SAM(
            model.parameters(),
            base_optimizer=torch.optim.SGD,
            rho=rho,
            l2_skip_1d=bool(training.get("no_weight_decay_on_bn", False)),
            **optimizer_args,
        )
    if optimizer_name == "sgd":
        return make_sgd(model.parameters(), **optimizer_args)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _write_epoch_metrics(run_dir: Path, history: list[dict]) -> None:
    with (run_dir / "metrics.csv").open("w", newline="") as csv_file:
        fieldnames = list(history[-1])
        normalized_history = [
            {key: row.get(key, "") for key in fieldnames} for row in history
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_history)


def train(model: nn.Module, data, config: dict, optimizer_name: str,
          run_dir: str | Path) -> dict:
    """Train with automatic epoch checkpoint recovery and write run reports."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    training = config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    optimizer = _make_optimizer(model, optimizer_name, config, training)
    ema_decay = float(training.get("ema_decay", 0.0))
    ema = ExponentialMovingAverage(model, ema_decay) if ema_decay else None
    smooth = float(training.get("label_smoothing", 0.0))
    top_k = 5 if training.get("compute_top_5_error_rate", False) else None
    evaluate_every = int(training.get("evaluate_every", 1))
    extra_checkpoints = set(training.get("additional_checkpoints_at_epochs") or [])

    epochs, max_steps = training.get("epochs"), training.get("max_steps")
    if (epochs is None) == (max_steps is None):
        raise ValueError("Set exactly one of training.epochs or training.max_steps")
    batch_size = int(training["batch_size"])
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    if accumulation < 1:
        raise ValueError("training.gradient_accumulation_steps must be >= 1")
    effective = training.get("effective_batch_size")
    if effective is not None and int(effective) != batch_size * accumulation:
        raise ValueError(
            f"effective_batch_size {effective} != batch_size {batch_size} x "
            f"gradient_accumulation_steps {accumulation}")
    train_loader_steps = len(data.get_train_loader(seed=int(config.get("seed", 1)) + 1))
    steps_per_epoch = max(1, -(-train_loader_steps // accumulation))  # updates per epoch
    epochs = int(epochs) if epochs is not None else (int(max_steps) + steps_per_epoch - 1) // steps_per_epoch
    scheduler_steps = (int(max_steps) if max_steps is not None
                       else epochs * steps_per_epoch)
    scheduler = make_learning_rate_scheduler(
        optimizer, training.get("scheduler", "constant"), epochs,
        data.num_training_obs, batch_size * accumulation,
        warmup_epochs=float(training.get("warmup_epochs", 0)),
        steps_per_epoch_override=steps_per_epoch,
        total_steps_override=scheduler_steps)

    last_path = run_dir / "last.pt"
    history, started_at, elapsed_before, epoch_seconds = [], datetime.now(timezone.utc).isoformat(), 0.0, 0.0
    step, first_epoch = 0, 1
    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if (isinstance(optimizer, SAM) and
                state.get("base_optimizer") is not None):
            optimizer.base_optimizer.load_state_dict(state["base_optimizer"])
            optimizer.param_groups = optimizer.base_optimizer.param_groups
        if ema is not None and state.get("ema") is not None:
            ema.shadow = {k: v.to(device) for k, v in state["ema"].items()}
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        if "torch_rng" in state:
            torch.set_rng_state(state["torch_rng"].cpu())
            random.setstate(state["python_rng"])
            np.random.set_state(state["numpy_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda_rng"]])
        _set_mps_rng_state(state.get("mps_rng"))
        if "history" in state:
            history, started_at = state["history"], state["started_at"]
            elapsed_before, epoch_seconds = state["elapsed_seconds"], state["epoch_seconds"]
        else:  # Also recover the earlier checkpoint format already in outputs/.
            with (run_dir / "metrics.csv").open(newline="") as csv_file:
                history = []
                for old_row in csv.DictReader(csv_file):
                    row = {k: (int(v) if k in ("epoch", "step") else float(v))
                           for k, v in old_row.items()}
                    if "validation_accuracy" in row:
                        row["train_error_rate"] = 1 - row.pop("train_accuracy")
                        row["val_error_rate"] = 1 - row.pop("validation_accuracy")
                    history.append(row)
            log_lines = (run_dir / "train.log").read_text().splitlines()
            started_at = next(line.split(": ", 1)[1] for line in log_lines
                              if line.startswith("run_started_utc:"))
            result_lines = (run_dir / "results.txt").read_text().splitlines()
            elapsed_before = float(next(line.split(": ", 1)[1] for line in result_lines
                                        if line.startswith("run_elapsed_seconds:")))
            epoch_seconds = sum(row["epoch_seconds"] for row in history)
        step, first_epoch = state["step"], state["epoch"] + 1
        logging.info("Recovered checkpoint at epoch %d, step %d", first_epoch - 1, step)
    else:
        with (run_dir / "train.log").open("w") as log:
            log.write(f"run_started_utc: {started_at}\noptimizer: {optimizer_name}\nseed: {config.get('seed')}\ndevice: {device}\n")
    if last_path.exists():
        log_path = run_dir / "train.log"
        lines = [line for line in log_path.read_text().splitlines()
                 if not line.startswith(("run_completed_utc:", "run_elapsed_seconds:"))]
        log_path.write_text("\n".join(lines) + "\n")
    session_started = time.time()
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    def _eval_model():
        """Model evaluated each epoch: the EMA weights when EMA is on."""
        if ema is None:
            return model
        ema_model = copy.deepcopy(model)
        ema_model.load_state_dict(ema.shadow)
        return ema_model

    for epoch in range(first_epoch, epochs + 1):
        if (epoch - 1) in extra_checkpoints:
            # JAX saves these before training 0-indexed epoch n, in their own file.
            _atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "ema": ema.shadow if ema else None, "epoch": epoch - 1,
                          "step": step}, run_dir / f"additional_ckpt_{epoch - 1}.pt")
        tick = time.time()
        batches = data.get_train_loader(seed=int(config.get("seed", 1)) + epoch)
        remaining = None if max_steps is None else max(0, int(max_steps) - step)
        train_metrics, step = train_for_one_epoch(
            model, optimizer, batches, step=step, scheduler=scheduler, max_steps=remaining,
            accumulation_steps=accumulation,
            freeze_batch_norm_statistics=bool(training.get("freeze_batch_norm_statistics", False)),
            ema=ema, label_smoothing=smooth,
            l2_reg=float(training.get("weight_decay", 0.001)),
            gradient_clip_norm=training.get("gradient_clip_norm", 5.0),
            no_weight_decay_on_bn=bool(training.get("no_weight_decay_on_bn", False)))
        # As in JAX, evaluate only every `evaluate_every` epochs.
        evaluate = epoch % evaluate_every == 0
        nan = float("nan")
        validation = {"loss": nan, "error_rate": nan}
        extra_metrics = {}
        if evaluate:
            eval_model = _eval_model()
            validation = eval_on_dataset(eval_model, data.get_validation_loader(),
                                         label_smoothing=smooth, top_k=top_k)
            if top_k is not None:
                extra_metrics["val_" + _top_k_metric_name(top_k)] = validation[_top_k_metric_name(top_k)]
            if training.get("also_eval_on_training_set", False):
                train_eval = eval_on_dataset(eval_model, _plain_train_loader(data, config),
                                             label_smoothing=smooth)
                extra_metrics["eval_on_train_loss"] = train_eval["loss"]
                extra_metrics["eval_on_train_error_rate"] = train_eval["error_rate"]
        duration = time.time() - tick
        epoch_seconds += duration
        row = {"epoch": epoch, "step": step,
               "learning_rate": train_metrics.get("learning_rate", optimizer.param_groups[0]["lr"]),
               "train_loss": train_metrics.get("train_loss", 0.0),
               "train_error_rate": train_metrics.get("train_error_rate", 0.0),
               "validation_loss": validation["loss"],
               "val_error_rate": validation["error_rate"],
               "epoch_seconds": duration, **extra_metrics}
        history.append(row)
        elapsed = elapsed_before + time.time() - session_started
        _atomic_save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "ema": ema.shadow if ema else None,
            "base_optimizer": (optimizer.base_optimizer.state_dict()
                               if isinstance(optimizer, SAM) else None),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "mps_rng": _get_mps_rng_state(),
            "history": history, "started_at": started_at,
            "elapsed_seconds": elapsed,
            "epoch_seconds": epoch_seconds, "step": step, "epoch": epoch,
        }, last_path)
        with (run_dir / "train.log").open("a") as log:
            log.write(
                "epoch={epoch} step={step} lr={learning_rate:.8g} "
                "train_loss={train_loss:.4f} "
                "train_error_rate={train_error_rate:.4f} "
                "val_loss={validation_loss:.4f} "
                "val_error_rate={val_error_rate:.4f} "
                "seconds={epoch_seconds:.1f}\n".format(**row)
            )
        _write_epoch_metrics(run_dir, history)
        logging.info("epoch=%d step=%d val_error_rate=%.4f", epoch, step, row["val_error_rate"])
        if max_steps is not None and step >= int(max_steps):
            break

    # As in the JAX reference, the reported result is the final epoch's model
    # (EMA weights when EMA is on); no checkpoint is selected on validation.
    model.load_state_dict(_eval_state(model, ema))
    smooth = float(training.get("label_smoothing", 0.0))
    train_loader = _plain_train_loader(data, config)
    validation_loader, test_loader = data.get_validation_loader(), data.get_test_loader()
    train_final, train_report = _classification_results(model, train_loader, data.class_names, smooth)
    test_final, test_report = _classification_results(model, test_loader, data.class_names, smooth)
    completed = datetime.now(timezone.utc).isoformat()
    total_runtime = elapsed_before + time.time() - session_started
    with (run_dir / "train.log").open("a") as log:
        log.write(f"run_completed_utc: {completed}\nrun_elapsed_seconds: {total_runtime:.2f}\n")

    lines = [f"experiment: {config.get('experiment', 'N/A')}",
             f"dataset: {config.get('dataset', 'N/A')}", f"model: {config.get('model', 'N/A')}",
             f"optimizer: {optimizer_name}", f"seed: {config.get('seed', 'N/A')}",
             f"started_at_utc: {started_at}", f"ended_at_utc: {completed}", "status: completed",
             f"training_epoch_seconds_total: {epoch_seconds:.2f}", f"run_elapsed_seconds: {total_runtime:.2f}",
             "", "[environment]", f"python: {sys.version.split()[0]}",
             f"pytorch: {torch.__version__}", f"torchvision: {torchvision.__version__}",
             f"device: {device}", f"data_root: {config.get('data_root', 'N/A')}",
             f"pretrained_weights: {config.get('pretrained_weights', 'N/A')}",
             "", "[hyperparameters]", yaml.safe_dump(config, sort_keys=False).rstrip(), "",
             "[data]", f"train_samples: {len(train_loader.dataset)}",
             f"validation_samples: {len(validation_loader.dataset)}",
             f"test_samples: {len(test_loader.dataset)}", f"num_classes: {len(data.class_names)}",
             "classes_by_index: " + ", ".join(f"{i}={name}" for i, name in enumerate(data.class_names)), "",
             "[epoch_metrics]", "epoch,step,learning_rate,train_loss,train_error_rate,validation_loss,val_error_rate,epoch_seconds"]
    lines.extend(",".join(str(row[key]) for key in ("epoch", "step", "learning_rate", "train_loss", "train_error_rate", "validation_loss", "val_error_rate", "epoch_seconds")) for row in history)
    lines.extend(["", "[checkpoint]", "last_checkpoint: last.pt",
                  "final_evaluation_checkpoint: last epoch (no best-checkpoint selection)",
                  "", "[train_metrics]"])
    lines.extend(f"{key}: {value:.6f}" for key, value in train_final.items())
    lines.extend(["", "[train_classification_report]", train_report, "", "[test_metrics]"])
    lines.extend(f"{key}: {value:.6f}" for key, value in test_final.items())
    lines.extend(["", "[test_classification_report]", test_report])
    (run_dir / "results.txt").write_text("\n".join(lines) + "\n")
    (run_dir / "metrics.txt").unlink(missing_ok=True)
    return {"history": history,
            "train_metrics": train_final, "test_metrics": test_final}
