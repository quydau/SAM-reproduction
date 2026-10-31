"""Small PyTorch training helpers corresponding to the SAM training code.

These functions operate on an already-created model, optimizer and iterable
of ``(images, labels)`` batches. Dataset construction and model architecture
remain in their existing modules. SAM's parameter perturbation/restoration is
delegated to :class:`sam_pytorch.optimizers.SAM`.
"""

from __future__ import annotations

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

from sam_pytorch.optimizers import SAM


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
                        max_steps: Optional[int] = None,
                        freeze_batch_norm_statistics: bool = False,
                        **step_kwargs):
    """Train for one epoch, average batch metrics, and optionally log them.

    ``step`` is the current global update count. The scheduler advances once
    after each optimizer update; the returned step is the updated count.
    """
    model.train()
    if freeze_batch_norm_statistics:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    start_time = time.time()
    totals: dict[str, float] = {}
    count = 0
    for batch in islice(batches, max_steps):
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


def train(model: nn.Module, data, config: dict, optimizer_name: str,
          run_dir: str | Path) -> dict:
    """Train with automatic epoch checkpoint recovery and write run reports."""
    from sam_pytorch.optimizers import make_sgd

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    training = config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    optimizer_args = {
        "lr": float(training["learning_rate"]),
        "momentum": float(training.get("momentum", 0.0)),
        # The reference implements weight decay as an L2 loss penalty in train_step.
        "weight_decay": 0.0,
        "nesterov": bool(training.get("nesterov", False)),
    }
    if optimizer_name == "sam":
        optimizer = SAM(model.parameters(), base_optimizer=torch.optim.SGD,
                        rho=float(config.get("sam", {}).get("rho", 0.05)), **optimizer_args)
    elif optimizer_name == "sgd":
        optimizer = make_sgd(model.parameters(), **optimizer_args)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    epochs, max_steps = training.get("epochs"), training.get("max_steps")
    if (epochs is None) == (max_steps is None):
        raise ValueError("Set exactly one of training.epochs or training.max_steps")
    batch_size = int(training["batch_size"])
    steps_per_epoch = max(1, data.num_training_obs // batch_size)
    epochs = int(epochs) if epochs is not None else (int(max_steps) + steps_per_epoch - 1) // steps_per_epoch
    scheduler = make_learning_rate_scheduler(
        optimizer, training.get("scheduler", "constant"), epochs,
        data.num_training_obs, batch_size,
        warmup_epochs=float(training.get("warmup_epochs", 0)))

    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    history, started_at, elapsed_before, epoch_seconds = [], datetime.now(timezone.utc).isoformat(), 0.0, 0.0
    best_error_rate, best_epoch, step, first_epoch = float("inf"), 0, 0, 1
    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        if "torch_rng" in state:
            torch.set_rng_state(state["torch_rng"])
            random.setstate(state["python_rng"])
            np.random.set_state(state["numpy_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if "history" in state:
            history, started_at = state["history"], state["started_at"]
            elapsed_before, epoch_seconds = state["elapsed_seconds"], state["epoch_seconds"]
            best_error_rate, best_epoch = state["best_error_rate"], state["best_epoch"]
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
            best_epoch = int(torch.load(best_path, map_location="cpu", weights_only=False)["epoch"])
            best_error_rate = next(row["val_error_rate"] for row in history
                                   if row["epoch"] == best_epoch)
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

    for epoch in range(first_epoch, epochs + 1):
        tick = time.time()
        batches = data.get_train_loader(seed=int(config.get("seed", 1)) + epoch)
        remaining = None if max_steps is None else max(0, int(max_steps) - step)
        train_metrics, step = train_for_one_epoch(
            model, optimizer, batches, step=step, scheduler=scheduler, max_steps=remaining,
            freeze_batch_norm_statistics=bool(training.get("freeze_batch_norm_statistics", False)),
            label_smoothing=float(training.get("label_smoothing", 0.0)),
            l2_reg=float(training.get("weight_decay", 0.0)),
            gradient_clip_norm=training.get("gradient_clip_norm"))
        validation = eval_on_dataset(model, data.get_validation_loader(),
                                     label_smoothing=float(training.get("label_smoothing", 0.0)))
        duration = time.time() - tick
        epoch_seconds += duration
        row = {"epoch": epoch, "step": step, "learning_rate": optimizer.param_groups[0]["lr"],
               "train_loss": train_metrics.get("train_loss", 0.0),
               "train_error_rate": train_metrics.get("train_error_rate", 0.0),
               "validation_loss": validation["loss"],
               "val_error_rate": validation["error_rate"],
               "epoch_seconds": duration}
        history.append(row)
        if row["val_error_rate"] < best_error_rate:
            best_error_rate, best_epoch = row["val_error_rate"], epoch
            _atomic_save({"model": model.state_dict(), "epoch": epoch}, best_path)
        _atomic_save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "history": history, "started_at": started_at, "elapsed_seconds": elapsed,
            "epoch_seconds": epoch_seconds, "best_error_rate": best_error_rate,
            "best_epoch": best_epoch, "step": step, "epoch": epoch,
        }, last_path)
        with (run_dir / "train.log").open("a") as log:
            log.write("epoch={epoch} step={step} lr={learning_rate:.8g} train_loss={train_loss:.4f} train_error_rate={train_error_rate:.4f} val_loss={validation_loss:.4f} val_error_rate={val_error_rate:.4f} seconds={epoch_seconds:.1f}\n".format(**row))
        with (run_dir / "metrics.csv").open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        logging.info("epoch=%d step=%d val_error_rate=%.4f", epoch, step, row["val_error_rate"])
        if max_steps is not None and step >= int(max_steps):
            break

    if not best_path.exists():
        raise RuntimeError("Training ended without producing best.pt")
    # Final train/test metrics are measured using the selected best.pt weights.
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    train_loader = (data.get_train_loader(use_augmentations=False, seed=int(config.get("seed", 1)))
                    if hasattr(data, "_train_augmented")
                    else data.get_train_loader(seed=int(config.get("seed", 1))))
    validation_loader, test_loader = data.get_validation_loader(), data.get_test_loader()
    smooth = float(training.get("label_smoothing", 0.0))
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
    lines.extend(["", "[checkpoint_selection]", "criterion: lowest validation error rate",
                  f"best_epoch: {best_epoch}", f"best_validation_error_rate: {best_error_rate:.6f}",
                  "best_checkpoint: best.pt", "last_checkpoint: last.pt",
                  "final_evaluation_checkpoint: best.pt", "", "[train_metrics]"])
    lines.extend(f"{key}: {value:.6f}" for key, value in train_final.items())
    lines.extend(["", "[train_classification_report]", train_report, "", "[test_metrics]"])
    lines.extend(f"{key}: {value:.6f}" for key, value in test_final.items())
    lines.extend(["", "[test_classification_report]", test_report])
    (run_dir / "results.txt").write_text("\n".join(lines) + "\n")
    (run_dir / "metrics.txt").unlink(missing_ok=True)
    return {"history": history, "best_epoch": best_epoch,
            "train_metrics": train_final, "test_metrics": test_final}
