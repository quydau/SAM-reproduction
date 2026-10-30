"""Shared training and evaluation loop for CIFAR-10 and fine-tuning runs."""

from __future__ import annotations

import math
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import classification_report, confusion_matrix

from sam_pytorch.optimizers import SAM, make_sgd


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    """Prefer CUDA, then Apple's Metal Performance Shaders, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_optimizer(model: nn.Module, config: dict, name: str):
    training = config["training"]
    options = {
        "lr": float(training["learning_rate"]),
        "momentum": float(training.get("momentum", 0.0)),
        "weight_decay": float(training.get("weight_decay", 0.0)),
        "nesterov": bool(training.get("nesterov", False)),
    }
    if name == "sgd":
        return make_sgd(model.parameters(), **options)
    if name == "sam":
        return SAM(model.parameters(), torch.optim.SGD,
                   rho=float(config["sam"]["rho"]), **options)
    raise ValueError(f"Unsupported optimizer: {name}")


def _set_batch_norm(model: nn.Module, training: bool) -> None:
    """Set BatchNorm mode without changing dropout or other module modes."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.train(training)


def evaluate(model: nn.Module, loader, device: torch.device,
             criterion: nn.Module) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = count = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += labels.size(0)
    return {"loss": loss_sum / max(count, 1),
            "accuracy": correct / max(count, 1)}


def evaluate_classification(model: nn.Module, loader, device: torch.device,
                            criterion: nn.Module, num_classes: int,
                            class_names: list[str]) -> dict:
    """Collect per-example predictions for a final classification report."""
    model.eval()
    loss_sum = count = 0
    targets, predictions = [], []
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            predicted = logits.argmax(dim=1)
            targets.extend(labels.cpu().tolist())
            predictions.extend(predicted.cpu().tolist())
            count += labels.size(0)
    labels = list(range(num_classes))
    report = classification_report(
        targets, predictions, labels=labels, target_names=class_names,
        digits=4, zero_division=0)
    matrix = confusion_matrix(targets, predictions, labels=labels)
    accuracy = sum(a == b for a, b in zip(targets, predictions)) / max(count, 1)
    return {"loss": loss_sum / max(count, 1), "accuracy": accuracy,
            "error": 1.0 - accuracy, "report": report, "confusion_matrix": matrix}


def _write_run_report(path: Path, *, config: dict, optimizer_name: str,
                      device: torch.device, started_at: str, ended_at: str,
                      elapsed_seconds: float,
                      history: list[dict], best_epoch: int, best_validation: dict,
                      test: dict, data, status: str) -> None:
    classes = getattr(data, "class_names", [str(i) for i in range(config["num_classes"])])
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        device_name = "Apple Metal Performance Shaders (MPS)"
    else:
        device_name = platform.processor() or platform.machine() or "CPU"
    text = [
        f"experiment: {config.get('experiment', 'N/A')}",
        f"dataset: {config.get('dataset', 'N/A')}", f"model: {config.get('model', 'N/A')}",
        f"augmentation: {config.get('augmentation', 'N/A')}",
        f"optimizer: {optimizer_name}", f"seed: {config.get('seed', 'N/A')}",
        f"started_at_utc: {started_at}", f"ended_at_utc: {ended_at}", f"status: {status}",
        f"training_epoch_seconds_total: {sum(row['epoch_seconds'] for row in history):.2f}",
        f"run_elapsed_seconds: {elapsed_seconds:.2f}",
        "", "[environment]", f"python: {platform.python_version()}",
        f"pytorch: {torch.__version__}",
        f"torchvision: {__import__('torchvision').__version__}",
        f"device: {device} ({device_name})", f"data_root: {config.get('data_root', 'N/A')}",
        f"pretrained_weights: {config.get('pretrained_weights', 'N/A')}",
        "", "[hyperparameters]", yaml.safe_dump(config, sort_keys=False).rstrip(),
        "", "[data]", f"train_samples: {getattr(data, 'num_training_obs', 'N/A')}",
        f"validation_samples: {len(data.get_validation_loader().dataset)}",
        f"test_samples: {len(data.get_test_loader().dataset)}",
        f"num_classes: {config['num_classes']}",
        "classes_by_index: " + ", ".join(f"{i}={name}" for i, name in enumerate(classes)),
        "", "[epoch_metrics]", "epoch,step,learning_rate,train_loss,train_accuracy,validation_loss,validation_accuracy,epoch_seconds",
    ]
    text.extend(",".join(str(row[key]) for key in (
        "epoch", "step", "learning_rate", "train_loss", "train_accuracy",
        "validation_loss", "validation_accuracy", "epoch_seconds")) for row in history)
    text += ["", "[checkpoint_selection]", "criterion: highest validation accuracy",
             f"best_epoch: {best_epoch}", f"best_validation_loss: {best_validation['loss']:.6f}",
             f"best_validation_accuracy: {best_validation['accuracy']:.6f}",
             f"best_validation_error: {1.0-best_validation['accuracy']:.6f}",
             f"best_checkpoint: best.pt (epoch {best_epoch})", f"last_checkpoint: last.pt (epoch {history[-1]['epoch']})",
             "", "[test_metrics]", f"test_loss: {test['loss']:.6f}",
             f"test_accuracy: {test['accuracy']:.6f}", f"test_error: {test['error']:.6f}",
             f"test_error_percent: {100.0*test['error']:.2f}%", "",
             "[test_classification_report]", test["report"],
             "[test_confusion_matrix] rows=true class, columns=predicted class",
             "true_class," + ",".join(classes)]
    text.extend(f"{classes[i]}," + ",".join(str(int(value)) for value in row)
                for i, row in enumerate(test["confusion_matrix"]))
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def _write_progress_report(path: Path, *, config: dict, optimizer_name: str,
                           epoch: int, step: int, row: dict,
                           best_epoch: int, best_accuracy: float,
                           elapsed_seconds: float) -> None:
    """Persist the latest completed epoch so progress survives interruption."""
    training = config["training"]
    target = (f"{training['epochs']} epochs" if training.get("max_steps") is None
              else f"{training['max_steps']} steps")
    text = [
        f"experiment: {config.get('experiment', 'N/A')}",
        f"dataset: {config.get('dataset', 'N/A')}",
        f"model: {config.get('model', 'N/A')}",
        f"augmentation: {config.get('augmentation', 'N/A')}",
        f"optimizer: {optimizer_name}",
        f"seed: {config.get('seed', 'N/A')}",
        "status: running",
        f"target: {target}",
        f"completed_epoch: {epoch}",
        f"completed_step: {step}",
        f"latest_train_loss: {row['train_loss']:.6f}",
        f"latest_train_accuracy: {row['train_accuracy']:.6f}",
        f"latest_validation_loss: {row['validation_loss']:.6f}",
        f"latest_validation_accuracy: {row['validation_accuracy']:.6f}",
        f"best_epoch: {best_epoch}",
        f"best_validation_accuracy: {best_accuracy:.6f}",
        f"elapsed_seconds: {elapsed_seconds:.2f}",
    ]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def train_epoch(model: nn.Module, loader, optimizer, criterion: nn.Module,
                device: torch.device, accumulation_steps: int,
                gradient_clip_norm: float | None,
                freeze_batch_norm: bool,
                max_updates: int | None = None) -> tuple[dict[str, float], int]:
    model.train()
    if freeze_batch_norm:
        _set_batch_norm(model, False)
    loss_sum = 0.0
    correct = count = updates = 0
    batches = iter(loader)
    while max_updates is None or updates < max_updates:
        group = []
        for _ in range(accumulation_steps):
            try:
                group.append(next(batches))
            except StopIteration:
                break
        if not group:
            break
        sample_count = sum(labels.size(0) for _, labels in group)

        optimizer.zero_grad(set_to_none=True)
        is_sam = isinstance(optimizer, SAM)
        # Cache one effective batch on CPU so both SAM passes see identical
        # samples and augmentation, while only one microbatch is on the GPU.
        for images, labels in group:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            (loss * (labels.size(0) / sample_count)).backward()
            loss_sum += loss.detach().item() * labels.size(0)
            correct += (logits.detach().argmax(dim=1) == labels).sum().item()
            count += labels.size(0)
        if is_sam:
            optimizer.first_step(zero_grad=True)
            # Match the reference SAM example: update BN statistics only on
            # the first pass, and use the same minibatches for the second.
            if not freeze_batch_norm:
                _set_batch_norm(model, False)
            for images, labels in group:
                images, labels = images.to(device), labels.to(device)
                (criterion(model(images), labels)
                 * (labels.size(0) / sample_count)).backward()
            if gradient_clip_norm is not None:
                clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.second_step(zero_grad=True)
        else:
            if gradient_clip_norm is not None:
                clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
        updates += 1
        if not freeze_batch_norm:
            model.train()

    return {"loss": loss_sum / max(count, 1),
            "accuracy": correct / max(count, 1)}, updates


def fit(model: nn.Module, data, config: dict, optimizer_name: str,
        output_dir: str | Path | None = None,
        device: str | torch.device | None = None) -> list[dict]:
    """Train a model with the YAML settings and return per-epoch metrics."""
    training = config["training"]
    seed_everything(int(config["seed"]))
    device = torch.device(device) if device is not None else select_device()
    model.to(device)
    optimizer = make_optimizer(model, config, optimizer_name)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(training.get("label_smoothing", 0.0)))

    batch_size = int(training["batch_size"])
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    effective = int(training.get("effective_batch_size", batch_size * accumulation))
    if batch_size * accumulation != effective:
        raise ValueError("batch_size * gradient_accumulation_steps must equal effective_batch_size")
    max_steps = training.get("max_steps")
    epochs = training.get("epochs")
    if max_steps is None and epochs is None:
        raise ValueError("Set training.epochs or training.max_steps")
    if max_steps is not None and int(max_steps) < 1:
        raise ValueError("training.max_steps must be positive")
    if epochs is not None and int(epochs) < 1:
        raise ValueError("training.epochs must be positive")

    train_loader = data.get_train_loader(seed=int(config["seed"]))
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_steps = (int(max_steps) if max_steps is not None else
                   int(epochs) * updates_per_epoch)
    scheduler_name = str(training.get("scheduler", "none")).lower()
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps)
    elif scheduler_name in ("none", "constant"):
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    freeze_bn = bool(training.get("freeze_batch_norm_statistics", False))
    clip = training.get("gradient_clip_norm")
    clip = None if clip is None else float(clip)
    history = []
    step = epoch = 0
    output_dir = Path(output_dir) if output_dir else None
    started_at = datetime.now(timezone.utc).isoformat()
    run_started = time.perf_counter()
    best_accuracy = float("-inf")
    best_epoch = 0
    best_validation = None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
        with (output_dir / "train.log").open("w", encoding="utf-8") as log_file:
            log_file.write(
                f"run_started_utc: {started_at}\noptimizer: {optimizer_name}\n"
                f"seed: {config.get('seed', 'N/A')}\ndevice: {device}\n\n")
    while (step < int(max_steps) if max_steps is not None else epoch < int(epochs)):
        epoch += 1
        started = time.perf_counter()
        remaining = (int(max_steps) - step) if max_steps is not None else None
        train_metrics, updates = train_epoch(
            model, train_loader, optimizer, criterion, device, accumulation,
            clip, freeze_bn, max_updates=remaining)
        # Scheduler advances once per optimizer update, including partial
        # final epochs when max_steps stops a run early.
        completed_updates = min(updates, total_steps - step)
        for _ in range(completed_updates):
            if scheduler is not None:
                scheduler.step()
        step += completed_updates
        validation = evaluate(model, data.get_validation_loader(), device, criterion)
        row = {"epoch": epoch, "step": step,
               "learning_rate": optimizer.param_groups[0]["lr"],
               "train_loss": train_metrics["loss"],
               "train_accuracy": train_metrics["accuracy"],
               "validation_loss": validation["loss"],
               "validation_accuracy": validation["accuracy"],
               "epoch_seconds": time.perf_counter() - started}
        history.append(row)
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            best_epoch = epoch
            best_validation = validation
            if output_dir:
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict() if scheduler else None,
                            "epoch": epoch, "step": step, "config": config,
                            "validation": validation}, output_dir / "best.pt")
        log_line = ("epoch={epoch} step={step} lr={learning_rate:.6g} "
                    "train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
                    "val_loss={validation_loss:.4f} val_acc={validation_accuracy:.4f} "
                    "seconds={epoch_seconds:.1f}".format(**row))
        print(log_line, flush=True)
        if output_dir:
            with (output_dir / "train.log").open("a", encoding="utf-8") as log_file:
                log_file.write(log_line + "\n")
            output_dir.mkdir(parents=True, exist_ok=True)
            with (output_dir / "metrics.csv").open("w", encoding="utf-8") as f:
                import csv
                writer = csv.DictWriter(f, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler else None,
                        "epoch": epoch, "step": step, "config": config},
                       output_dir / "last.pt")
            _write_progress_report(
                output_dir / "metrics.txt", config=config,
                optimizer_name=optimizer_name, epoch=epoch, step=step, row=row,
                best_epoch=best_epoch, best_accuracy=best_accuracy,
                elapsed_seconds=time.perf_counter() - run_started)
    if output_dir:
        # Test is evaluated exactly once, using the checkpoint selected on validation.
        checkpoint = torch.load(output_dir / "best.pt", map_location=device,
                                weights_only=False)
        model.load_state_dict(checkpoint["model"])
        test = evaluate_classification(
            model, data.get_test_loader(), device, criterion,
            int(config["num_classes"]),
            getattr(data, "class_names", [str(i) for i in range(config["num_classes"])]))
        ended_at = datetime.now(timezone.utc).isoformat()
        _write_run_report(
            output_dir / "metrics.txt", config=config, optimizer_name=optimizer_name,
            device=device, started_at=started_at, ended_at=ended_at,
            elapsed_seconds=time.perf_counter() - run_started,
            history=history, best_epoch=best_epoch, best_validation=best_validation,
            test=test, data=data, status="completed")
        with (output_dir / "train.log").open("a", encoding="utf-8") as log_file:
            log_file.write(f"run_completed_utc: {ended_at}\n")
        # Keep plan.md's canonical filename while also writing the requested metrics.txt.
        (output_dir / "results.txt").write_text(
            (output_dir / "metrics.txt").read_text(encoding="utf-8"), encoding="utf-8")
    return history
