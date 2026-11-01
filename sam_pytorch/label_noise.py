"""SAM paper §3.3: ResNet-32 on CIFAR-10 with noisy labels (Table 4).

Methods:
  sgd            plain SGD on the noisy labels
  sam            SAM on the noisy labels (rho from sam.rho_by_noise_rate)
  bootstrap_sam  take the finished ``sam`` run for the same noise rate and
                 seed (training it first if missing), predict the 45k training
                 labels with it, then retrain SAM from scratch on those labels

Examples:
  python -m sam_pytorch.label_noise --config configs/cifar10_noisy_resnet32.yaml \\
      --method sgd sam bootstrap_sam --noise-rate 0.2 0.4 0.6 0.8 --seed 1
  python -m sam_pytorch.label_noise --config configs/cifar10_noisy_resnet32.yaml --summarize
"""

import argparse
import copy
import itertools
import logging
import re
from pathlib import Path

import torch

from sam_pytorch.datasets import CIFAR10DataModule
from sam_pytorch.models import create_model
from sam_pytorch.train_utils import load_config, seed_everything, train

METHODS = ("sgd", "sam", "bootstrap_sam")
# Table 4 of the SAM paper (test accuracy %, clean test set).
PAPER_TABLE4 = {
    "sgd": {0.2: 84.8, 0.4: 68.8, 0.6: 48.2, 0.8: 26.2},
    "sam": {0.2: 95.1, 0.4: 93.4, 0.6: 90.5, 0.8: 77.9},
    "bootstrap_sam": {0.2: 95.4, 0.4: 94.2, 0.6: 91.8, 0.8: 79.9},
}


def _rate_key(rate: float) -> float:
    return round(float(rate), 4)


def run_dir_for(config: dict, rate: float, method: str, seed: int) -> Path:
    return (Path(config.get("output_root", "outputs")) / "cifar10_noisy"
            / config["model"] / f"noise_{round(rate * 100)}" / method
            / f"seed_{seed}")


def _rho_for(config: dict, rate: float) -> float:
    table = {_rate_key(k): float(v)
             for k, v in config["sam"]["rho_by_noise_rate"].items()}
    if _rate_key(rate) not in table:
        raise ValueError(f"No rho configured for noise rate {rate}")
    return table[_rate_key(rate)]


def _make_data(config: dict, rate: float, seed: int, train_labels=None):
    training = config["training"]
    aug = config.get("augmentation_params", {})
    split = config.get("split", {})
    return CIFAR10DataModule(
        data_root=config["data_root"], batch_size=int(training["batch_size"]),
        augmentation=config["augmentation"],
        validation_seed=int(split.get("seed", seed)),
        num_workers=int(training.get("num_workers", 0)),
        crop_padding=int(aug.get("crop_padding", 4)),
        horizontal_flip_probability=float(aug.get("horizontal_flip_probability", 0.5)),
        label_noise_rate=rate,
        noise_seed=int(config["label_noise"].get("noise_seed_offset", 1000)) + seed,
        train_labels_override=train_labels)


def _run_config(base: dict, rate: float, method: str, seed: int, args) -> dict:
    config = copy.deepcopy(base)
    config["seed"] = seed
    config["label_noise"]["rate"] = rate
    config["label_noise"]["noise_seed"] = (
        int(config["label_noise"].get("noise_seed_offset", 1000)) + seed)
    config["method"] = method
    # SGD ignores rho; record it as null so its config is not misleading.
    config.setdefault("sam", {})["rho"] = None if method == "sgd" else _rho_for(config, rate)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
        config["training"]["max_steps"] = None
    if args.steps is not None:
        config["training"]["max_steps"] = args.steps
        config["training"]["epochs"] = None
    return config


def _is_complete(run_dir: Path) -> bool:
    return (run_dir / "results.txt").exists() and (run_dir / "last.pt").exists()


def _train_on_noisy(base, rate, method, seed, args) -> Path:
    """Train ``sgd`` or ``sam`` on the noisy labels; return the run dir."""
    config = _run_config(base, rate, method, seed, args)
    run_dir = run_dir_for(config, rate, method, seed)
    data = _make_data(config, rate, seed)
    config["label_noise"]["num_wrong_train_labels"] = data.num_wrong_train_labels
    config["label_noise"]["num_train_labels"] = len(data.train_labels)
    seed_everything(seed)
    model = create_model(config["model"], int(config["num_classes"]))
    train(model, data, config, method, run_dir)
    return run_dir


@torch.no_grad()
def _predict_train_labels(model, data) -> list[int]:
    device = next(model.parameters()).device
    model.eval()
    predictions = []
    for images, _ in data.get_train_prediction_loader():
        predictions.extend(model(images.to(device)).argmax(-1).cpu().tolist())
    return predictions


def _bootstrap_sam(base, rate, seed, args) -> Path:
    stage1_dir = run_dir_for(base, rate, "sam", seed)
    if not _is_complete(stage1_dir):
        print(f"  bootstrap: stage-1 SAM run missing at {stage1_dir}; training it first",
              flush=True)
        _train_on_noisy(base, rate, "sam", seed, args)

    config = _run_config(base, rate, "bootstrap_sam", seed, args)
    run_dir = run_dir_for(config, rate, "bootstrap_sam", seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    labels_path = run_dir / "bootstrap_labels.pt"

    noisy_data = _make_data(config, rate, seed)
    if labels_path.exists():
        predicted = torch.load(labels_path)["predicted_labels"]
    else:
        # "the model is first trained as usual and then retrained from scratch
        # on the labels predicted by the initially trained model" (§3.3).
        # The stage-1 model is its final epoch, the same model reported for SAM.
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
        stage1 = create_model(config["model"], int(config["num_classes"])).to(device)
        state = torch.load(stage1_dir / "last.pt", map_location=device, weights_only=False)
        stage1.load_state_dict(state["ema"] or state["model"])
        predicted = _predict_train_labels(stage1, noisy_data)
        n = len(predicted)
        agree_clean = sum(p == c for p, c in zip(predicted, noisy_data.clean_train_labels))
        agree_noisy = sum(p == y for p, y in zip(predicted, noisy_data.train_labels))
        torch.save({"predicted_labels": predicted,
                    "train_indices": noisy_data.train_indices,
                    "source_checkpoint": str(stage1_dir / "last.pt")}, labels_path)
        (run_dir / "bootstrap_stats.txt").write_text(
            f"source_checkpoint: {stage1_dir / 'last.pt'}\n"
            f"noise_rate: {rate}\n"
            f"train_labels: {n}\n"
            f"noisy_labels_wrong_vs_clean: {noisy_data.num_wrong_train_labels / n:.6f}\n"
            f"predicted_agree_with_clean: {agree_clean / n:.6f}\n"
            f"predicted_agree_with_noisy: {agree_noisy / n:.6f}\n")
        print(f"  bootstrap labels: {agree_clean / n:.4f} agree with clean labels "
              f"(noisy labels: {1 - noisy_data.num_wrong_train_labels / n:.4f})", flush=True)

    data = _make_data(config, rate, seed, train_labels=predicted)
    config["label_noise"]["num_wrong_train_labels"] = data.num_wrong_train_labels
    config["label_noise"]["num_train_labels"] = len(data.train_labels)
    config["bootstrap_source"] = str(stage1_dir / "last.pt")
    seed_everything(seed)
    model = create_model(config["model"], int(config["num_classes"]))
    train(model, data, config, "sam", run_dir)
    return run_dir


def _test_accuracy(results_path: Path):
    text = results_path.read_text()
    section = text.split("[test_metrics]", 1)
    if len(section) < 2:
        return None
    match = re.search(r"^accuracy:\s*([0-9.]+)", section[1], re.MULTILINE)
    return float(match.group(1)) * 100 if match else None


def summarize(config: dict) -> str:
    root = Path(config.get("output_root", "outputs")) / "cifar10_noisy" / config["model"]
    rates = [float(r) for r in config["label_noise"]["rates"]]
    header = "| Method | " + " | ".join(f"{round(r * 100)}%" for r in rates) + " |"
    lines = ["Test accuracy (%) on the clean CIFAR-10 test set: ours (paper Table 4).",
             "Ours = mean ± std over finished seeds; n = number of seeds.", "",
             header, "|" + "---|" * (len(rates) + 1)]
    for method in METHODS:
        cells = []
        for rate in rates:
            accs = [a for a in (_test_accuracy(p) for p in sorted(
                (root / f"noise_{round(rate * 100)}" / method).glob("seed_*/results.txt")))
                if a is not None]
            paper = PAPER_TABLE4[method][_rate_key(rate)]
            if not accs:
                cells.append(f"– ({paper})")
                continue
            mean = sum(accs) / len(accs)
            std = (sum((a - mean) ** 2 for a in accs) / (len(accs) - 1)) ** 0.5 if len(accs) > 1 else 0.0
            cells.append(f"{mean:.1f} ± {std:.1f}, n={len(accs)} ({paper})")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    table = "\n".join(lines) + "\n"
    root.parent.mkdir(parents=True, exist_ok=True)
    (root.parent / "summary.md").write_text(table)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/cifar10_noisy_resnet32.yaml")
    parser.add_argument("--method", nargs="+", choices=METHODS)
    parser.add_argument("--noise-rate", nargs="+", type=float,
                        help="Noise rates, e.g. 0.2 0.4 (default: label_noise.rates)")
    parser.add_argument("--seed", nargs="+", type=int, help="Seeds (default: config.seed)")
    parser.add_argument("--summarize", action="store_true",
                        help="Only build the Table-4 style summary from finished runs")
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument("--epochs", type=int, help="Override training.epochs (quick run)")
    limits.add_argument("--steps", type=int, help="Override training.max_steps (quick run)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    config = load_config(args.config)
    if args.summarize:
        print(summarize(config))
        return
    if not args.method:
        parser.error("--method is required unless --summarize is given")

    rates = args.noise_rate or [float(r) for r in config["label_noise"]["rates"]]
    seeds = args.seed or [int(config["seed"])]
    combos = list(itertools.product(dict.fromkeys(seeds), dict.fromkeys(rates),
                                    dict.fromkeys(args.method)))
    for i, (seed, rate, method) in enumerate(combos, 1):
        print(f"[{i}/{len(combos)}] seed={seed} noise={rate} method={method} "
              f"rho={_rho_for(config, rate) if method != 'sgd' else 'N/A'}", flush=True)
        if method == "bootstrap_sam":
            _bootstrap_sam(config, rate, seed, args)
        else:
            _train_on_noisy(config, rate, method, seed, args)
    print(summarize(config))


if __name__ == "__main__":
    main()
