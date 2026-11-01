"""Train WRN-16-8 on CIFAR-10/100 from a YAML experiment config."""

import argparse
import copy
import itertools
import logging
from pathlib import Path

from sam_pytorch.datasets import CIFAR10DataModule, CIFAR100DataModule
from sam_pytorch.models import create_model
from sam_pytorch.train_utils import load_config, seed_everything, train


DATA_MODULES = {"cifar10": CIFAR10DataModule, "cifar100": CIFAR100DataModule}


def run(config, optimizer, augmentation, seed, args) -> None:
    config = copy.deepcopy(config)
    config["seed"] = seed
    if augmentation is not None:
        config["augmentation"] = augmentation
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
        config["training"]["max_steps"] = None
    if args.steps is not None:
        config["training"]["max_steps"] = args.steps
        config["training"]["epochs"] = None

    training = config["training"]
    aug = config.get("augmentation_params", {})
    split = config.get("split", {})
    data = DATA_MODULES[config["dataset"]](
        data_root=config["data_root"], batch_size=int(training["batch_size"]),
        augmentation=config["augmentation"],
        validation_seed=int(split.get("seed", config["seed"])),
        num_workers=int(training.get("num_workers", 0)),
        cutout_length=int(aug.get("cutout_length", 16)),
        crop_padding=int(aug.get("crop_padding", 4)),
        horizontal_flip_probability=float(
            aug.get("horizontal_flip_probability", 0.5)))

    seed_everything(int(config["seed"]))
    model = create_model(config["model"], int(config["num_classes"]))
    run_dir = (Path(config.get("output_root", "outputs")) / config["dataset"]
               / config["model"] / config["augmentation"] / optimizer
               / f"seed_{config['seed']}")
    train(model, data, config, optimizer, run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cifar10_wrn16x8.yaml")
    parser.add_argument("--optimizer", nargs="+", choices=("sgd", "sam"), required=True,
                        help="One or more optimizers; each runs in turn")
    parser.add_argument("--seed", nargs="+", type=int,
                        help="One or more seeds (default: config.seed)")
    parser.add_argument("--augmentation", nargs="+",
                        choices=("basic", "cutout", "autoaugment"),
                        help="One or more augmentations (default: config.augmentation)")
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument("--epochs", type=int, help="Override training.epochs (quick run)")
    limits.add_argument("--steps", type=int, help="Override training.max_steps (quick run)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    config = load_config(args.config)
    if config["dataset"] not in DATA_MODULES or config["model"] != "wrn16x8":
        raise ValueError("train.py requires a CIFAR-10/100 WRN-16-8 config")

    seeds = args.seed or [int(config["seed"])]
    augmentations = args.augmentation or [None]
    # dict.fromkeys drops duplicates while keeping order.
    combos = list(itertools.product(
        dict.fromkeys(seeds), dict.fromkeys(augmentations), dict.fromkeys(args.optimizer)))
    for i, (seed, augmentation, optimizer) in enumerate(combos, 1):
        print(f"[{i}/{len(combos)}] seed={seed} augmentation="
              f"{augmentation or config['augmentation']} optimizer={optimizer}", flush=True)
        run(config, optimizer, augmentation, seed, args)


if __name__ == "__main__":
    main()
