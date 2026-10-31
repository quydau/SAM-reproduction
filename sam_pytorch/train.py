"""Train WRN-28-10 on CIFAR-10 from a YAML experiment config."""

import argparse
from pathlib import Path

from sam_pytorch.datasets import CIFAR10DataModule
from sam_pytorch.models import create_model
from sam_pytorch.trainer import load_config, seed_everything, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cifar10_wrn28x10.yaml")
    parser.add_argument("--optimizer", choices=("sgd", "sam"), required=True,
                        help="Optimizer for this run")
    parser.add_argument("--seed", type=int, help="Override config.seed")
    parser.add_argument("--augmentation", choices=("basic", "cutout", "autoaugment"))
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument("--epochs", type=int, help="Override training.epochs (quick run)")
    limits.add_argument("--steps", type=int, help="Override training.max_steps (quick run)")
    args = parser.parse_args()

    config = load_config(args.config)
    if config["dataset"] != "cifar10" or config["model"] != "wrn28x10":
        raise ValueError("train.py requires a CIFAR-10 WRN-28-10 config")
    if args.seed is not None:
        config["seed"] = args.seed
    if args.augmentation is not None:
        config["augmentation"] = args.augmentation
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
        config["training"]["max_steps"] = None
    if args.steps is not None:
        config["training"]["max_steps"] = args.steps
        config["training"]["epochs"] = None

    training = config["training"]
    aug = config.get("augmentation_params", {})
    split = config.get("split", {})
    data = CIFAR10DataModule(
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
    run_dir = (Path(config.get("output_root", "outputs")) / "cifar10"
               / config["model"] / config["augmentation"] / args.optimizer
               / f"seed_{config['seed']}")
    train(model, data, config, args.optimizer, run_dir)


if __name__ == "__main__":
    main()
