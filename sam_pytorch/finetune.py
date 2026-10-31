"""Fine-tune EfficientNet-B0 on Flowers102 or Oxford-IIIT Pet."""

import argparse
from pathlib import Path

from sam_pytorch.datasets import (Flowers102DataModule,
                                  OxfordIIITPetDataModule)
from sam_pytorch.models import create_model
from sam_pytorch.train_utils import load_config, seed_everything, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True,
                        help="Flowers102 or Oxford-IIIT Pet YAML config")
    parser.add_argument("--optimizer", choices=("sgd", "sam"), required=True,
                        help="Optimizer for this run")
    parser.add_argument("--seed", type=int, help="Override config.seed")
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument("--epochs", type=int, help="Override training.epochs (quick run)")
    limits.add_argument("--steps", type=int, help="Override training.max_steps (quick run)")
    args = parser.parse_args()

    config = load_config(args.config)
    if config["model"] != "efficientnet_b0":
        raise ValueError("finetune.py requires an EfficientNet-B0 config")
    if args.seed is not None:
        config["seed"] = args.seed
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
        config["training"]["max_steps"] = None
    if args.steps is not None:
        config["training"]["max_steps"] = args.steps
        config["training"]["epochs"] = None
    training = config["training"]
    split = config.get("split", {})
    common = dict(
        data_root=config["data_root"], batch_size=int(training["batch_size"]),
        validation_fraction=float(split.get("validation_fraction", 0.1)),
        split_seed=int(split.get("seed", config["seed"])),
        num_workers=int(training.get("num_workers", 0)))
    if config["dataset"] == "flowers102":
        data = Flowers102DataModule(**common)
    elif config["dataset"] == "oxford_iiit_pet":
        data = OxfordIIITPetDataModule(**common)
    else:
        raise ValueError(f"Unsupported fine-tuning dataset: {config['dataset']}")

    seed_everything(int(config["seed"]))
    model = create_model(config["model"], int(config["num_classes"]),
                         config.get("pretrained_weights"))
    run_dir = (Path(config.get("output_root", "outputs"))
               / config["dataset"] / config["model"] / args.optimizer
               / f"seed_{config['seed']}")
    train(model, data, config, args.optimizer, run_dir)


if __name__ == "__main__":
    main()
