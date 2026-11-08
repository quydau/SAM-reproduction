"""Fine-tune EfficientNet models on supported image classification datasets."""

import argparse
import copy
import itertools
import logging
import time
from pathlib import Path

from sam_pytorch.datasets import (FGVCAircraftDataModule,
                                  Food101DataModule,
                                  Flowers102DataModule,
                                  OxfordIIITPetDataModule,
                                  CIFAR10DataModule,
                                  CIFAR100DataModule,
                                  StanfordCarsDataModule)
from sam_pytorch.models import create_model, uses_timm_pretrained_weights
from sam_pytorch.train_utils import load_config, seed_everything, train


def run(config, optimizer, seed, args) -> None:
    config = copy.deepcopy(config)
    config["seed"] = seed
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
    seed_everything(int(config["seed"]))
    model = create_model(config["model"], int(config["num_classes"]),
                         config.get("pretrained_weights"))
    if uses_timm_pretrained_weights(
            config["model"], config.get("pretrained_weights")):
        from timm.data import resolve_model_data_config

        model_data = resolve_model_data_config(model)
        image_size = int(config.get("image", {}).get("crop_size", model_data["input_size"][-1]))
        expected_size = int(model_data["input_size"][-1])
        if image_size != expected_size:
            raise ValueError(
                f"{config['model']} expects image size {expected_size}, got {image_size}")
        common.update(image_size=image_size,
                      image_mean=model_data["mean"],
                      image_std=model_data["std"])
    if config["dataset"] == "cifar10":
        data = CIFAR10DataModule(
            data_root=config["data_root"],
            batch_size=int(training["batch_size"]),
            augmentation=config.get("augmentation", "none"),
            validation_seed=int(split.get("seed", config["seed"])),
            num_workers=int(training.get("num_workers", 0)))
    elif config["dataset"] == "cifar100":
        # Keep the existing CIFAR-100 split/loader; this run only adds a model.
        data = CIFAR100DataModule(
            data_root=config["data_root"],
            batch_size=int(training["batch_size"]),
            augmentation=config.get("augmentation", "none"),
            validation_seed=int(split.get("seed", config["seed"])),
            num_workers=int(training.get("num_workers", 0)))
    elif config["dataset"] == "flowers102":
        data = Flowers102DataModule(**common)
    elif config["dataset"] == "food101":
        data = Food101DataModule(**common)
    elif config["dataset"] == "oxford_iiit_pet":
        data = OxfordIIITPetDataModule(**common)
    elif config["dataset"] == "fgvc_aircraft":
        data = FGVCAircraftDataModule(**common)
    elif config["dataset"] == "stanford_cars":
        data = StanfordCarsDataModule(**common)
    else:
        raise ValueError(f"Unsupported fine-tuning dataset: {config['dataset']}")

    run_dir = (Path(config.get("output_root", "outputs"))
               / config["dataset"] / config["model"] / optimizer
               / f"seed_{config['seed']}")
    train(model, data, config, optimizer, run_dir, log_every=args.log_every)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True,
                        help="Fine-tuning YAML config")
    parser.add_argument("--optimizer", nargs="+", choices=("sgd", "sam"), required=True,
                        help="One or more optimizers; each runs in turn")
    parser.add_argument("--seed", nargs="+", type=int,
                        help="One or more seeds (default: config.seed)")
    parser.add_argument("--log-every", type=int, default=10,
                        help="Log progress every N epochs (default: 10)")
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument("--epochs", type=int, help="Override training.epochs (quick run)")
    limits.add_argument("--steps", type=int, help="Override training.max_steps (quick run)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    config = load_config(args.config)
    if config["model"] not in ("efficientnet_b0", "efficientnet_b3_jft",
                                "efficientnet_b4_jft"):
        raise ValueError("finetune.py requires an EfficientNet-B0, B3-JFT or B4-JFT config")

    seeds = args.seed or [int(config["seed"])]
    combos = list(itertools.product(
        dict.fromkeys(seeds), dict.fromkeys(args.optimizer)))
    for i, (seed, optimizer) in enumerate(combos, 1):
        print(f"[{i}/{len(combos)}] START seed={seed} optimizer={optimizer}", flush=True)
        start = time.time()
        run(config, optimizer, seed, args)
        print(f"[{i}/{len(combos)}] DONE seed={seed} optimizer={optimizer} "
              f"in {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
