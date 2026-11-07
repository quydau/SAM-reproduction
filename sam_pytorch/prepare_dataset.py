"""Download/cache a fine-tuning dataset and its pretrained model weights."""

import argparse

from sam_pytorch.datasets import (CIFAR10DataModule, Food101DataModule,
                                  Flowers102DataModule)
from sam_pytorch.models import create_model
from sam_pytorch.train_utils import load_config


CONFIGS = {
    "flowers102": "configs/flowers102_efficientnet_b3_jft.yaml",
    "food101": "configs/food101_efficientnet_b0.yaml",
    "cifar10": "configs/cifar10_efficientnet_b0.yaml",
}
DATA_MODULES = {
    "flowers102": Flowers102DataModule,
    "food101": Food101DataModule,
    "cifar10": CIFAR10DataModule,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(CONFIGS), default="flowers102",
                        help="Dataset to prepare (default: flowers102)")
    args = parser.parse_args()
    config = load_config(CONFIGS[args.dataset])

    # Populate the model-weight cache before two training processes start.
    model = create_model(config["model"], config["num_classes"],
                         config["pretrained_weights"])
    model_data = None
    if config["model"].endswith("_jft"):
        from timm.data import resolve_model_data_config

        model_data = resolve_model_data_config(model)
    del model

    training = config["training"]
    split = config.get("split", {})
    if args.dataset == "cifar10":
        data = CIFAR10DataModule(
            data_root=config["data_root"],
            batch_size=int(training["batch_size"]),
            augmentation=config.get("augmentation", "none"),
            validation_seed=int(split.get("seed", config["seed"])),
            num_workers=int(training.get("num_workers", 0)),
        )
    else:
        datamodule = DATA_MODULES[args.dataset]
        image = config.get("image", {})
        kwargs = dict(
            data_root=config["data_root"],
            batch_size=int(training["batch_size"]),
            validation_fraction=float(split["validation_fraction"]),
            split_seed=int(split["seed"]),
            image_size=int(image.get("crop_size", 224)),
        )
        if model_data is not None:
            kwargs.update(image_mean=model_data["mean"], image_std=model_data["std"])
        data = datamodule(**kwargs)
    # These calls synchronously download/extract any missing data and create
    # any uint8 tensor caches. Training processes can then read them safely.
    data.get_train_loader(seed=config["seed"])
    data.get_validation_loader()
    data.get_test_loader()
    print(f"{args.dataset} and its pretrained model weights are ready.", flush=True)


if __name__ == "__main__":
    main()
