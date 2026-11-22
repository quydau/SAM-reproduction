"""Download/cache one supported fine-tuning dataset and EfficientNet-B0 weights."""

import argparse

from sam_pytorch.datasets import (FGVCAircraftDataModule,
                                  Flowers102DataModule,
                                  OxfordIIITPetDataModule,
                                  StanfordCarsDataModule)
from sam_pytorch.models import create_model
from sam_pytorch.train_utils import load_config


CONFIGS = {
    "fgvc_aircraft": ("configs/fgvc_aircraft_efficientnet_b0.yaml",
                      FGVCAircraftDataModule),
    "flowers102": ("configs/flowers102_efficientnet_b0.yaml",
                   Flowers102DataModule),
    "oxford_iiit_pet": ("configs/oxford_iiit_pet_efficientnet_b0.yaml",
                        OxfordIIITPetDataModule),
    "stanford_cars": ("configs/stanford_cars_efficientnet_b0.yaml",
                      StanfordCarsDataModule),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(CONFIGS), required=True)
    args = parser.parse_args()
    config_path, data_module = CONFIGS[args.dataset]
    config = load_config(config_path)

    # Download/cache the pretrained weights before starting training processes.
    model = create_model(config["model"], config["num_classes"],
                         config.get("pretrained_weights"))
    del model

    training = config["training"]
    split = config.get("split", {})
    data = data_module(
        data_root=config["data_root"],
        batch_size=int(training["batch_size"]),
        validation_fraction=float(split.get("validation_fraction", 0.1)),
        split_seed=int(split.get("seed", config["seed"])),
        num_workers=int(training.get("num_workers", 0)),
        image_size=int(config.get("image", {}).get("crop_size", 224)),
    )
    data.get_train_loader(seed=int(config["seed"]))
    data.get_validation_loader()
    data.get_test_loader()
    print(f"{args.dataset} data and EfficientNet-B0 weights are ready.", flush=True)


if __name__ == "__main__":
    main()
