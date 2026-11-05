"""Download and cache Flowers102 plus the EfficientNet-B3 checkpoint."""

from sam_pytorch.datasets import Flowers102DataModule
from sam_pytorch.models import create_model
from sam_pytorch.train_utils import load_config
from timm.data import resolve_model_data_config


def main() -> None:
    config = load_config("configs/flowers102_efficientnet_b3_jft.yaml")

    # Resolve the checkpoint's input normalization and populate the local
    # timm/Hugging Face weight cache before two training processes start.
    model = create_model(config["model"], config["num_classes"],
                         config["pretrained_weights"])
    model_data = resolve_model_data_config(model)
    del model

    data = Flowers102DataModule(
        data_root=config["data_root"],
        batch_size=config["training"]["batch_size"],
        validation_fraction=config["split"]["validation_fraction"],
        split_seed=config["split"]["seed"],
        image_size=config["image"]["crop_size"],
        image_mean=model_data["mean"],
        image_std=model_data["std"],
    )
    # These calls synchronously download/extract any missing data and create
    # the uint8 tensor caches. Training processes can then read them safely.
    data.get_train_loader(seed=config["seed"])
    data.get_test_loader()
    print("Flowers102 and EfficientNet-B3 checkpoint are ready.", flush=True)


if __name__ == "__main__":
    main()
