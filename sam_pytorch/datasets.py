"""Dataset splits and PyTorch data loaders for the experiments."""

import json
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, default_collate
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from sam_pytorch.augmentation import CIFAR10Augmentation


class _BatchCollator:
    """Collate image-label pairs and optionally apply batch-level Cutout."""

    def __init__(self, augmentation: CIFAR10Augmentation, enabled: bool):
        self.augmentation = augmentation
        self.enabled = enabled

    def __call__(self, samples):
        images, labels = default_collate(samples)
        if self.enabled:
            images = self.augmentation.apply_cutout(images)
        return images, labels


class CIFAR10DataModule:
    """Reproducible CIFAR-10 train/validation/test loaders."""

    NUM_CLASSES = 10
    TRAIN_SIZE = 45_000
    DATASET_CLS = CIFAR10
    MEAN = CIFAR10Augmentation.MEAN
    STD = CIFAR10Augmentation.STD

    def __init__(self, data_root: str | Path, batch_size: int,
                 augmentation: str = "basic", validation_seed: int = 1,
                 num_workers: int = 4, cutout_length: int = 16,
                 crop_padding: int = 4, pin_memory: Optional[bool] = None,
                 horizontal_flip_probability: float = 0.5):
        if batch_size < 1 or num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers >= 0")

        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = self.NUM_CLASSES
        self.num_training_obs = self.TRAIN_SIZE
        self.pin_memory = (torch.cuda.is_available()
                           if pin_memory is None else pin_memory)
        self.augmentation = CIFAR10Augmentation(
            augmentation, crop_padding=crop_padding,
            cutout_length=cutout_length,
            horizontal_flip_probability=horizontal_flip_probability,
            mean=self.MEAN, std=self.STD)
        plain = CIFAR10Augmentation("none", mean=self.MEAN, std=self.STD)

        self._train_augmented = self.DATASET_CLS(
            root=str(self.data_root), train=True, download=True,
            transform=self.augmentation.image_transform)
        self._train_plain = self.DATASET_CLS(
            root=str(self.data_root), train=True, download=False,
            transform=plain.image_transform)
        self._test = self.DATASET_CLS(
            root=str(self.data_root), train=False, download=True,
            transform=plain.image_transform)
        self.class_names = list(self._test.classes)

        generator = torch.Generator().manual_seed(validation_seed)
        indices = torch.randperm(len(self._train_augmented),
                                 generator=generator).tolist()
        self.train_indices = indices[:self.TRAIN_SIZE]
        self.validation_indices = indices[self.TRAIN_SIZE:]

    def _loader(self, dataset: Dataset, *, shuffle: bool, drop_last: bool,
                seed: int, cutout: bool = False,
                batch_size: Optional[int] = None) -> DataLoader:
        return DataLoader(
            dataset, batch_size=batch_size or self.batch_size,
            shuffle=shuffle, drop_last=drop_last,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            generator=torch.Generator().manual_seed(seed),
            collate_fn=_BatchCollator(self.augmentation, enabled=cutout),
            persistent_workers=self.num_workers > 0)

    def get_train_loader(self, use_augmentations: bool = True, seed: int = 1):
        source = self._train_augmented if use_augmentations else self._train_plain
        dataset = Subset(source, self.train_indices)
        return self._loader(dataset, shuffle=True,
                            drop_last=use_augmentations, seed=seed,
                            cutout=use_augmentations)

    def get_validation_loader(self):
        return self._loader(Subset(self._train_plain, self.validation_indices),
                            shuffle=False, drop_last=False, seed=0)

    def get_test_loader(self):
        return self._loader(
            self._test, shuffle=False, drop_last=False, seed=0,
            batch_size=min(32, self.batch_size))


class CIFAR100DataModule(CIFAR10DataModule):
    """CIFAR-100 loaders with the same split and preprocessing."""

    NUM_CLASSES = 100
    DATASET_CLS = CIFAR100
    MEAN = CIFAR10Augmentation.CIFAR100_MEAN
    STD = CIFAR10Augmentation.CIFAR100_STD


class _ImageNetDataModule:
    """Shared deterministic splits and loaders for fine-tuning datasets."""

    NUM_CLASSES: int

    def __init__(self, data_root: str | Path, batch_size: int,
                 validation_fraction: float = 0.1, split_seed: int = 1,
                 num_workers: int = 4, pin_memory: Optional[bool] = None,
                 validation_dataset: Optional[Dataset] = None):
        if batch_size < 1 or num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers >= 0")
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")

        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = self.NUM_CLASSES
        self.pin_memory = (torch.cuda.is_available()
                           if pin_memory is None else pin_memory)
        self._validation_dataset = validation_dataset
        if validation_dataset is None:
            self.train_indices, self.validation_indices = self._stratified_split(
                self._split_labels, validation_fraction, split_seed)
            self.num_training_obs = len(self.train_indices)
        else:
            self.train_indices = list(range(len(self._train_dataset)))
            self.validation_indices = list(range(len(validation_dataset)))
            self.num_training_obs = len(self._train_dataset)

    @staticmethod
    def _stratified_split(labels: list[int], validation_fraction: float,
                          seed: int) -> tuple[list[int], list[int]]:
        labels_tensor = torch.as_tensor(labels, dtype=torch.long)
        generator = torch.Generator().manual_seed(seed)
        train_indices, validation_indices = [], []
        for class_id in torch.unique(labels_tensor, sorted=True).tolist():
            indices = torch.where(labels_tensor == class_id)[0]
            indices = indices[torch.randperm(len(indices), generator=generator)]
            validation_count = max(1, round(len(indices) * validation_fraction))
            if validation_count >= len(indices):
                raise ValueError(
                    f"Class {class_id} needs at least two samples for a split")
            validation_indices.extend(indices[:validation_count].tolist())
            train_indices.extend(indices[validation_count:].tolist())
        train_order = torch.randperm(len(train_indices), generator=generator).tolist()
        validation_order = torch.randperm(
            len(validation_indices), generator=generator).tolist()
        return ([train_indices[i] for i in train_order],
                [validation_indices[i] for i in validation_order])

    def _loader(self, dataset: Dataset, *, shuffle: bool, seed: int,
                drop_last: bool = False) -> DataLoader:
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle,
            drop_last=drop_last, num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            generator=torch.Generator().manual_seed(seed),
            persistent_workers=self.num_workers > 0)

    def get_train_loader(self, seed: int = 1, drop_last: bool = True):
        return self._loader(
            Subset(self._train_dataset, self.train_indices),
            shuffle=True, seed=seed, drop_last=drop_last)

    def get_validation_loader(self):
        if self._validation_dataset is not None:
            dataset = self._validation_dataset
        else:
            dataset = Subset(self._train_dataset, self.validation_indices)
        return self._loader(dataset, shuffle=False, seed=0)

    def get_test_loader(self):
        return self._loader(self._test_dataset, shuffle=False, seed=0)


class _SAMCenterCrop:
    """Center crop the shorter side to 87.5%, then resize bicubically."""

    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    def __call__(self, image):
        width, height = TF.get_image_size(image)
        crop_size = int((self.image_size / (self.image_size + 32))
                        * min(height, width))
        top = (height - crop_size + 1) // 2
        left = (width - crop_size + 1) // 2
        return TF.resized_crop(
            image, top, left, crop_size, crop_size,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BICUBIC)


def _imagenet_transform(image_size: int = 224):
    return transforms.Compose((
        _SAMCenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ))


class _HFImageDataset(Dataset):
    """Read images extracted from Hugging Face parquet shards in a local cache."""

    def __init__(self, repo_id: str, parquet_files: list[str],
                 cache_dir: Path, label_column: str = "label", transform=None):
        self.cache_dir = Path(cache_dir)
        self.transform = transform
        labels_file = self.cache_dir / "labels.json"
        if not labels_file.exists():
            self._extract(repo_id, parquet_files, label_column)
        meta = json.loads(labels_file.read_text())
        if "class_names" not in meta:
            meta["class_names"] = _hf_label_names(
                repo_id, parquet_files[0], label_column)
            labels_file.write_text(json.dumps(meta))
        self._labels = [int(label) for label in meta["labels"]]
        self._files = meta["files"]
        self.class_names = meta["class_names"]

    def _extract(self, repo_id: str, parquet_files: list[str],
                 label_column: str) -> None:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        labels, files = [], []
        for name in parquet_files:
            path = hf_hub_download(repo_id, name, repo_type="dataset")
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    data = row["image"]["bytes"]
                    ext = "png" if data[:4] == b"\x89PNG" else "jpg"
                    file_name = f"{len(files):05d}.{ext}"
                    (self.cache_dir / file_name).write_bytes(data)
                    files.append(file_name)
                    labels.append(int(row[label_column]))
        (self.cache_dir / "labels.json").write_text(json.dumps({
            "labels": labels, "files": files,
            "class_names": _hf_label_names(
                repo_id, parquet_files[0], label_column)}))

    def __len__(self):
        return len(self._files)

    def __getitem__(self, index: int):
        with Image.open(self.cache_dir / self._files[index]) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self._labels[index]


def _hf_label_names(repo_id: str, parquet_file: str,
                    label_column: str = "label") -> list[str]:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(repo_id, parquet_file, repo_type="dataset")
    metadata = json.loads(
        pq.ParquetFile(path).schema_arrow.metadata[b"huggingface"])
    return metadata["info"]["features"][label_column]["names"]


class Flowers102DataModule(_ImageNetDataModule):
    """Flowers102 official test split and a fixed stratified train/validation split."""

    NUM_CLASSES = 102
    REPO = "dpdl-benchmark/oxford_flowers102"
    TRAIN_FILES = ["data/train-00000-of-00001.parquet"]
    VALIDATION_FILES = ["data/validation-00000-of-00001.parquet"]
    TEST_FILES = [f"data/test-{i:05d}-of-00006.parquet" for i in range(6)]

    def __init__(self, data_root: str | Path, batch_size: int,
                 validation_fraction: float = 0.1, split_seed: int = 1,
                 num_workers: int = 4, pin_memory: Optional[bool] = None,
                 image_size: int = 224):
        cache = Path(data_root) / "flowers-102-hf"
        transform = _imagenet_transform(image_size)
        train = _HFImageDataset(self.REPO, self.TRAIN_FILES + self.VALIDATION_FILES,
                                cache / "trainval", transform=transform)
        test = _HFImageDataset(self.REPO, self.TEST_FILES, cache / "test",
                               transform=transform)
        self._train_dataset, self._test_dataset = train, test
        self._split_labels = train._labels
        super().__init__(data_root, batch_size, validation_fraction, split_seed,
                         num_workers, pin_memory)
        self.class_names = train.class_names


class OxfordIIITPetDataModule(_ImageNetDataModule):
    """Oxford-IIIT Pet official test split and a fixed stratified train/validation split."""

    NUM_CLASSES = 37
    REPO = "timm/oxford-iiit-pet"

    def __init__(self, data_root: str | Path, batch_size: int,
                 validation_fraction: float = 0.1, split_seed: int = 1,
                 num_workers: int = 4, pin_memory: Optional[bool] = None,
                 image_size: int = 224):
        cache = Path(data_root) / "oxford-iiit-pet-hf"
        transform = _imagenet_transform(image_size)
        train = _HFImageDataset(
            self.REPO, ["data/train-00000-of-00001.parquet"],
            cache / "trainval", transform=transform)
        test = _HFImageDataset(
            self.REPO, ["data/test-00000-of-00001.parquet"],
            cache / "test", transform=transform)
        self._train_dataset, self._test_dataset = train, test
        self._split_labels = train._labels
        super().__init__(data_root, batch_size, validation_fraction, split_seed,
                         num_workers, pin_memory)
        self.class_names = [" ".join(part.title() for part in name.split("_"))
                            for name in train.class_names]


class FGVCAircraftDataModule(_ImageNetDataModule):
    """FGVC Aircraft official train/validation/test splits (100 variants)."""

    NUM_CLASSES = 100
    REPO = "dpdl-benchmark/fgvc_aircraft"
    TRAIN_FILES = ["data/train-00000-of-00002.parquet",
                   "data/train-00001-of-00002.parquet"]
    VALIDATION_FILES = ["data/validation-00000-of-00002.parquet",
                        "data/validation-00001-of-00002.parquet"]
    TEST_FILES = ["data/test-00000-of-00002.parquet",
                  "data/test-00001-of-00002.parquet"]

    def __init__(self, data_root: str | Path, batch_size: int,
                 validation_fraction: float = 0.1, split_seed: int = 1,
                 num_workers: int = 4, pin_memory: Optional[bool] = None,
                 image_size: int = 224):
        cache = Path(data_root) / "fgvc-aircraft-hf"
        transform = _imagenet_transform(image_size)
        train = _HFImageDataset(
            self.REPO, self.TRAIN_FILES, cache / "train",
            label_column="variant", transform=transform)
        validation = _HFImageDataset(
            self.REPO, self.VALIDATION_FILES, cache / "validation",
            label_column="variant", transform=transform)
        test = _HFImageDataset(
            self.REPO, self.TEST_FILES, cache / "test",
            label_column="variant", transform=transform)
        self._train_dataset, self._test_dataset = train, test
        self._split_labels = train._labels
        super().__init__(data_root, batch_size, validation_fraction, split_seed,
                         num_workers, pin_memory, validation_dataset=validation)
        self.class_names = train.class_names


class StanfordCarsDataModule(_ImageNetDataModule):
    """Stanford Cars official train/test splits and fixed stratified validation."""

    NUM_CLASSES = 196
    REPO = "mteb/StanfordCars"
    TRAIN_FILES = ["data/train-00000-of-00002.parquet",
                   "data/train-00001-of-00002.parquet"]
    TEST_FILES = ["data/test-00000-of-00002.parquet",
                  "data/test-00001-of-00002.parquet"]

    def __init__(self, data_root: str | Path, batch_size: int,
                 validation_fraction: float = 0.1, split_seed: int = 1,
                 num_workers: int = 4, pin_memory: Optional[bool] = None,
                 image_size: int = 224):
        cache = Path(data_root) / "stanford-cars-hf"
        transform = _imagenet_transform(image_size)
        train = _HFImageDataset(self.REPO, self.TRAIN_FILES, cache / "train",
                                transform=transform)
        test = _HFImageDataset(self.REPO, self.TEST_FILES, cache / "test",
                               transform=transform)
        self._train_dataset, self._test_dataset = train, test
        self._split_labels = train._labels
        super().__init__(data_root, batch_size, validation_fraction, split_seed,
                         num_workers, pin_memory)
        self.class_names = train.class_names
