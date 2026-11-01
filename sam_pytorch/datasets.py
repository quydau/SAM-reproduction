"""Dataset splits and PyTorch data loaders for the experiments."""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import (ConcatDataset, DataLoader, Dataset, Subset,
                              default_collate)
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.datasets import CIFAR10, CIFAR100
from PIL import Image

from sam_pytorch.augmentation import CIFAR10Augmentation


class _BatchCollator:
    """Collate image-label pairs and optionally apply batch-level Cutout."""

    def __init__(self, augmentation: CIFAR10Augmentation, enabled: bool):
        self.augmentation = augmentation
        self.enabled = enabled

    def __call__(self, samples):
        # Ghép từng cặp (ảnh, nhãn) thành hai tensor theo batch.
        images, labels = default_collate(samples)
        if self.enabled:
            # Cutout cần tensor cả batch nên được áp dụng sau khi ghép ảnh.
            images = self.augmentation.apply_cutout(images)
        return images, labels


def make_flipped_labels(clean_labels: list[int], rate: float, num_classes: int,
                        seed: int) -> list[int]:
    """Flip exactly ``floor(rate * N)`` labels, each to a *different* class.

    The flipped examples are a seeded uniform sample without replacement; each
    new label is drawn uniformly from the other ``num_classes - 1`` classes,
    so the fraction of wrong labels is exactly ``rate`` (up to flooring).
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError("label noise rate must be in [0, 1]")
    labels = torch.as_tensor(clean_labels, dtype=torch.long).clone()
    generator = torch.Generator().manual_seed(seed)
    num_noisy = int(rate * len(labels))
    flipped = torch.randperm(len(labels), generator=generator)[:num_noisy]
    offsets = torch.randint(1, num_classes, (num_noisy,), generator=generator)
    labels[flipped] = (labels[flipped] + offsets) % num_classes
    return labels.tolist()


class _RelabeledSubset(Dataset):
    """Subset of a labelled dataset whose labels are replaced by ``labels``."""

    def __init__(self, source: Dataset, indices: list[int], labels: list[int]):
        if len(indices) != len(labels):
            raise ValueError("indices and labels must have the same length")
        self.source = source
        self.indices = list(indices)
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i):
        image, _ = self.source[self.indices[i]]
        return image, self.labels[i]


class CIFAR10DataModule:
    """Owns reproducible CIFAR-10 train/validation/test loaders.

    The official 50,000-image training set is split 45,000/5,000 using a
    seeded random permutation. The 10,000-image official test set remains
    separate. Loaders yield normalized ``(images, labels)`` tensors.
    """

    # CIFAR-10 có 10 lớp; tập train chính thức được chia 45k/5k.
    NUM_CLASSES = 10
    TRAIN_SIZE = 45_000
    VALIDATION_SIZE = 5_000
    DATASET_CLS = CIFAR10
    MEAN = CIFAR10Augmentation.MEAN
    STD = CIFAR10Augmentation.STD

    def __init__(self, data_root: str | Path, batch_size: int,
                 augmentation: str = "basic", validation_seed: int = 1,
                 num_workers: int = 4, cutout_length: int = 16,
                 crop_padding: int = 4, pin_memory: Optional[bool] = None,
                 horizontal_flip_probability: float = 0.5,
                 label_noise_rate: float = 0.0, noise_seed: int = 0,
                 train_labels_override: Optional[list[int]] = None):
        """``label_noise_rate`` flips that fraction of the 45k *training*
        labels (validation and test stay clean); ``train_labels_override``
        replaces the training labels entirely (used by bootstrapping)."""
        if batch_size < 1 or num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers >= 0")

        # Lưu cấu hình dùng chung cho các DataLoader.
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = self.NUM_CLASSES
        self.num_training_obs = self.TRAIN_SIZE
        # Mặc định bật pin_memory khi có GPU CUDA, giúp chuyển tensor lên GPU nhanh hơn.
        self.pin_memory = (torch.cuda.is_available()
                           if pin_memory is None else pin_memory)
        # Đối tượng này cung cấp biến đổi ảnh và Cutout theo cấu hình đã chọn.
        self.augmentation = CIFAR10Augmentation(
            augmentation, crop_padding=crop_padding,
            cutout_length=cutout_length,
            horizontal_flip_probability=horizontal_flip_probability,
            mean=self.MEAN, std=self.STD)
        plain = CIFAR10Augmentation("none", mean=self.MEAN, std=self.STD)

        # Hai dataset train cùng trỏ tới 50.000 ảnh, nhưng dùng transform khác nhau.
        # Bản augmented dành cho train; bản plain dành cho validation/đánh giá công bằng.
        self._train_augmented = self.DATASET_CLS(
            root=str(self.data_root), train=True, download=True,
            transform=self.augmentation.image_transform)
        self._train_plain = self.DATASET_CLS(
            root=str(self.data_root), train=True, download=False,
            transform=plain.image_transform)
        # Test chính thức (10.000 ảnh) luôn được giữ riêng và không augmentation.
        self._test = self.DATASET_CLS(
            root=str(self.data_root), train=False, download=True,
            transform=plain.image_transform)
        self.class_names = list(self._test.classes)

        # Tạo phép chia có thể tái lập: cùng seed sẽ cho cùng train/validation indices.
        split_generator = torch.Generator().manual_seed(validation_seed)
        indices = torch.randperm(len(self._train_augmented),
                                 generator=split_generator).tolist()
        self.train_indices = indices[:self.TRAIN_SIZE]
        self.validation_indices = indices[self.TRAIN_SIZE:]

        # Nhãn train: sạch, bị lật một phần (label noise), hoặc thay hẳn (bootstrap).
        targets = self._train_plain.targets
        self.clean_train_labels = [int(targets[i]) for i in self.train_indices]
        if train_labels_override is not None:
            if len(train_labels_override) != self.TRAIN_SIZE:
                raise ValueError("train_labels_override must have 45,000 labels")
            self.train_labels = [int(y) for y in train_labels_override]
        elif label_noise_rate > 0:
            self.train_labels = make_flipped_labels(
                self.clean_train_labels, label_noise_rate, self.NUM_CLASSES,
                noise_seed)
        else:
            self.train_labels = list(self.clean_train_labels)
        self.num_wrong_train_labels = sum(
            a != b for a, b in zip(self.train_labels, self.clean_train_labels))

    def _loader(self, dataset: Dataset, *, shuffle: bool, drop_last: bool,
                seed: int, cutout: bool = False,
                batch_size: Optional[int] = None) -> DataLoader:
        # Seed điều khiển thứ tự lấy mẫu khi shuffle=True.
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            generator=generator,
            collate_fn=_BatchCollator(self.augmentation, enabled=cutout),
            # Giữ worker sống giữa các epoch; chỉ bật khi thực sự có worker.
            persistent_workers=self.num_workers > 0,
        )

    def get_train_loader(self, use_augmentations: bool = True,
                         seed: int = 1) -> DataLoader:
        """Return train data, dropping an incomplete batch for training."""
        # Subset chọn đúng các chỉ số đã phân vào train, không gồm validation.
        source = self._train_augmented if use_augmentations else self._train_plain
        dataset = _RelabeledSubset(source, self.train_indices, self.train_labels)
        return self._loader(dataset, shuffle=True,
                            # Khi augmentation bật, bỏ batch cuối nếu batch đó không đủ kích thước.
                            drop_last=use_augmentations, seed=seed,
                            cutout=use_augmentations)

    def get_train_prediction_loader(self) -> DataLoader:
        """Train images in fixed order, no augmentation, for bootstrapping.

        Batch ``i`` of this loader covers ``train_indices[i*B:(i+1)*B]``.
        """
        dataset = _RelabeledSubset(self._train_plain, self.train_indices,
                                   self.train_labels)
        return self._loader(dataset, shuffle=False, drop_last=False, seed=0)

    def get_validation_loader(self) -> DataLoader:
        """Return the fixed validation split without augmentation."""
        # Validation dùng ảnh nguyên bản, đọc tuần tự và giữ lại toàn bộ mẫu.
        dataset = Subset(self._train_plain, self.validation_indices)
        return self._loader(dataset, shuffle=False, drop_last=False, seed=0)

    def get_test_loader(self) -> DataLoader:
        """Return the official test split without augmentation."""
        # Giới hạn batch test tối đa 32 để giảm bộ nhớ khi đánh giá.
        return self._loader(
            self._test, shuffle=False, drop_last=False, seed=0,
            batch_size=min(32, self.batch_size))


class CIFAR100DataModule(CIFAR10DataModule):
    """CIFAR-100 with the same 45k/5k split and loaders as CIFAR-10."""

    NUM_CLASSES = 100
    DATASET_CLS = CIFAR100
    MEAN = CIFAR10Augmentation.CIFAR100_MEAN
    STD = CIFAR10Augmentation.CIFAR100_STD


class _ImageNetDataModule:
    """Shared deterministic splitting and loader behavior for fine-tuning."""

    NUM_CLASSES: int

    # Khởi tạo cấu hình chung và tạo chỉ số train/validation. Lớp con phải
    # chuẩn bị dataset cùng danh sách nhãn trước khi gọi phần khởi tạo này.
    def __init__(self, data_root: str | Path, batch_size: int,
                 validation_fraction: float = 0.1, split_seed: int = 1,
                 num_workers: int = 4, pin_memory: Optional[bool] = None):
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
        self.train_indices, self.validation_indices = self._stratified_split(
            self._split_labels, validation_fraction, split_seed)
        self.num_training_obs = len(self.train_indices)

    # Chia theo từng lớp: labels[i] là nhãn mẫu thứ i; trả về chỉ số train và
    # validation sao cho mỗi lớp có mặt trong cả hai tập, cố định theo seed.
    @staticmethod
    def _stratified_split(labels: list[int], validation_fraction: float,
                          seed: int) -> tuple[list[int], list[int]]:
        """Split each class independently so rare classes appear in both sets."""
        labels_tensor = torch.as_tensor(labels, dtype=torch.long)
        generator = torch.Generator().manual_seed(seed)
        train_indices: list[int] = []
        validation_indices: list[int] = []
        for class_id in torch.unique(labels_tensor, sorted=True).tolist():
            indices = torch.where(labels_tensor == class_id)[0]
            indices = indices[torch.randperm(len(indices), generator=generator)]
            validation_count = max(1, round(len(indices) * validation_fraction))
            if validation_count >= len(indices):
                raise ValueError(
                    f"Class {class_id} needs at least two samples for a split")
            validation_indices.extend(indices[:validation_count].tolist())
            train_indices.extend(indices[validation_count:].tolist())
        # Keep the resulting order deterministic and avoid class-grouped batches.
        train_order = torch.randperm(len(train_indices), generator=generator).tolist()
        validation_order = torch.randperm(
            len(validation_indices), generator=generator).tolist()
        return ([train_indices[i] for i in train_order],
                [validation_indices[i] for i in validation_order])

    # Tạo DataLoader với cấu hình chung; shuffle/seed điều khiển thứ tự đọc,
    # drop_last quyết định có bỏ batch cuối nếu batch đó thiếu mẫu hay không.
    def _loader(self, dataset: Dataset, *, shuffle: bool, seed: int,
                drop_last: bool = False) -> DataLoader:
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle,
            drop_last=drop_last, num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            generator=torch.Generator().manual_seed(seed),
            persistent_workers=self.num_workers > 0)

    def get_train_loader(self, seed: int = 1,
                         drop_last: bool = True) -> DataLoader:
        """Trả DataLoader cho tập train, có xáo trộn theo ``seed``.

        Mặc định bỏ batch cuối thiếu mẫu, giống ``floor(N / batch_size)`` của
        bản JAX; truyền ``drop_last=False`` khi đánh giá toàn bộ tập train.

        Ảnh đã được tiền xử lý theo ImageNet; các chỉ số lấy từ lần chia cố định
        khi khởi tạo module.
        """
        return self._loader(
            Subset(self._train_dataset, self.train_indices),
            shuffle=True, seed=seed, drop_last=drop_last)

    def get_validation_loader(self) -> DataLoader:
        """Trả DataLoader validation, đọc tuần tự để đánh giá ổn định."""
        return self._loader(Subset(self._train_dataset,
                                   self.validation_indices),
                            shuffle=False, seed=0)

    def get_test_loader(self) -> DataLoader:
        """Trả DataLoader của partition test chính thức, không xáo trộn."""
        return self._loader(self._test_dataset, shuffle=False, seed=0)


def _imagenet_transform() -> transforms.Compose:
    """Tạo pipeline biến đổi ảnh dùng cho các dataset fine-tuning.

    Ảnh được crop/resize về 224×224, đổi thành tensor rồi chuẩn hóa theo
    mean và độ lệch chuẩn ImageNet.
    """
    return transforms.Compose((
        _SAMCenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ))


class _SAMCenterCrop:
    """Port SAM's 224/(224+32) center crop followed by bicubic resize."""

    """Thiết lập kích thước vuông đầu ra của phép crop/resize."""
    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    """Crop vùng trung tâm của ảnh rồi resize về ``image_size`` mỗi cạnh."""
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


class _HFImageDataset(Dataset):
    """Ảnh từ một split parquet trên Hugging Face, cache thành file trên đĩa.

    Lần đầu, các shard parquet được tải về rồi giải từng ảnh ra
    ``cache_dir`` (giữ nguyên byte JPEG/PNG của mirror) cùng ``labels.json``;
    các lần sau chỉ đọc cache. Nhãn là chỉ số lớp 0-based có sẵn trong parquet.
    """

    def __init__(self, repo_id: str, parquet_files: list[str],
                 cache_dir: Path, transform=None):
        self.cache_dir = Path(cache_dir)
        self.transform = transform
        labels_file = self.cache_dir / "labels.json"
        if not labels_file.exists():
            self._extract(repo_id, parquet_files)
        meta = json.loads(labels_file.read_text())
        if "class_names" not in meta:
            # Cache cũ chưa lưu tên lớp: lấy một lần rồi ghi lại để lần sau offline.
            meta["class_names"] = _hf_label_names(repo_id, parquet_files[0])
            labels_file.write_text(json.dumps(meta))
        self._labels: list[int] = meta["labels"]
        self._files: list[str] = meta["files"]
        self.class_names: list[str] = meta["class_names"]

    def _extract(self, repo_id: str, parquet_files: list[str]) -> None:
        # Import tại chỗ để chỉ cần các thư viện này khi dùng nguồn Hugging Face.
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
                    labels.append(int(row["label"]))
        # Ghi labels.json cuối cùng để cache dở dang không bị coi là hợp lệ.
        (self.cache_dir / "labels.json").write_text(
            json.dumps({"labels": labels, "files": files,
                        "class_names": _hf_label_names(repo_id,
                                                       parquet_files[0])}))

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, index: int):
        with Image.open(self.cache_dir / self._files[index]) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self._labels[index]


def _hf_label_names(repo_id: str, parquet_file: str) -> list[str]:
    """Đọc tên lớp (theo thứ tự chỉ số nhãn) từ metadata của dataset Hugging Face."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    path = hf_hub_download(repo_id, parquet_file, repo_type="dataset")
    info = json.loads(pq.ParquetFile(path).schema_arrow.metadata[b"huggingface"])
    return info["info"]["features"]["label"]["names"]


class Flowers102DataModule(_ImageNetDataModule):
    """Flowers102 loaders using official test data and a fixed stratified split.

    Dữ liệu lấy từ mirror Hugging Face ``dpdl-benchmark/oxford_flowers102``
    giữ đúng split chính thức 1020/1020/6149. Train và validation chính thức
    được gộp làm nguồn, rồi chia lại theo từng lớp (mặc định 90/10); test
    chính thức được giữ riêng. Nhãn là chỉ số lớp 0–101 có sẵn trong parquet.
    """

    NUM_CLASSES = 102
    REPO = "dpdl-benchmark/oxford_flowers102"

    def __init__(self, data_root: str | Path, batch_size: int,
                 validation_fraction: float = 0.1, split_seed: int = 1,
                 num_workers: int = 4, pin_memory: Optional[bool] = None):
        self.data_root = Path(data_root)
        transform = _imagenet_transform()
        cache = self.data_root / "flowers-102-hf"
        train, val, test = (
            _HFImageDataset(self.REPO, files, cache / name, transform)
            for name, files in (
                ("train", ["data/train-00000-of-00001.parquet"]),
                ("val", ["data/validation-00000-of-00001.parquet"]),
                ("test", [f"data/test-{i:05d}-of-00006.parquet"
                          for i in range(6)])))
        self._train_dataset = ConcatDataset([train, val])
        self._test_dataset = test
        self._split_labels = train._labels + val._labels
        super().__init__(data_root, batch_size, validation_fraction,
                         split_seed, num_workers, pin_memory)
        self.class_names = train.class_names


class OxfordIIITPetDataModule(_ImageNetDataModule):
    """Oxford-IIIT Pet loaders with a seeded stratified train/validation split.

    Dữ liệu lấy từ ``timm/oxford-iiit-pet``: split ``train`` là trainval chính
    thức (3680), ``test`` là test chính thức (3669). Phần trainval được chia
    thành train/validation theo từng lớp; test được giữ riêng.
    """

    NUM_CLASSES = 37
    REPO = "timm/oxford-iiit-pet"

    def __init__(self, data_root: str | Path, batch_size: int,
                 validation_fraction: float = 0.1, split_seed: int = 1,
                 num_workers: int = 4, pin_memory: Optional[bool] = None):
        self.data_root = Path(data_root)
        transform = _imagenet_transform()
        cache = self.data_root / "oxford-iiit-pet-hf"
        self._train_dataset = _HFImageDataset(
            self.REPO, ["data/train-00000-of-00001.parquet"],
            cache / "trainval", transform)
        self._test_dataset = _HFImageDataset(
            self.REPO, ["data/test-00000-of-00001.parquet"],
            cache / "test", transform)
        self._split_labels = list(self._train_dataset._labels)
        super().__init__(data_root, batch_size, validation_fraction,
                         split_seed, num_workers, pin_memory)
        self.class_names = [" ".join(part.title() for part in name.split("_"))
                            for name in self._train_dataset.class_names]
