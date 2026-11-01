"""Dataset splits and PyTorch data loaders for the experiments."""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import (ConcatDataset, DataLoader, Dataset, Subset,
                              default_collate)
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.datasets import CIFAR10, Flowers102, OxfordIIITPet

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
            horizontal_flip_probability=horizontal_flip_probability)

        # Hai dataset train cùng trỏ tới 50.000 ảnh, nhưng dùng transform khác nhau.
        # Bản augmented dành cho train; bản plain dành cho validation/đánh giá công bằng.
        self._train_augmented = CIFAR10(
            root=str(self.data_root), train=True, download=True,
            transform=self.augmentation.image_transform)
        self._train_plain = CIFAR10(
            root=str(self.data_root), train=True, download=False,
            transform=CIFAR10Augmentation("none").image_transform)
        # Test chính thức (10.000 ảnh) luôn được giữ riêng và không augmentation.
        self._test = CIFAR10(
            root=str(self.data_root), train=False, download=True,
            transform=CIFAR10Augmentation("none").image_transform)
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


def _dataset_labels(dataset: Dataset, *, one_based: bool = False) -> list[int]:
    """Đọc nhãn từng mẫu từ dataset torchvision thành danh sách số nguyên.

    Nếu ``one_based=True``, trừ 1 để đổi nhãn bắt đầu từ 1 sang chỉ số từ 0.
    """
    labels = getattr(dataset, "_labels", None)
    if labels is None:
        labels = getattr(dataset, "targets", None)
    if labels is None:
        raise RuntimeError(
            f"Cannot find labels on torchvision dataset {type(dataset).__name__}")
    result = [int(label) for label in labels]
    return [label - 1 for label in result] if one_based else result


def _flowers_label_to_zero_based(label: int) -> int:
    """Đổi nhãn Flowers102 từ khoảng 1–102 sang chỉ số lớp 0–101."""
    return label - 1


class Flowers102DataModule(_ImageNetDataModule):
    """Flowers102 loaders using official test data and a fixed stratified split.

    The official train and validation partitions are combined as the training
    pool, then split class-wise into train and validation (default 90/10).
    The official test partition remains untouched for final evaluation.
    Labels are exposed as zero-based class indices.
    """

    NUM_CLASSES = 102

    # Gộp train và val chính thức làm nguồn để chia lại; giữ test riêng.
    # download=True cho phép tự tải dữ liệu nếu máy chưa có.
    def __init__(self, data_root: str | Path, batch_size: int,
                 validation_fraction: float = 0.1, split_seed: int = 1,
                 num_workers: int = 4, pin_memory: Optional[bool] = None,
                 download: bool = True):
        self.data_root = Path(data_root)
        transform = _imagenet_transform()
        self._train_dataset = ConcatDataset([
            Flowers102(root=str(self.data_root), split="train",
                       transform=transform,
                       target_transform=_flowers_label_to_zero_based,
                       download=download),
            Flowers102(root=str(self.data_root), split="val",
                       transform=transform,
                       target_transform=_flowers_label_to_zero_based,
                       download=download),
        ])
        self._test_dataset = Flowers102(
            root=str(self.data_root), split="test", transform=transform,
            target_transform=_flowers_label_to_zero_based, download=download)
        self._split_labels = (
            _dataset_labels(self._train_dataset.datasets[0], one_based=True)
            + _dataset_labels(self._train_dataset.datasets[1], one_based=True))
        super().__init__(data_root, batch_size, validation_fraction,
                         split_seed, num_workers, pin_memory)
        self.class_names = [str(index) for index in range(self.NUM_CLASSES)]


class OxfordIIITPetDataModule(_ImageNetDataModule):
    """Oxford-IIIT Pet loaders with a seeded stratified train/validation split."""

    NUM_CLASSES = 37

    # Chia partition trainval thành train/validation; giữ partition test riêng.
    # download=True cho phép tự tải dữ liệu nếu máy chưa có.
    def __init__(self, data_root: str | Path, batch_size: int,
                 validation_fraction: float = 0.1, split_seed: int = 1,
                 num_workers: int = 4, pin_memory: Optional[bool] = None,
                 download: bool = True):
        self.data_root = Path(data_root)
        transform = _imagenet_transform()
        self._train_dataset = OxfordIIITPet(
            root=str(self.data_root), split="trainval",
            transform=transform, download=download)
        self._test_dataset = OxfordIIITPet(
            root=str(self.data_root), split="test",
            transform=transform, download=download)
        self._split_labels = _dataset_labels(self._train_dataset)
        super().__init__(data_root, batch_size, validation_fraction,
                         split_seed, num_workers, pin_memory)
        self.class_names = list(self._train_dataset.classes)
