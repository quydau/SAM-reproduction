"""Dataset splits and PyTorch data loaders for the experiments."""

import csv
import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import (ConcatDataset, DataLoader, Dataset, Subset,
                              default_collate)
import torch.nn.functional as F
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


class _CIFARTensorBatches:
    """Iterate ``(images, labels)`` batches from CIFAR uint8 tensors on-device.

    Replaces the PIL/DataLoader pipeline for the none/basic/cutout conditions:
    horizontal flip, reflect-padded random crop, normalization and batch Cutout
    run on ``device``. Shuffle order matches ``DataLoader(shuffle=True,
    generator=manual_seed(seed))``; per-image flip/crop draws come from the same
    generator, so a given seed is reproducible (but not bit-identical to PIL).
    """

    def __init__(self, images: torch.Tensor, labels: torch.Tensor,
                 indices: torch.Tensor, batch_size: int, *, shuffle: bool,
                 seed: int, drop_last: bool, augmentation: CIFAR10Augmentation,
                 mean: tuple, std: tuple, crop_padding: int, flip: float,
                 augment: bool):
        self.images, self.labels, self.indices = images, labels, indices
        self.batch_size, self.shuffle, self.seed = batch_size, shuffle, seed
        self.drop_last, self.augmentation = drop_last, augmentation
        self.crop_padding, self.flip, self.augment = crop_padding, flip, augment
        device = images.device
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)

    @property
    def dataset(self) -> torch.Tensor:
        """Sized stand-in for ``DataLoader.dataset`` (sample count reporting)."""
        return self.indices

    def __len__(self) -> int:
        count = len(self.indices)
        if self.drop_last:
            return count // self.batch_size
        return -(-count // self.batch_size)

    def _augment(self, images: torch.Tensor,
                 generator: torch.Generator) -> torch.Tensor:
        count, device = images.shape[0], images.device
        flipped = torch.rand(count, generator=generator) < self.flip
        images = torch.where(flipped.to(device).view(-1, 1, 1, 1),
                             images.flip(3), images)
        pad = self.crop_padding
        if pad:
            size = images.shape[-1]
            padded = F.pad(images, (pad,) * 4, mode="reflect")
            offsets = torch.randint(0, 2 * pad + 1, (2, count),
                                    generator=generator).to(device)
            span = torch.arange(size, device=device)
            rows = (offsets[0].view(-1, 1) + span).view(count, 1, size, 1)
            cols = (offsets[1].view(-1, 1) + span).view(count, 1, 1, size)
            batch = torch.arange(count, device=device).view(count, 1, 1, 1)
            channel = torch.arange(images.shape[1], device=device).view(1, -1, 1, 1)
            images = padded[batch, channel, rows, cols]
        return images

    def __iter__(self):
        count = len(self.indices)
        generator = torch.Generator().manual_seed(self.seed)
        if self.shuffle:
            # DataLoader draws its worker base seed from the generator first.
            torch.empty((), dtype=torch.int64).random_(generator=generator)
            order = torch.randperm(count, generator=generator)
        else:
            order = torch.arange(count)
        order = order.to(self.indices.device)
        for batch in range(len(self)):
            picked = self.indices[order[batch * self.batch_size:
                                        (batch + 1) * self.batch_size]]
            images = self.images[picked].float().div_(255)
            if self.augment:
                images = self._augment(images, generator)
            images = (images - self.mean) / self.std
            if self.augment:
                images = self.augmentation.apply_cutout(images)
            yield images, self.labels[picked]


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
                 horizontal_flip_probability: float = 0.5):
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
        # AutoAugment is PIL-only, so it keeps the DataLoader path; the other
        # conditions run entirely on tensors cached on the device.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._gpu_pipeline = augmentation != "autoaugment"
        self._crop_padding = crop_padding
        self._flip = horizontal_flip_probability
        self._tensors: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        # Tạo phép chia có thể tái lập: cùng seed sẽ cho cùng train/validation indices.
        split_generator = torch.Generator().manual_seed(validation_seed)
        indices = torch.randperm(len(self._train_augmented),
                                 generator=split_generator).tolist()
        self.train_indices = indices[:self.TRAIN_SIZE]
        self.validation_indices = indices[self.TRAIN_SIZE:]

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
            # Loader được tạo mới mỗi epoch/lần eval nên không giữ worker sống,
            # tránh rò rỉ pipe/file descriptor ("Too many open files").
            persistent_workers=False,
        )

    def _tensor_batches(self, source, indices: list[int], *, shuffle: bool,
                        drop_last: bool, seed: int, augment: bool,
                        batch_size: Optional[int] = None) -> _CIFARTensorBatches:
        key = id(source)
        if key not in self._tensors:
            images = torch.from_numpy(source.data).permute(0, 3, 1, 2).contiguous()
            self._tensors[key] = (images.to(self.device),
                                  torch.tensor(source.targets).to(self.device))
        images, labels = self._tensors[key]
        return _CIFARTensorBatches(
            images, labels, torch.as_tensor(indices, device=self.device),
            batch_size or self.batch_size, shuffle=shuffle, seed=seed,
            drop_last=drop_last, augmentation=self.augmentation,
            mean=self.MEAN, std=self.STD, crop_padding=self._crop_padding,
            flip=self._flip, augment=augment and self.augmentation.name != "none")

    def get_train_loader(self, use_augmentations: bool = True, seed: int = 1):
        """Return train data, dropping an incomplete batch for training."""
        if self._gpu_pipeline:
            return self._tensor_batches(
                self._train_augmented, self.train_indices, shuffle=True,
                drop_last=use_augmentations, seed=seed,
                augment=use_augmentations)
        # Subset chọn đúng các chỉ số đã phân vào train, không gồm validation.
        source = self._train_augmented if use_augmentations else self._train_plain
        dataset = Subset(source, self.train_indices)
        return self._loader(dataset, shuffle=True,
                            # Khi augmentation bật, bỏ batch cuối nếu batch đó không đủ kích thước.
                            drop_last=use_augmentations, seed=seed,
                            cutout=use_augmentations)

    def get_validation_loader(self):
        """Return the fixed validation split without augmentation."""
        if self._gpu_pipeline:
            return self._tensor_batches(
                self._train_plain, self.validation_indices, shuffle=False,
                drop_last=False, seed=0, augment=False)
        # Validation dùng ảnh nguyên bản, đọc tuần tự và giữ lại toàn bộ mẫu.
        dataset = Subset(self._train_plain, self.validation_indices)
        return self._loader(dataset, shuffle=False, drop_last=False, seed=0)

    def get_test_loader(self):
        """Return the official test split without augmentation."""
        if self._gpu_pipeline:
            return self._tensor_batches(
                self._test, list(range(len(self._test))), shuffle=False,
                drop_last=False, seed=0, augment=False)
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
                 num_workers: int = 4, pin_memory: Optional[bool] = None,
                 image_mean=(0.485, 0.456, 0.406),
                 image_std=(0.229, 0.224, 0.225)):
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
        # num_workers/pin_memory are kept for config compatibility only: the
        # cached uint8 tensors need no worker processes.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_mean, self.image_std = image_mean, image_std
        self._tensor_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
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

    # Ảnh đã tiền xử lý (uint8) được nạp một lần rồi giữ trong bộ nhớ, nên không
    # cần DataLoader/worker: mỗi epoch chỉ là chọn chỉ số và chuẩn hóa trên GPU.
    def _batches(self, dataset: Dataset, indices: Optional[list[int]], *,
                 shuffle: bool, seed: int, drop_last: bool,
                 on_device: bool) -> "_TensorBatches":
        key = id(dataset)
        if key not in self._tensor_cache:
            parts = (dataset.datasets if isinstance(dataset, ConcatDataset)
                     else [dataset])
            images = torch.cat([part.uint8_tensors() for part in parts])
            if on_device:
                images = images.to(self.device)
            labels = torch.tensor([label for part in parts
                                   for label in part._labels])
            self._tensor_cache[key] = (images, labels)
        images, labels = self._tensor_cache[key]
        if indices is None:
            indices = list(range(len(labels)))
        return _TensorBatches(
            images, labels, torch.as_tensor(indices, device=images.device),
            self.batch_size, shuffle=shuffle, seed=seed, drop_last=drop_last,
            device=images.device, mean=self.image_mean, std=self.image_std)

    def get_train_loader(self, seed: int = 1,
                         drop_last: bool = True) -> "_TensorBatches":
        """Trả các batch của tập train, xáo trộn theo ``seed``.

        Mặc định bỏ batch cuối thiếu mẫu, giống ``floor(N / batch_size)`` của
        bản JAX; truyền ``drop_last=False`` khi đánh giá toàn bộ tập train.

        Ảnh đã được tiền xử lý theo ImageNet; các chỉ số lấy từ lần chia cố định
        khi khởi tạo module.
        """
        return self._batches(self._train_dataset, self.train_indices,
                             shuffle=True, seed=seed, drop_last=drop_last,
                             on_device=False)

    def get_validation_loader(self) -> "_TensorBatches":
        """Trả các batch validation, đọc tuần tự để đánh giá ổn định."""
        return self._batches(self._train_dataset, self.validation_indices,
                             shuffle=False, seed=0, drop_last=False,
                             on_device=False)

    def get_test_loader(self) -> "_TensorBatches":
        """Trả các batch của partition test chính thức, không xáo trộn.

        Tập test lớn hơn nhiều và chỉ dùng một lần ở cuối, nên giữ trên CPU.
        """
        return self._batches(self._test_dataset, None, shuffle=False, seed=0,
                             drop_last=False, on_device=False)


class _TensorBatches:
    """Iterate normalized ``(images, labels)`` batches from a uint8 tensor.

    Replaces a ``DataLoader`` over the deterministic (augmentation-free)
    pipeline: the shuffle order matches ``DataLoader(shuffle=True,
    generator=manual_seed(seed))``, and uint8 -> /255 -> mean/std
    normalization is done on the storage device. Fine-tuning image batches
    remain on CPU until the training step transfers one microbatch to CUDA.
    """

    def __init__(self, images: torch.Tensor, labels: torch.Tensor,
                 indices: torch.Tensor, batch_size: int, *, shuffle: bool,
                 seed: int, drop_last: bool, device: torch.device,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.images, self.labels, self.indices = images, labels, indices
        self.batch_size, self.shuffle, self.seed = batch_size, shuffle, seed
        self.drop_last, self.device = drop_last, device
        self.mean, self.std = mean, std

    @property
    def dataset(self) -> torch.Tensor:
        """Sized stand-in for ``DataLoader.dataset`` (sample count reporting)."""
        return self.indices

    def __len__(self) -> int:
        count = len(self.indices)
        if self.drop_last:
            return count // self.batch_size
        return -(-count // self.batch_size)

    def __iter__(self):
        count = len(self.indices)
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed)
            # DataLoader draws its worker base seed from the generator first.
            torch.empty((), dtype=torch.int64).random_(generator=generator)
            order = torch.randperm(count, generator=generator)
            order = order.to(self.indices.device)
        else:
            order = torch.arange(count, device=self.indices.device)
        mean = torch.tensor(self.mean, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=self.device).view(1, 3, 1, 1)
        for batch in range(len(self)):
            picked = self.indices[order[batch * self.batch_size:
                                        (batch + 1) * self.batch_size]]
            images = self.images[picked].to(self.device, non_blocking=True)
            images = (images.float().div_(255) - mean) / std
            labels = self.labels[picked.cpu()].to(self.device)
            yield images, labels


def _imagenet_transform(image_size: int = 224) -> transforms.Compose:
    """Tạo pipeline biến đổi ảnh dùng cho các dataset fine-tuning.

    Ảnh được crop/resize về image_size×image_size và giữ dạng tensor uint8; việc chia 255 và
    chuẩn hóa theo mean/độ lệch chuẩn ImageNet do ``_TensorBatches`` làm trên GPU.
    """
    return transforms.Compose((
        _SAMCenterCrop(image_size),
        transforms.PILToTensor(),
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
                 cache_dir: Path, transform=None, image_size: int = 224):
        self.cache_dir = Path(cache_dir)
        self.transform = transform
        self.image_size = image_size
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
            for batch in pq.ParquetFile(path).iter_batches(
                    batch_size=256, columns=["image", "label"]):
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

    def uint8_tensors(self) -> torch.Tensor:
        """All transformed images as one uint8 tensor, cached on disk.

        Only valid when ``transform`` returns a uint8 tensor. The decode +
        crop + resize pass runs once and is reused by later runs.
        """
        path = self.cache_dir / f"tensors_{self.image_size}.pt"
        if path.exists():
            return torch.load(path)
        images = torch.stack([self[i][0] for i in range(len(self))])
        temporary = path.with_suffix(".tmp")
        torch.save(images, temporary)
        temporary.replace(path)
        return images

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
                 num_workers: int = 4, pin_memory: Optional[bool] = None,
                 image_size: int = 224,
                 image_mean=(0.485, 0.456, 0.406),
                 image_std=(0.229, 0.224, 0.225)):
        self.data_root = Path(data_root)
        transform = _imagenet_transform(image_size)
        cache = self.data_root / "flowers-102-hf"
        train, val, test = (
            _HFImageDataset(self.REPO, files, cache / name, transform, image_size)
            for name, files in (
                ("train", ["data/train-00000-of-00001.parquet"]),
                ("val", ["data/validation-00000-of-00001.parquet"]),
                ("test", [f"data/test-{i:05d}-of-00006.parquet"
                          for i in range(6)])))
        self._train_dataset = ConcatDataset([train, val])
        self._test_dataset = test
        self._split_labels = train._labels + val._labels
        super().__init__(data_root, batch_size, validation_fraction,
                         split_seed, num_workers, pin_memory,
                         image_mean, image_std)
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
                 num_workers: int = 4, pin_memory: Optional[bool] = None,
                 image_size: int = 224,
                 image_mean=(0.485, 0.456, 0.406),
                 image_std=(0.229, 0.224, 0.225)):
        self.data_root = Path(data_root)
        transform = _imagenet_transform(image_size)
        cache = self.data_root / "oxford-iiit-pet-hf"
        self._train_dataset = _HFImageDataset(
            self.REPO, ["data/train-00000-of-00001.parquet"],
            cache / "trainval", transform, image_size)
        self._test_dataset = _HFImageDataset(
            self.REPO, ["data/test-00000-of-00001.parquet"],
            cache / "test", transform, image_size)
        self._split_labels = list(self._train_dataset._labels)
        super().__init__(data_root, batch_size, validation_fraction,
                         split_seed, num_workers, pin_memory,
                         image_mean, image_std)
        self.class_names = [" ".join(part.title() for part in name.split("_"))
                            for name in self._train_dataset.class_names]


class FGVCAircraftDataModule(_ImageNetDataModule):
    """FGVC Aircraft (nhãn variant, 100 lớp) với split train/validation cố định.

    Dữ liệu lấy từ mirror ``Multimodal-Fatima/FGVC_Aircraft_{train,test}``:
    split ``train`` chính thức (3334 ảnh) được chia theo từng lớp (mặc định
    90/10) thành train/validation, test chính thức (3333 ảnh) được giữ riêng.
    Mirror này không có split ``val`` chính thức nên không gộp thêm.
    """

    NUM_CLASSES = 100
    REPO = "Multimodal-Fatima/FGVC_Aircraft"

    def __init__(self, data_root: str | Path, batch_size: int,
                 validation_fraction: float = 0.1, split_seed: int = 1,
                 num_workers: int = 4, pin_memory: Optional[bool] = None,
                 image_size: int = 224,
                 image_mean=(0.485, 0.456, 0.406),
                 image_std=(0.229, 0.224, 0.225)):
        self.data_root = Path(data_root)
        transform = _imagenet_transform(image_size)
        cache = self.data_root / "fgvc-aircraft-hf"
        self._train_dataset = _HFImageDataset(
            f"{self.REPO}_train",
            ["data/train-00000-of-00002-8e8c2fac664e9852.parquet",
             "data/train-00001-of-00002-367db3639819cbee.parquet"],
            cache / "train", transform, image_size)
        self._test_dataset = _HFImageDataset(
            f"{self.REPO}_test",
            ["data/test-00000-of-00002-f701e149b12ef922.parquet",
             "data/test-00001-of-00002-995f2f92aa55f647.parquet"],
            cache / "test", transform, image_size)
        self._split_labels = list(self._train_dataset._labels)
        super().__init__(data_root, batch_size, validation_fraction,
                         split_seed, num_workers, pin_memory,
                         image_mean, image_std)
        self.class_names = self._train_dataset.class_names


class StanfordCarsDataModule(_ImageNetDataModule):
    """Stanford Cars loaders with a fixed stratified train/validation split.

    The Hugging Face mirror ``mteb/StanfordCars`` preserves the official
    8,144-image training and 8,041-image test partitions. The official train
    partition is divided by class into train/validation; test stays separate.
    Labels are zero-based class ids and class names come from parquet metadata.
    """

    NUM_CLASSES = 196
    REPO = "mteb/StanfordCars"

    def __init__(self, data_root: str | Path, batch_size: int,
                 validation_fraction: float = 0.1, split_seed: int = 1,
                 num_workers: int = 4, pin_memory: Optional[bool] = None,
                 image_size: int = 224,
                 image_mean=(0.485, 0.456, 0.406),
                 image_std=(0.229, 0.224, 0.225)):
        self.data_root = Path(data_root)
        transform = _imagenet_transform(image_size)
        cache = self.data_root / "stanford-cars-hf"
        self._train_dataset = _HFImageDataset(
            self.REPO,
            ["data/train-00000-of-00002.parquet",
             "data/train-00001-of-00002.parquet"],
            cache / "train", transform, image_size)
        self._test_dataset = _HFImageDataset(
            self.REPO,
            ["data/test-00000-of-00002.parquet",
             "data/test-00001-of-00002.parquet"],
            cache / "test", transform, image_size)
        self._split_labels = list(self._train_dataset._labels)
        super().__init__(data_root, batch_size, validation_fraction,
                         split_seed, num_workers, pin_memory,
                         image_mean, image_std)
        self.class_names = self._train_dataset.class_names


class _BirdsnapDataset(Dataset):
    """Birdsnap images with an optional per-epoch PIL transform."""

    def __init__(self, samples: list[tuple[Path, int]], transform):
        self.samples, self.transform = samples, transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), label


class BirdsnapDataModule:
    """Birdsnap official train/test split with a stratified validation split.

    Expects the extracted dataset under ``data_root/birdsnap``. Annotation
    files may be in that directory or its ``annotations`` subdirectory; images
    may be under ``images/`` or directly under the dataset directory.
    AutoAugment is applied to PIL images on every training epoch.
    """

    NUM_CLASSES = 500

    def __init__(self, data_root: str | Path, batch_size: int,
                 validation_fraction: float = 0.1, split_seed: int = 1,
                 num_workers: int = 4, pin_memory: Optional[bool] = None,
                 augmentation: str = "autoaugment",
                 image_size: int = 224,
                 image_mean=(0.485, 0.456, 0.406),
                 image_std=(0.229, 0.224, 0.225)):
        if batch_size < 1 or num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers >= 0")
        if augmentation != "autoaugment":
            raise ValueError("Birdsnap fine-tuning requires augmentation='autoaugment'")
        self.data_root = Path(data_root) / "birdsnap"
        self.batch_size, self.num_workers = batch_size, num_workers
        self.pin_memory = (torch.cuda.is_available()
                           if pin_memory is None else pin_memory)
        self.num_classes = self.NUM_CLASSES
        self.image_size = image_size

        annotation_roots = (self.data_root / "annotations", self.data_root)
        def annotation_file(name: str) -> Path:
            for root in annotation_roots:
                path = root / name
                if path.is_file():
                    return path
            raise FileNotFoundError(
                f"Birdsnap {name} not found under {self.data_root}")

        species = {}
        with annotation_file("species.txt").open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                species[int(row["id"])] = row["common"]
        if set(species) != set(range(1, self.NUM_CLASSES + 1)):
            raise ValueError("Birdsnap species.txt must list species IDs 1 through 500")
        self.class_names = [species[i] for i in range(1, self.NUM_CLASSES + 1)]

        test_paths = set()
        with annotation_file("test_images.txt").open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                path = row.get("path") or next(iter(row.values()), "")
                if path:
                    test_paths.add(path)

        def image_path(relative: str) -> Optional[Path]:
            candidates = (self.data_root / "images" / relative,
                          self.data_root / relative)
            return next((path for path in candidates if path.is_file()), None)

        train_samples, test_samples = [], []
        with annotation_file("images.txt").open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                relative = row["path"]
                path = image_path(relative)
                if path is None:
                    continue  # Original Birdsnap can contain unavailable Flickr images.
                sample = (path, int(row["species_id"]) - 1)
                (test_samples if relative in test_paths else train_samples).append(sample)
        if not train_samples or not test_samples:
            raise ValueError(
                "No Birdsnap images found; extract the official images and annotations "
                f"under {self.data_root}")

        labels = [label for _, label in train_samples]
        train_indices, validation_indices = _ImageNetDataModule._stratified_split(
            labels, validation_fraction, split_seed)
        self.num_training_obs = len(train_indices)
        self._train_augmented = _BirdsnapDataset(
            train_samples,
            transforms.Compose((
                transforms.AutoAugment(
                    policy=transforms.AutoAugmentPolicy.IMAGENET),
                _SAMCenterCrop(image_size), transforms.ToTensor(),
                transforms.Normalize(image_mean, image_std))))
        self._train_plain = _BirdsnapDataset(
            train_samples,
            transforms.Compose((_SAMCenterCrop(image_size), transforms.ToTensor(),
                                transforms.Normalize(image_mean, image_std))))
        self._test = _BirdsnapDataset(
            test_samples,
            transforms.Compose((_SAMCenterCrop(image_size), transforms.ToTensor(),
                                transforms.Normalize(image_mean, image_std))))
        self._train_indices, self._validation_indices = train_indices, validation_indices

    def _loader(self, dataset: Dataset, *, shuffle: bool, seed: int,
                drop_last: bool = False) -> DataLoader:
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle,
            drop_last=drop_last, num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            generator=torch.Generator().manual_seed(seed),
            persistent_workers=False)

    def get_train_loader(self, seed: int = 1,
                         drop_last: bool = True) -> DataLoader:
        dataset = Subset(self._train_augmented, self._train_indices)
        return self._loader(dataset, shuffle=True, seed=seed, drop_last=drop_last)

    def get_plain_train_loader(self, seed: int = 1) -> DataLoader:
        dataset = Subset(self._train_plain, self._train_indices)
        return self._loader(dataset, shuffle=False, seed=seed)

    def get_validation_loader(self) -> DataLoader:
        dataset = Subset(self._train_plain, self._validation_indices)
        return self._loader(dataset, shuffle=False, seed=0)

    def get_test_loader(self) -> DataLoader:
        return self._loader(self._test, shuffle=False, seed=0)
