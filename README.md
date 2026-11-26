# Tái lập SAM bằng PyTorch

Repo so sánh SGD và Sharpness-Aware Minimization (SAM) trên hai nhóm thí nghiệm:

- Huấn luyện từ đầu WRN-16-8 trên CIFAR-10 và CIFAR-100.
- Fine-tune EfficientNet-B0 trên FGVC Aircraft, Flowers102, Oxford-IIIT Pet và Stanford Cars.

Mỗi YAML ghi dataset, mô hình, seed mặc định và hyperparameter. Optimizer, seed và augmentation được chọn khi chạy lệnh. Kết quả được ghi dưới `outputs/`.

## §3.3 Robustness to label noise (Table 4): SGD, SAM, Bootstrap + SAM

Mục này tái hiện ba dòng SGD, SAM, Bootstrap + SAM trong Table 4 (không tái lập Mixup): ResNet-32 train từ đầu trên CIFAR-10 có nhãn nhiễu, đánh giá trên tập test sạch. Repo gốc google-research/sam không có code cho thí nghiệm này, nên pipeline được viết mới: [`label_noise.py`](sam_pytorch/label_noise.py), [`resnet_cifar.py`](sam_pytorch/resnet_cifar.py), `CIFAR10DataModuleForNoise` trong [`datasets.py`](sam_pytorch/datasets.py), bước train chia micro-batch (tùy chọn) trong [`noise_train_step.py`](sam_pytorch/noise_train_step.py), và config [`cifar10_noisy_resnet32.yaml`](configs/cifar10_noisy_resnet32.yaml).

**Toàn bộ thông số, câu trích dẫn chứng minh từ bài báo và lý do chọn, cùng cách chạy: xem [readme_Noise.md](readme_Noise.md).**

```bash
python -m sam_pytorch.label_noise --config configs/cifar10_noisy_resnet32.yaml \
  --method sgd sam bootstrap_sam --noise-rate 0.2 0.4 0.6 0.8 --seed 1
python -m sam_pytorch.label_noise --config configs/cifar10_noisy_resnet32.yaml --summarize
```

## Cài đặt

Dùng Python tương thích với phiên bản PyTorch/torchvision đã chọn. Cài cặp PyTorch/torchvision phù hợp hệ điều hành và CUDA từ [trang cài đặt PyTorch](https://pytorch.org/get-started/locally/), sau đó cài các thư viện còn lại:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## CIFAR: WRN-16-8 huấn luyện từ đầu

Các augmentation hỗ trợ: `basic`, `cutout`, `autoaugment`. Cấu hình mặc định là `basic`.

```bash
python -m sam_pytorch.train --config configs/cifar10_wrn16x8.yaml --augmentation basic --optimizer sgd sam --seed 1
python -m sam_pytorch.train --config configs/cifar100_wrn16x8.yaml --augmentation cutout --optimizer sgd sam --seed 1
```

Có thể truyền nhiều optimizer, seed và augmentation trong một lệnh; các tổ hợp chạy tuần tự. Ví dụ:

```bash
python -m sam_pytorch.train --config configs/cifar10_wrn16x8.yaml \
  --augmentation basic cutout autoaugment --optimizer sgd sam --seed 1 2 3
```

## Fine-tune EfficientNet-B0

Bốn cấu hình fine-tuning hiện có:

| Dataset | Config |
|---|---|
| FGVC Aircraft | `configs/fgvc_aircraft_efficientnet_b0.yaml` |
| Flowers102 | `configs/flowers102_efficientnet_b0.yaml` |
| Oxford-IIIT Pet | `configs/oxford_iiit_pet_efficientnet_b0.yaml` |
| Stanford Cars | `configs/stanford_cars_efficientnet_b0.yaml` |

Chạy một hoặc nhiều optimizer/seeds:

```bash
python -m sam_pytorch.finetune --config configs/fgvc_aircraft_efficientnet_b0.yaml --optimizer sgd sam --seed 1
python -m sam_pytorch.finetune --config configs/flowers102_efficientnet_b0.yaml --optimizer sgd sam --seed 1
python -m sam_pytorch.finetune --config configs/oxford_iiit_pet_efficientnet_b0.yaml --optimizer sgd sam --seed 1
python -m sam_pytorch.finetune --config configs/stanford_cars_efficientnet_b0.yaml --optimizer sgd sam --seed 1
```

Lần chạy đầu cần kết nối Internet để tải dữ liệu từ các mirror Hugging Face và weights `efficientnet_b0.ra_in1k`. Dataset được cache bên dưới `data/`; weights dùng cache mặc định của Hugging Face/timm. Có thể tải trước dataset và weights bằng lệnh tương ứng:

```bash
python -m sam_pytorch.prepare_dataset --dataset fgvc_aircraft
python -m sam_pytorch.prepare_dataset --dataset flowers102
python -m sam_pytorch.prepare_dataset --dataset oxford_iiit_pet
python -m sam_pytorch.prepare_dataset --dataset stanford_cars
```

Fine-tuning dùng preprocessing xác định: center crop 87.5% cạnh ngắn, resize bicubic về kích thước trong YAML và ImageNet normalization. FGVC Aircraft dùng các split train/validation/test chính thức; Flowers102, Oxford-IIIT Pet và Stanford Cars giữ test chính thức riêng, rồi chia stratified phần train thành train/validation cố định theo seed.

## Giới hạn thời gian chạy

Dùng một trong hai tùy chọn để ghi đè giới hạn trong YAML:

- `--epochs N` đặt số epoch.
- `--steps N` đặt số lần cập nhật optimizer.

```bash
python -m sam_pytorch.train --config configs/cifar10_wrn16x8.yaml --optimizer sam --epochs 2
python -m sam_pytorch.finetune --config configs/flowers102_efficientnet_b0.yaml --optimizer sam --steps 1
```

`--steps` chỉ giới hạn phần huấn luyện. Khi kết thúc, chương trình vẫn tính báo cáo trên các tập train và test; bước đánh giá này có thể lâu trên CPU.

Mỗi run ghi `config.yaml`, `train.log`, `metrics.csv`, `last.pt` và `results.txt` vào thư mục riêng dưới `outputs/`. CIFAR có thêm augmentation trong đường dẫn kết quả; fine-tuning không có augmentation.

## Nguồn dữ liệu

- [FGVC Aircraft mirror](https://huggingface.co/datasets/dpdl-benchmark/fgvc_aircraft)
- [Flowers102 mirror](https://huggingface.co/datasets/dpdl-benchmark/oxford_flowers102)
- [Oxford-IIIT Pet mirror](https://huggingface.co/datasets/timm/oxford-iiit-pet)
- [Stanford Cars mirror](https://huggingface.co/datasets/mteb/StanfordCars)
- [EfficientNet-B0 weights `efficientnet_b0.ra_in1k`](https://huggingface.co/timm/efficientnet_b0.ra_in1k)
- [Sharpness-Aware Minimization](https://arxiv.org/abs/2010.01412)
