# Tái lập thí nghiệm SAM bằng PyTorch

Repo phục vụ đồ án môn Học máy, tái lập và so sánh Sharpness-Aware Minimization (SAM) với SGD theo paper [*Sharpness-Aware Minimization for Efficiently Improving Generalization*](https://arxiv.org/abs/2010.01412) của Foret et al. (ICLR 2021). Mã nguồn tham khảo chính thức: [google-research/sam](https://github.com/google-research/sam). Repo JAX/Flax gốc được giữ riêng tại `../sam`; repo này dùng PyTorch và tự chọn CUDA, Apple MPS (GPU Apple Silicon), hoặc CPU.

## Câu hỏi và phạm vi đồ án

SAM thực hiện cập nhật SGD trên loss tại các trọng số đã bị perturb trong bán kính `rho`, nhằm tìm nghiệm có loss ổn định trong lân cận. Đồ án so sánh SGD và SAM với cùng dữ liệu, seed, khởi tạo, lịch học và các hyperparameter còn lại.

Các nhóm thực nghiệm theo kế hoạch:

- CIFAR-10, WRN-28-10 huấn luyện từ đầu với `basic`, `cutout`, `autoaugment`.
- Fine-tune EfficientNet-B0 trên Flowers102 và Oxford-IIIT Pet.
- Ba seed ghép cặp giữa SGD và SAM cho mỗi điều kiện; chọn checkpoint trên validation rồi đánh giá test một lần.

Paper báo cáo năm lần chạy độc lập cho fine-tuning. Ba seed hiện dùng là giới hạn phạm vi đồ án; vì vậy độ bất định thống kê sẽ khác paper. Ngoài ra, paper fine-tune EfficientNet-B7 và EfficientNet-L2 với checkpoint công khai, trong khi cấu hình T4 ở đây dùng EfficientNet-B0 với weights ImageNet từ torchvision. Đây là tái lập có điều chỉnh về mô hình/phần cứng, không phải kết quả số học tương đương với thí nghiệm gốc.

## Hyperparameter và nguồn gốc

| Thí nghiệm | Giá trị cấu hình | Nguồn |
|---|---|---|
| CIFAR-10 / WRN-28-10 | Batch hiệu dụng 256, 200 epoch, LR 0.1, WD 0.0005, `rho=0.05`, cosine, SGD momentum 0.9, Nesterov | Paper Appendix C.1, Table 6 nêu batch/LR/WD/rho; repo chính thức dùng cosine, momentum 0.9, Nesterov và clipping mặc định 5 |
| CIFAR augmentation | Basic: reflect-pad 4, random crop + flip; AutoAugment CIFAR rồi basic; Cutout: mask 16×16 chung batch sau normalization | Paper Appendix C; thứ tự/policy/chi tiết mask theo repo chính thức |
| Fine-tune Flowers/Pet theo C.2 | WD 1e-5; LR 0.016 với batch 1024, LR 0.008 với batch 512; cosine; momentum 0.9; đóng băng thống kê BatchNorm; không augmentation. | Paper Appendix C.2. Repo SAM không dispatch Flowers/Pet; preprocessing ở đây dùng center crop, resize và ImageNet normalization, không có phép biến đổi ngẫu nhiên. |
| Fine-tune đề xuất cho T4 | Batch hiệu dụng 256 (microbatch 16 × tích lũy 16), LR 0.004, 20,000 bước, WD 1e-5, `rho=0.05`, EfficientNet-B0 | Batch/LR/steps là suy luận để khớp xấp xỉ số mẫu đã thấy: LR scale tuyến tính từ C.2; 20k × 256 tương đương 5k × 1024. B0 là thay đổi để phù hợp bộ nhớ, không phải model trong paper |

Các giá trị cho CIFAR được nêu trực tiếp trong paper. `rho=0.05` là lựa chọn có căn cứ: paper dùng giá trị này cho CIFAR-10 và cũng báo cáo kết quả fine-tuning với `rho=0.05`; kết quả fine-tuning tốt nhất có thể dùng rho đã tinh chỉnh riêng. Với Flowers/Pet, LR 0.004 và batch hiệu dụng 256 là ngoại suy theo tỷ lệ từ 0.016 @ 1024 và 0.008 @ 512. Paper không xác nhận trực tiếp giá trị 0.004. Weights ImageNet1K_V1 và kích thước đầu vào 224 là thay đổi để dùng EfficientNet-B0; cách crop/resize theo code SAM, không dùng transform dành riêng cho inference của torchvision.

`nesterov: true` được đặt rõ trong optimizer SGD của repo SAM. Với CIFAR, `gradient_clip_norm: 5.0` và `label_smoothing: 0.0` giữ mặc định code path của repo gốc. Với fine-tuning EfficientNet, cấu hình dùng `gradient_clip_norm: 1.0` và `label_smoothing: 0.1` theo lệnh fine-tune EfficientNet trong README của [repo SAM gốc](../sam/README.md); đây là cấu hình lệnh ví dụ cho EfficientNet-L2 trên ImageNet, được áp dụng cho fine-tune B0 Flowers/Pet của đồ án, không phải giá trị được báo cáo trực tiếp trong paper C.2.

Trên T4 16 GB, cấu hình CIFAR dùng microbatch 64 × tích lũy 4; fine-tuning dùng microbatch 16 × tích lũy 16. Đây là ước lượng ban đầu, cần giảm microbatch nếu gặp out-of-memory. Tích lũy gradient giữ batch hiệu dụng cho bước tối ưu, nhưng BatchNorm trong WRN vẫn thống kê trên microbatch vật lý; điều này có thể làm kết quả lệch khỏi batch 256 thực sự. Fine-tuning cố định BatchNorm ở chế độ eval như C.2, nên vấn đề này không áp dụng cùng cách cho EfficientNet. Đối với Flowers/Pet, repo gốc không có nhánh dataset; theo C.2, loader dùng cùng preprocessing xác định cho train/validation/test: center crop 87.5% cạnh ngắn, resize bicubic về 224×224 và ImageNet normalization. Không áp dụng random crop, flip hay RandAugment.

Để phù hợp thiết kế đồ án, kế hoạch dùng validation split cố định 10% ở CIFAR và Flowers/Pet, cùng ba seed. Flowers102 gộp split `train` và `val` có sẵn của torchvision rồi chia stratified 90/10; Oxford-IIIT Pet chia stratified 90/10 từ split `trainval`. Đây là quy ước triển khai của repo; paper dùng validation 10% để chọn rho trong một số thí nghiệm nhưng không mô tả đúng split này cho toàn bộ pipeline. Không dùng test set để chọn hyperparameter hoặc checkpoint. Repo SAM khai báo tên Flowers/Pet trong enum nhưng không dispatch các dataset này trong `main`, nên split và preprocessing hiện tại là lựa chọn mới của đồ án, không phải port tương đương từ code SAM.

## Chạy thí nghiệm

Các YAML trong [`configs/`](configs/) là cấu hình mẫu được ghi chú nguồn trực tiếp và suy luận. Mỗi lần gọi chạy đúng **một optimizer, một seed và một cấu hình augmentation/dataset**. `--optimizer` là bắt buộc; nếu bỏ `--seed`, dùng seed trong YAML. Ví dụ:

```bash
python -m sam_pytorch.train --config configs/cifar10_wrn28x10.yaml --augmentation basic --optimizer sgd --seed 1
python -m sam_pytorch.train --config configs/cifar10_wrn28x10.yaml --augmentation basic --optimizer sam --seed 1
python -m sam_pytorch.train --config configs/cifar10_wrn28x10.yaml --augmentation cutout --optimizer sam --seed 2
python -m sam_pytorch.train --config configs/cifar10_wrn28x10.yaml --augmentation autoaugment --optimizer sgd --seed 3
python -m sam_pytorch.finetune --config configs/flowers102_efficientnet_b0.yaml --optimizer sgd --seed 1
python -m sam_pytorch.finetune --config configs/oxford_iiit_pet_efficientnet_b0.yaml --optimizer sam --seed 2
```

Để chạy thử ngắn, truyền **một** trong `--epochs N` hoặc `--steps N`. Tham số được chọn ghi đè giới hạn tương ứng trong YAML; hai cờ không dùng đồng thời. Nếu không truyền cờ giới hạn, chương trình dùng nguyên `training.epochs` hoặc `training.max_steps` trong YAML. `--steps` là số lần cập nhật optimizer. Ví dụ `--steps 5` cho fine-tuning hoặc `--epochs 2` cho CIFAR. Tiến độ epoch được in ra terminal và append ngay vào `train.log`; `metrics.csv`, `metrics.txt` (báo cáo tiến độ trong lúc chạy) và checkpoint cũng được cập nhật sau mỗi epoch. Khi hoàn tất, `metrics.txt`/`results.txt` chứa báo cáo đầy đủ; test được đánh giá một lần. Báo cáo gồm `test_error`, classification report theo lớp và confusion matrix. Trên Mac, cài PyTorch/torchvision bản macOS; MPS được tự chọn khi có GPU Apple Silicon.

Để chạy toàn bộ ba seed, ba augmentation CIFAR và hai optimizer, có thể phát sinh từng lệnh riêng theo vòng lặp shell (bỏ `--epochs`/`--steps` để dùng cấu hình đầy đủ):

```bash
for seed in 1 2 3; do
  for augmentation in basic cutout autoaugment; do
    for optimizer in sgd sam; do
      python -m sam_pytorch.train --config configs/cifar10_wrn28x10.yaml \
        --augmentation "$augmentation" --optimizer "$optimizer" --seed "$seed"
    done
  done
done

for seed in 1 2 3; do
  for optimizer in sgd sam; do
    python -m sam_pytorch.finetune --config configs/flowers102_efficientnet_b0.yaml \
      --optimizer "$optimizer" --seed "$seed"
    python -m sam_pytorch.finetune --config configs/oxford_iiit_pet_efficientnet_b0.yaml \
      --optimizer "$optimizer" --seed "$seed"
  done
done
```

Mỗi lệnh Python trên là một run độc lập; các vòng lặp chỉ tuần tự gọi những lệnh này. Để smoke-run nhanh, thêm `--steps 5` (fine-tuning) hoặc `--epochs 2` (CIFAR) vào lệnh tương ứng.

Output có dạng:

```text
outputs/{dataset}/{model}/{augmentation}/{optimizer}/seed_{seed}/
```

Ví dụ: `outputs/cifar10/wrn28x10/basic/sam/seed_1/`.

## Cài đặt

Khuyến nghị Python 3.10 hoặc 3.11. Cài cặp `torch`/`torchvision` phù hợp hệ điều hành và CUDA bằng [bộ chọn cài đặt PyTorch chính thức](https://pytorch.org/get-started/locally/), sau đó cài phần còn lại:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Tài liệu tham khảo

- Foret et al., [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412), ICLR 2021.
- [Mã nguồn chính thức Google Research](https://github.com/google-research/sam) (JAX/Flax, TensorFlow Datasets).
- [Weights EfficientNet-B0 của torchvision](https://docs.pytorch.org/vision/stable/models/efficientnet.html) dùng trong cấu hình fine-tune đã điều chỉnh.
