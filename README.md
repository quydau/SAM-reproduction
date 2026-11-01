# Tái lập thí nghiệm SAM bằng PyTorch

Repo phục vụ đồ án môn Học máy, tái lập và so sánh Sharpness-Aware Minimization (SAM) với SGD theo paper [*Sharpness-Aware Minimization for Efficiently Improving Generalization*](https://arxiv.org/abs/2010.01412) của Foret et al. (ICLR 2021). Mã nguồn tham khảo chính thức: [google-research/sam](https://github.com/google-research/sam). Repo JAX/Flax gốc được giữ riêng tại `../sam`; repo này dùng PyTorch và tự chọn CUDA, Apple MPS (GPU Apple Silicon), hoặc CPU.

## Câu hỏi và phạm vi đồ án

SAM thực hiện cập nhật SGD trên loss tại các trọng số đã bị perturb trong bán kính `rho`, nhằm tìm nghiệm có loss ổn định trong lân cận. Đồ án so sánh SGD và SAM với cùng dữ liệu, seed, khởi tạo, lịch học và các hyperparameter còn lại.

Các nhóm thực nghiệm theo kế hoạch:

- CIFAR-10, WRN-16-8 huấn luyện từ đầu với `basic`, `cutout`, `autoaugment`.
- Fine-tune EfficientNet-B0 trên Flowers102 và Oxford-IIIT Pet.
- Ba seed ghép cặp giữa SGD và SAM cho mỗi điều kiện; chọn checkpoint trên validation rồi đánh giá test một lần.

Paper báo cáo năm lần chạy độc lập cho fine-tuning. Ba seed hiện dùng là giới hạn phạm vi đồ án; vì vậy độ bất định thống kê sẽ khác paper. Ngoài ra, paper fine-tune EfficientNet-B7 và EfficientNet-L2 với checkpoint công khai, trong khi cấu hình T4 ở đây dùng EfficientNet-B0 với weights ImageNet từ torchvision. Đây là tái lập có điều chỉnh về mô hình/phần cứng, không phải kết quả số học tương đương với thí nghiệm gốc.

## Hyperparameter và nguồn gốc

| Thí nghiệm | Giá trị cấu hình | Nguồn |
|---|---|---|
| CIFAR-10 / WRN-16-8 (paper dùng WRN-28-10; ở đây thay bằng WRN-16-8 nhỏ hơn) | Batch hiệu dụng 256, 200 epoch, LR 0.1, WD 0.0005, `rho=0.05`, cosine, SGD momentum 0.9, Nesterov | Paper Appendix C.1, Table 6 nêu batch/LR/WD/rho; repo chính thức dùng cosine, momentum 0.9, Nesterov và clipping mặc định 5 |
| CIFAR augmentation | Basic: reflect-pad 4, random crop + flip; AutoAugment CIFAR rồi basic; Cutout: mask 16×16 chung batch sau normalization | Paper Appendix C; thứ tự/policy/chi tiết mask theo repo chính thức |
| Fine-tune Flowers/Pet theo C.2 | WD 1e-5; LR 0.016 với batch 1024, LR 0.008 với batch 512; cosine; momentum 0.9; đóng băng thống kê BatchNorm; không augmentation. | Paper Appendix C.2. Repo SAM không dispatch Flowers/Pet; preprocessing ở đây dùng center crop, resize và ImageNet normalization, không có phép biến đổi ngẫu nhiên. |
| Fine-tune đề xuất cho T4 | Batch hiệu dụng 256 (microbatch 16 × tích lũy 16), LR 0.004, 20,000 bước, WD 1e-5, `rho=0.05`, EfficientNet-B0 | Batch/LR/steps là suy luận để khớp xấp xỉ số mẫu đã thấy: LR scale tuyến tính từ C.2; 20k × 256 tương đương 5k × 1024. B0 là thay đổi để phù hợp bộ nhớ, không phải model trong paper |

Các giá trị cho CIFAR được nêu trực tiếp trong paper. `rho=0.05` là lựa chọn có căn cứ: paper dùng giá trị này cho CIFAR-10 và cũng báo cáo kết quả fine-tuning với `rho=0.05`; kết quả fine-tuning tốt nhất có thể dùng rho đã tinh chỉnh riêng. Với Flowers/Pet, LR 0.004 và batch hiệu dụng 256 là ngoại suy theo tỷ lệ từ 0.016 @ 1024 và 0.008 @ 512. Paper không xác nhận trực tiếp giá trị 0.004. Weights ImageNet1K_V1 và kích thước đầu vào 224 là thay đổi để dùng EfficientNet-B0; cách crop/resize theo code SAM, không dùng transform dành riêng cho inference của torchvision.

`nesterov: true` được đặt rõ trong optimizer SGD của repo SAM. Với CIFAR, `gradient_clip_norm: 5.0` và `label_smoothing: 0.0` giữ mặc định code path của repo gốc. Với fine-tuning EfficientNet, cấu hình dùng `gradient_clip_norm: 1.0` và `label_smoothing: 0.1` theo lệnh fine-tune EfficientNet trong README của [repo SAM gốc](../sam/README.md); đây là cấu hình lệnh ví dụ cho EfficientNet-L2 trên ImageNet, được áp dụng cho fine-tune B0 Flowers/Pet của đồ án, không phải giá trị được báo cáo trực tiếp trong paper C.2.

Trên T4 16 GB, cấu hình CIFAR dùng microbatch 64 × tích lũy 4; fine-tuning dùng microbatch 16 × tích lũy 16. Đây là ước lượng ban đầu, cần giảm microbatch nếu gặp out-of-memory. Tích lũy gradient giữ batch hiệu dụng cho bước tối ưu, nhưng BatchNorm trong WRN vẫn thống kê trên microbatch vật lý; điều này có thể làm kết quả lệch khỏi batch 256 thực sự. Fine-tuning cố định BatchNorm ở chế độ eval như C.2, nên vấn đề này không áp dụng cùng cách cho EfficientNet. Đối với Flowers/Pet, repo gốc không có nhánh dataset; theo C.2, loader dùng cùng preprocessing xác định cho train/validation/test: center crop 87.5% cạnh ngắn, resize bicubic về 224×224 và ImageNet normalization. Không áp dụng random crop, flip hay RandAugment.

Để phù hợp thiết kế đồ án, kế hoạch dùng validation split cố định 10% ở CIFAR và Flowers/Pet, cùng ba seed. Flowers102 gộp split `train` và `val` có sẵn của torchvision rồi chia stratified 90/10; Oxford-IIIT Pet chia stratified 90/10 từ split `trainval`. Đây là quy ước triển khai của repo; paper dùng validation 10% để chọn rho trong một số thí nghiệm nhưng không mô tả đúng split này cho toàn bộ pipeline. Không dùng test set để chọn hyperparameter hoặc checkpoint. Repo SAM khai báo tên Flowers/Pet trong enum nhưng không dispatch các dataset này trong `main`, nên split và preprocessing hiện tại là lựa chọn mới của đồ án, không phải port tương đương từ code SAM.

## Chạy thí nghiệm

Các YAML trong [`configs/`](configs/) là cấu hình mẫu được ghi chú nguồn trực tiếp và suy luận. `--optimizer` là bắt buộc; `--optimizer`, `--seed` và `--augmentation` (chỉ CIFAR) nhận **một hoặc nhiều giá trị**, và một lệnh sẽ chạy **tuần tự** mọi tổ hợp (mỗi tổ hợp là một run độc lập, có thư mục output riêng). Nếu bỏ `--seed`/`--augmentation`, dùng giá trị trong YAML. Ví dụ:

```bash
python -m sam_pytorch.train --config configs/cifar10_wrn16x8.yaml --augmentation basic --optimizer sgd --seed 1
python -m sam_pytorch.train --config configs/cifar10_wrn16x8.yaml --augmentation basic --optimizer sam --seed 1
python -m sam_pytorch.train --config configs/cifar10_wrn16x8.yaml --augmentation cutout --optimizer sam --seed 2
python -m sam_pytorch.train --config configs/cifar10_wrn16x8.yaml --augmentation autoaugment --optimizer sgd --seed 3
python -m sam_pytorch.finetune --config configs/flowers102_efficientnet_b0.yaml --optimizer sgd --seed 1
python -m sam_pytorch.finetune --config configs/oxford_iiit_pet_efficientnet_b0.yaml --optimizer sam --seed 2

# Nhiều giá trị trong một lệnh (tuần tự): 3 seed x 3 augmentation x 2 optimizer = 18 run
python -m sam_pytorch.train --config configs/cifar10_wrn16x8.yaml \
  --augmentation basic cutout autoaugment --optimizer sgd sam --seed 1 2 3
python -m sam_pytorch.finetune --config configs/flowers102_efficientnet_b0.yaml --optimizer sgd sam --seed 1 2 3
```

Để chạy thử ngắn, truyền **một** trong `--epochs N` hoặc `--steps N`. Tham số được chọn ghi đè giới hạn tương ứng trong YAML; hai cờ không dùng đồng thời. Nếu không truyền cờ giới hạn, chương trình dùng nguyên `training.epochs` hoặc `training.max_steps` trong YAML. `--steps` là số lần cập nhật optimizer. Ví dụ `--steps 5` cho fine-tuning hoặc `--epochs 2` cho CIFAR. Tiến độ epoch được in ra terminal và append ngay vào `train.log`; `metrics.csv`, `metrics.txt` (báo cáo tiến độ trong lúc chạy) và checkpoint cũng được cập nhật sau mỗi epoch. Khi hoàn tất, `metrics.txt`/`results.txt` chứa báo cáo đầy đủ; test được đánh giá một lần. Báo cáo gồm `test_error`, classification report theo lớp và confusion matrix. Trên Mac, cài PyTorch/torchvision bản macOS; MPS được tự chọn khi có GPU Apple Silicon.

Các lệnh nhiều giá trị ở trên đã thay cho vòng lặp shell; thêm `--epochs 2` hoặc `--steps 5` để smoke-run. Các run chạy nối tiếp trong cùng một tiến trình, không song song (một GPU T4 không đủ chỗ cho nhiều run cùng lúc).

Output có dạng:

```text
outputs/{dataset}/{model}/{augmentation}/{optimizer}/seed_{seed}/
```

Ví dụ: `outputs/cifar10/wrn16x8/basic/sam/seed_1/`.

## §3.3 Robustness to label noise (Table 4): SGD, SAM, Bootstrap + SAM

Mục này tái hiện ba dòng SGD, SAM và Bootstrap + SAM trong Table 4. Mô hình là ResNet-32 **train từ đầu** (khởi tạo ngẫu nhiên) trên CIFAR-10. Một phần nhãn train bị lật, còn validation và test giữ sạch. Repo gốc google-research/sam không có code cho thí nghiệm này, nên toàn bộ pipeline ở đây được viết mới: [`sam_pytorch/label_noise.py`](sam_pytorch/label_noise.py), [`sam_pytorch/resnet_cifar.py`](sam_pytorch/resnet_cifar.py), phần nhãn nhiễu trong [`datasets.py`](sam_pytorch/datasets.py), và config [`configs/cifar10_noisy_resnet32.yaml`](configs/cifar10_noisy_resnet32.yaml).

| Mục | Giá trị | Nguồn |
|---|---|---|
| Model | ResNet-32 (He et al. 2016, n=5, kênh 16/32/64, 464K tham số), from scratch | SAM §3.3; Jiang et al. 2020 §5.1: *"we train networks from scratch using ResNet-32 for CIFAR"* |
| Epochs / batch | 200 / 256 | SAM §3.3 / SAM C.1 |
| ρ | 0.1 cho nhiễu 20/40/60%, 0.05 cho 80% | SAM §3.3, Table 8 |
| LR / WD | 0.1 / 5e-4 | **Tự chọn**: bài không ghi cho §3.3, lấy theo Table 6 (CIFAR-10, 200 epochs) |
| Lịch LR, optimizer, L2, clip | cosine; SGD momentum 0.9 Nesterov; L2 `0.5·wd·‖w‖²` trên mọi tham số; clip 5 | Code gốc SAM |
| Augmentation | basic (flip + pad 4 + crop) | Jiang B.2: *"standard data augmentation"* |
| Split | 45k train (nhiễu) / 5k validation (sạch) / 10k test (sạch) | SAM C.1: validation sạch = 10% tập train |
| Nhiễu | Chọn đúng ⌊r·45000⌋ mẫu, đổi nhãn sang một lớp **khác**, chọn ngẫu nhiên đều | **Tự chọn** cách hiểu chữ "randomly flipped" |
| Bootstrap | Model SAM ở epoch cuối dự đoán argmax trên 45k ảnh train (không augmentation); sau đó train SAM lại từ đầu trên nhãn dự đoán, cùng ρ | SAM §3.3: *"retrained from scratch on the labels predicted by the initially trained model"* |

Các điểm bài không ghi và repo phải tự chọn: LR/WD, kiểu lật nhãn, shortcut option A của ResNet-32, dùng nhãn cứng khi bootstrap, train trên 45k (giữ 5k làm validation). Cùng một seed thì cùng bộ nhãn nhiễu (`noise_seed = noise_seed_offset + seed`), nên SGD, SAM và Bootstrap được so trên đúng một bộ nhãn. Kết quả báo cáo là model ở epoch cuối, đánh giá test một lần. Mục `[train_metrics]` trong `results.txt` được tính trên nhãn train đang dùng: nhãn nhiễu, hoặc nhãn dự đoán với bootstrap.

```bash
# 4 mức nhiễu x 3 phương pháp; bootstrap_sam tái dùng run sam cùng mức nhiễu/seed (tự train nếu chưa có)
python -m sam_pytorch.label_noise --config configs/cifar10_noisy_resnet32.yaml \
  --method sgd sam bootstrap_sam --noise-rate 0.2 0.4 0.6 0.8 --seed 1
# Chạy thử nhanh
python -m sam_pytorch.label_noise --config configs/cifar10_noisy_resnet32.yaml \
  --method sgd sam bootstrap_sam --noise-rate 0.8 --steps 5
# Gom kết quả thành bảng giống Table 4 (ghi ra outputs/cifar10_noisy/summary.md)
python -m sam_pytorch.label_noise --config configs/cifar10_noisy_resnet32.yaml --summarize
```

Output nằm ở `outputs/cifar10_noisy/resnet32/noise_{20,40,60,80}/{sgd,sam,bootstrap_sam}/seed_{s}/`. Thư mục bootstrap có thêm `bootstrap_labels.pt` và `bootstrap_stats.txt`; file stats ghi tỷ lệ nhãn dự đoán trùng nhãn sạch và trùng nhãn nhiễu. Nếu train lại run `sam` thì phải xóa `bootstrap_labels.pt` để nhãn được dự đoán lại.

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
