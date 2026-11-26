# Tái hiện SAM §3.3: Robustness to label noise (Table 4)

Config: [`configs/cifar10_noisy_resnet32.yaml`](configs/cifar10_noisy_resnet32.yaml). File config chỉ ghi giá trị, còn lý do của từng giá trị nằm ở các bảng dưới đây.

**Mục tiêu.** Tái hiện Table 4 (SAM, trang 7): accuracy trên tập test **sạch** của CIFAR-10.

Repo tái lập 3 phương pháp: SGD, SAM, Bootstrap + SAM. Table 4 còn có Mixup và Bootstrap + Mixup, repo không tái lập hai dòng này.

Số của bài báo:

| Phương pháp | 20% | 40% | 60% | 80% |
|---|---|---|---|---|
| SGD | 84.8 | 68.8 | 48.2 | 26.2 |
| SAM | 95.1 | 93.4 | 90.5 | 77.9 |
| Bootstrap + SAM | 95.4 | 94.2 | 91.8 | 79.9 |

**Ký hiệu nguồn**, xếp theo thứ tự ưu tiên. Nguồn trên nói rõ thì không dùng nguồn dưới.

| Ký hiệu | Nguồn | Vì sao dùng |
|---|---|---|
| **§3.3** | SAM, mục 3.3 (trang 7) | Phần đang tái hiện, là nguồn quyết định |
| **SAM** | SAM, các phần chung: §2, §3, §3.1, Appendix C.1 | Cách chính các tác giả train trên CIFAR-10 |
| **Jiang** | Jiang et al. 2020, *Beyond Synthetic Noise* | §3.3 ghi *"following Jiang et al. (2019)"* |
| **repo** | Code chính thức [google-research/sam](https://github.com/google-research/sam) | Chi tiết cài đặt của chính tác giả mà bài báo không ghi |

---

## 1. Phương pháp

| Thông số | Chọn | Nằm ở đâu | Lý do |
|---|---|---|---|
| Các phương pháp | `sgd`, `sam`, `bootstrap_sam` | §3.3: *"We evaluate five variants of model training: standard SGD, SGD with Mixup, SAM, and "bootstrapped" variants of SGD with Mixup and SAM"* | Chọn 3 trong 5 biến thể của bài báo, bỏ hai biến thể Mixup |

## 2. Dữ liệu

| Thông số | Chọn | Nằm ở đâu | Lý do |
|---|---|---|---|
| Tập dữ liệu | CIFAR-10 | §3.3: *"the classical noisy-label setting for CIFAR-10"* | Bài báo nói trực tiếp |
| Tập train | **50,000 ảnh** (toàn bộ train, nhãn nhiễu), không tách validation (`validation_fraction: 0.0`) | repo SAM: `use_test_set` mặc định `True`, tức train trên toàn bộ train set. Jiang B.2: *"Both datasets contain 50,000 training and 10,000 validation images"* (validation của Jiang chính là tập test 10k) | Bài báo không nói lần train cuối dùng 45k hay 50k. Code tác giả SAM và Jiang đều dùng đủ 50k |
| Đánh giá mỗi epoch | Trên **tập test sạch 10k** (cột `val_*` trong log thật ra là test) | repo SAM với `use_test_set=True` cũng đánh giá trên test mỗi epoch | Chỉ để theo dõi, **không** dùng để chọn checkpoint hay chỉnh tham số |
| Tập test | 10,000 ảnh test chính thức, nhãn sạch | §3.3: *"the test set remains unmodified (i.e., clean)"* | Bài báo nói trực tiếp |
| Kiểu nhiễu | Mỗi nhãn train, **độc lập**, đổi với xác suất p sang một trong 9 lớp **khác**, chọn đều | §3.3 chỉ ghi *"randomly flipped"*. Jiang §3 (trang 3): *"the label of each training example is independently changed to a random incorrect class with a probability p"* | §3.3 không định nghĩa rõ nhưng làm theo Jiang, và Jiang định nghĩa rõ |
| Mức nhiễu p | 0.2, 0.4, 0.6, 0.8 | §3.3, Table 4: *"Noise rate (%) 20 40 60 80"* | Bài báo nói trực tiếp |
| Seed nhiễu | `1000 + seed`, dùng chung cho mọi phương pháp | Không có trong bài báo | Không phải siêu tham số, mà là điều kiện so sánh công bằng: cùng seed thì mọi phương pháp train trên đúng một bộ nhãn sai |
| Augmentation | basic: lật ngang (p=0.5), pad 4 kiểu reflect, crop ngẫu nhiên 32×32 | SAM §3.1 (trang 5): *"All results use basic data augmentations (horizontal flip, padding by four pixels, and random crop)"*. repo `augmentation.py`, hàm `weak_image_augmentation`: `tf.pad(image, ..., mode='REFLECT')`, `tf.image.random_crop`, `tf.image.random_flip_left_right` | Bài báo ghi pad 4 pixel nhưng không ghi kiểu pad. Code tác giả pad kiểu reflect, nên dùng reflect. §3.3 không nhắc lại augmentation, câu "All results" nằm ở §3.1 |
| Chuẩn hóa ảnh | Chia 255, trừ mean, chia std theo kênh: mean (0.4914, 0.4822, 0.4465), std (0.2470, 0.2435, 0.2616) | repo `dataset_source.py`: `image = tf.cast(image, tf.float32) / 255.0`, `image = (image - self._image_mean) / self._image_std`, cùng các hằng số này | Bài báo không ghi. Bắt buộc phải có, nên dùng đúng hằng số của code tác giả |

## 3. Model

| Thông số | Chọn | Nằm ở đâu | Lý do |
|---|---|---|---|
| Kiến trúc | ResNet-32 (6n+2, n=5, kênh 16/32/64), 464,154 tham số | §3.3: *"we train a simple model of similar size (ResNet-32) for 200 epochs"*. Jiang B.2: *"The ResNet-32 (He et al., 2016)"* | Bài báo nói trực tiếp. repo không có ResNet-32, nên model viết theo He et al. 2016 |
| Khởi tạo | Ngẫu nhiên, train từ đầu. Conv: Kaiming normal (fan_out). FC: U(−1/√10, 1/√10), bias 0. BN: scale 1, bias 0 | §3.3 không nhắc pretrain. Jiang §5.1 (trang 6): *"we train networks from scratch using ResNet-32 for CIFAR"*. repo `models/utils.py`: `variance_scaling(2.0, 'fan_out', 'normal')` cho conv; FC `uniform(...) * 1/sqrt(num_units_out)` | Cách khởi tạo bài báo không ghi. Dùng đúng cách của code tác giả cho model CIFAR |

## 4. Huấn luyện

| Thông số | Chọn | Nằm ở đâu | Lý do |
|---|---|---|---|
| Epochs | 200 | §3.3: *"for 200 epochs"* | Bài báo nói trực tiếp |
| Batch size | 256 | SAM C.1 (trang 15): *"For CIFAR10 ... we use a batch size of 256"* | C.1 áp cho mọi thực nghiệm CIFAR-10. Không dùng 128 của Jiang vì SAM có số riêng |
| Micro-batch | 1 (không chia), ε của SAM tính trên cả batch 256 | `num_micro_batches: 1` | Chọn để chạy nhanh trên 1 GPU T4. **Khác bài báo**: SAM §3.1 train trên 8 GPU, mỗi GPU tính ε riêng trên 32 ảnh (m = 32). Footnote 5 ghi không đồng bộ ε giúp SAM tốt hơn. Muốn giống bài báo thì thêm `--num-micro-batches 8` ([`noise_train_step.py`](sam_pytorch/noise_train_step.py)) |
| BN | Thống kê BN trên cả batch 256. Momentum 0.1, eps 1e-5 (theo quy ước PyTorch) | repo `models/utils.py`: `nn.BatchNorm.partial(..., momentum=0.9, epsilon=1e-5)` | Momentum 0.9 của flax tương đương momentum 0.1 của PyTorch |
| Learning rate | 0.1 | SAM Table 6 (trang 16), dòng CIFAR-10 *"WRN 28-10 (200 epochs)"*: 0.1. repo: `learning_rate` = 0.1. Jiang B.2: *"starting learning rate as 0.1"* | §3.3 không ghi. Mọi nguồn liên quan đều là 0.1, không nguồn nào cho số khác |
| Weight decay | 5e-4 | SAM Table 6, cùng dòng trên: 0.0005 | §3.3 không ghi (xem mục 7). Đây là số được công bố trong chính bài SAM cho thiết lập gần nhất: CIFAR-10, 200 epoch, batch 256, LR 0.1. repo mặc định 0.001 chỉ là giá trị mặc định của cờ |
| Dạng weight decay | L2 trong loss, trên mọi tham số (kể cả BN và bias), hệ số `0.5·wd·‖w‖²` | SAM §2, công thức (1): *"min_w L^SAM_S(w) + λ‖w‖²₂"*, với `w` là toàn bộ tham số. repo: `l2_reg * 0.5 * weight_l2` | Bài báo đặt L2 trên toàn bộ `w`. Hệ số 0.5 lấy từ code vì số WD trong Table 6 là giá trị `weight_decay` của chính code đó (xem mục 7) |
| Optimizer | SGD, momentum 0.9, **Nesterov** (`nesterov: true`) | SAM Algorithm 1 (trang 4): base optimizer là SGD. SAM C.1 (trang 16): *"SGD optimizer and momentum = 0.9"*. repo `flax_training.py`: Nesterov mặc định bật | Bài báo không ghi Nesterov. Dùng theo mặc định code tác giả |
| Lịch LR | Cosine, không warmup | SAM C.1 (trang 15–16, thực nghiệm ImageNet): *"The initial learning rate is set to 1.0 and decayed using a cosine schedule"*. SAM C.2 (fine-tuning): *"cosine learning rate decay"* | Bài báo không ghi lịch LR cho CIFAR. Cosine là lịch LR duy nhất bài báo dùng. Bài báo không nhắc warmup cho các lịch này nên không dùng |
| Gradient clipping | Chuẩn gradient tối đa 5.0 (`gradient_clip_norm: 5.0`) | repo `flax_training.py`: gradient clipping mặc định 5.0 | Bài báo không nhắc. Dùng theo mặc định code tác giả |
| Label smoothing | **Không dùng** | Bài báo không nhắc | Không có trong bài báo nên không thêm |
| Model báo cáo | Epoch 200 (cuối), không chọn checkpoint | §3.3: *"for 200 epochs"*. Table 4: *"Test accuracy on the clean test set"* | Train đủ 200 epoch rồi đánh giá model đó, không thêm bước chọn checkpoint mà bài báo không mô tả. Kết quả ghi trong `results.txt`, mục `[test_metrics]` |
| Seed | 1 (mới chạy 1 seed; chạy thêm bằng `--seed 1 2 3 4 5`) | SAM §3.1: *"We run five independent replicas"* | Câu này ở §3.1. Table 4 không có ±, nên không chắc §3.3 chạy 5 lần. Nên chạy 5 seed nếu đủ tài nguyên |

## 5. SAM

| Thông số | Chọn | Nằm ở đâu | Lý do |
|---|---|---|---|
| ρ | 0.1 cho 20/40/60%, 0.05 cho 80% | §3.3: *"we use ρ = 0.1 for all noise levels except 80%, for which we use ρ = 0.05 for more stable convergence"* | Bài báo nói trực tiếp. Khớp Table 8 nếu đọc số trong bảng là error rate (chú thích ghi "accuracy", nhưng số nhỏ nhất rơi đúng vào các ρ này) |
| ρ khi Bootstrap | Giống trên, ở cả hai giai đoạn | §3.3: *"When applying SAM, we use ρ = ..."* | Câu áp cho mọi lần dùng SAM |
| Chuẩn của ε | L2 (p = 2): `ε = ρ·g / ‖g‖₂` | SAM §2 (trang 3): *"p = 2 is typically optimal"* | SAM chuẩn, dùng cho mọi thực nghiệm |
| Xấp xỉ gradient | Bậc một, bỏ số hạng Hessian | SAM §2 (trang 4): *"we drop the second-order terms"* | SAM chuẩn |

## 6. Bootstrap

| Thông số | Chọn | Nằm ở đâu | Lý do |
|---|---|---|---|
| Bootstrap | (1) train SAM 200 epoch, (2) model đó dự đoán nhãn (lớp argmax) cho 50k ảnh train, (3) train lại từ đầu trên nhãn đó | §3.3: *"first trained as usual and then retrained from scratch on the labels predicted by the initially trained model"* | Ba bước khớp câu của bài báo. "Labels predicted" là lớp model dự đoán, tức argmax. Dự đoán dùng ảnh gốc (không augmentation), như khi đánh giá |

## 7. Những điểm chưa có bằng chứng

Các điểm sau bắt buộc phải có một giá trị, nhưng bài báo không ghi rõ. Repo dùng số hoặc cách đọc gần nhất với bài báo:

| Điểm | Đang chọn | Tình trạng |
|---|---|---|
| Weight decay cho ResNet-32 ở §3.3 | 5e-4 | §3.3 không ghi. Lấy từ Table 6 (CIFAR-10, 200 epoch), là số gần nhất có trong bài báo |
| Hệ số 0.5 của L2 | `0.5·wd·‖w‖²` | Công thức (1) viết `λ‖w‖²`, không có 0.5. Repo dùng 0.5 vì số WD của Table 6 được tạo ra bởi code dùng 0.5. Nếu đọc đúng chữ công thức (1) thì L2 mạnh gấp đôi |
| Nesterov, gradient clipping 5.0 | Bật | Bài báo không ghi. Lấy theo mặc định code tác giả |
| Micro-batch | 1 (ε trên cả batch 256) | Bài báo (§3.1) dùng m = 32 trên 8 GPU. Đây là khác biệt đã biết |
| Tập train 45k hay 50k | 50k | Bài báo không ghi. Code tác giả và Jiang dùng 50k |
| Kích thước ResNet-32 | ResNet-32 chuẩn của He et al. 2016 (16/32/64 kênh, 0.46M tham số, shortcut option A) | Bài báo chỉ ghi "a simple model of similar size (ResNet-32)". Jiang B.2 (trang 16) ghi *"training ResNet-32 on the clean training dataset, the validation accuracy reaches 95.2% on CIFAR-10"*. Con số này cao hơn nhiều so với mức thường thấy của ResNet-32 chuẩn trên CIFAR-10 sạch (khoảng 92–93%). Code của Jiang (MentorMix) dùng ResNet-32 **rộng hơn** (64/128/256/512 kênh, ~29M tham số). Model của SAM §3.3 thì vẫn chưa biết chắc |

## 8. Cách chạy

```bash
# 4 mức nhiễu x 3 phương pháp đã thực nghiệm.
# bootstrap_sam tái dùng run sam cùng mức nhiễu và seed (tự train nếu chưa có).
python -m sam_pytorch.label_noise --config configs/cifar10_noisy_resnet32.yaml \
  --method sgd sam bootstrap_sam --noise-rate 0.2 0.4 0.6 0.8 --seed 1

# Chạy thử nhanh (5 bước mỗi run)
python -m sam_pytorch.label_noise --config configs/cifar10_noisy_resnet32.yaml \
  --method sgd sam bootstrap_sam --noise-rate 0.8 --steps 5

# Gom kết quả thành bảng giống Table 4
python -m sam_pytorch.label_noise --config configs/cifar10_noisy_resnet32.yaml --summarize
```

| Mục | Ghi chú |
|---|---|
| Yêu cầu | Python ≥ 3.10, vì code có cú pháp `str \| Path` |
| Output | `outputs/authordefaults/cifar10_noisy/resnet32/noise_{20,40,60,80}/{sgd, sam, bootstrap_sam}/seed_{s}/` |
| Bảng tổng kết | `outputs/authordefaults/cifar10_noisy/summary.md` |
| File trong mỗi run | `results.txt` (accuracy test ở epoch cuối), `metrics.csv`, `train.log`, `config.yaml` (giá trị thật đã dùng, kể cả ρ) |
| Bootstrap | Có thêm `bootstrap_labels.pt` và `bootstrap_stats.txt`. Nếu train lại run gốc thì phải xóa `bootstrap_labels.pt` để nhãn được dự đoán lại |
| Số run | Mỗi mức nhiễu và mỗi seed: 3 run (SGD, SAM, Bootstrap + SAM), mỗi run 200 epoch |
