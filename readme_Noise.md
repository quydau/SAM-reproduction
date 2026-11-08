# Tái hiện SAM §3.3: Robustness to label noise (Table 4)

Config: [`configs/cifar10_noisy_resnet32.yaml`](configs/cifar10_noisy_resnet32.yaml). File config chỉ ghi giá trị, còn lý do của từng giá trị nằm ở các bảng dưới đây.

**Mục tiêu.** Tái hiện 5 dòng "our implementation" của Table 4 (SAM, trang 7): accuracy trên tập test **sạch** của CIFAR-10.

| Phương pháp | 20% | 40% | 60% | 80% |
|---|---|---|---|---|
| SGD | 84.8 | 68.8 | 48.2 | 26.2 |
| Mixup | 93.0 | 90.0 | 83.8 | 70.2 |
| Bootstrap + Mixup | 93.3 | 92.0 | 87.6 | 72.0 |
| SAM | 95.1 | 93.4 | 90.5 | 77.9 |
| Bootstrap + SAM | 95.4 | 94.2 | 91.8 | 79.9 |

**Ký hiệu nguồn**, xếp theo thứ tự ưu tiên. Nguồn trên nói rõ thì không dùng nguồn dưới.

| Ký hiệu | Nguồn | Vì sao dùng |
|---|---|---|
| **§3.3** | SAM, mục 3.3 (trang 7) | Phần đang tái hiện, là nguồn quyết định |
| **SAM** | SAM, các phần chung: §2, §3, §3.1, Appendix C.1 | Cách chính các tác giả train trên CIFAR-10 |
| **Jiang** | Jiang et al. 2020, *Beyond Synthetic Noise* | §3.3 ghi *"following Jiang et al. (2019)"* |
| **repo** | Code chính thức [google-research/sam](https://github.com/google-research/sam) | Chi tiết cài đặt của chính tác giả mà bài báo không ghi |
| **Mixup** | Zhang et al. 2018 và code [mixup-cifar10](https://github.com/facebookresearch/mixup-cifar10) | §3.3 dùng *"SGD with Mixup (Zhang et al., 2017)"* |

---

## 1. Phương pháp

| Thông số | Chọn | Nằm ở đâu | Lý do |
|---|---|---|---|
| Các phương pháp | `sgd`, `mixup`, `sam`, `bootstrap_mixup`, `bootstrap_sam` | §3.3: *"We evaluate five variants of model training: standard SGD, SGD with Mixup, SAM, and "bootstrapped" variants of SGD with Mixup and SAM"* | Đúng 5 biến thể bài báo liệt kê, đúng 5 dòng của Table 4. Khối trên của Table 4 là số trích từ bài khác, không cần chạy |

## 2. Dữ liệu

| Thông số | Chọn | Nằm ở đâu | Lý do |
|---|---|---|---|
| Tập dữ liệu | CIFAR-10 | §3.3: *"the classical noisy-label setting for CIFAR-10"* | Bài báo nói trực tiếp |
| Tập train | **45,000 ảnh** (nhãn nhiễu) + **5,000 validation** (nhãn sạch), chia ngẫu nhiên theo seed cố định (`validation_fraction: 0.1`). Chỉ lật nhãn phần 45k | SAM C.1 (trang 16): *"computing the accuracy on a (non-noisy) validation set composed of a random subset of 10% of the usual CIFAR training samples"* | Đồng bộ với các thí nghiệm CIFAR khác của repo (đều giữ 10% validation). Validation sạch dùng để theo dõi mỗi epoch nên tập test không bao giờ bị nhìn trong lúc train. **Tự chọn:** bài không nói lần train cuối dùng 45k hay 50k (repo tác giả mặc định `use_test_set=True` train 50k, Jiang B.2 cũng 50k); train 45k ít hơn 10% dữ liệu nên accuracy có thể thấp hơn Table 4 một chút |
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
| Micro-batch | 8 phần × 32 ảnh, mỗi phần có nhiễu SAM riêng, gradient lấy trung bình | SAM §3.1 (trang 5): *"we train all models on a single host having 8 Nvidia V100 GPUs. To compute the SAM update when parallelizing across multiple accelerators, we divide each data batch evenly among the accelerators, independently compute the SAM gradient on each accelerator, and average"*. Chú thích 5 (trang 5): *"Because SAM's performance is amplified by not syncing the perturbations, data parallelism is highly recommended"*. repo: `sync_perturbations` mặc định `False` | Câu này nằm ở §3.1, §3.3 không nhắc lại. Việc §3.3 cũng chạy trên 8 GPU đó là **suy luận** (xem mục 7). Nếu đúng thì 256 / 8 = 32 ảnh mỗi GPU. Repo chỉ có 1 GPU nên giả lập bằng 8 micro-batch ([`noise_train_step.py`](sam_pytorch/noise_train_step.py)) |
| Micro-batch cho SGD / Mixup | Cũng 8 × 32 | SAM §3.1: *"we train all models"* trên 8 GPU | Cùng suy luận như dòng trên. Áp cho mọi phương pháp để các phương pháp được train giống nhau |
| BN | Mỗi micro-batch 32 ảnh dùng thống kê BN của chính nó. Running stats lấy trung bình qua 8 phần. Momentum 0.1, eps 1e-5 (theo quy ước PyTorch) | SAM §3.1 mô tả 8 GPU, mỗi GPU một phần batch. repo `models/utils.py`: `nn.BatchNorm.partial(..., momentum=0.9, epsilon=1e-5)`, không truyền `axis_name`, nên không đồng bộ BN giữa các GPU. repo `eval_step`: *"# Averages the batch norm moving averages."* | Không đồng bộ BN thì mỗi GPU chỉ thấy 32 ảnh của mình. Momentum 0.9 của flax tương đương momentum 0.1 của PyTorch. Trung bình running stats mỗi bước giống hệt trung bình lúc eval, vì cập nhật EMA là tuyến tính |
| Learning rate | 0.1 | SAM Table 6 (trang 16), dòng CIFAR-10 *"WRN 28-10 (200 epochs)"*: 0.1. repo: `learning_rate` = 0.1. Jiang B.2: *"starting learning rate as 0.1"* | §3.3 không ghi. Mọi nguồn liên quan đều là 0.1, không nguồn nào cho số khác |
| Weight decay | 5e-4 | SAM Table 6, cùng dòng trên: 0.0005 | §3.3 không ghi (xem mục 7). Đây là số được công bố trong chính bài SAM cho thiết lập gần nhất: CIFAR-10, 200 epoch, batch 256, LR 0.1. repo mặc định 0.001 chỉ là giá trị mặc định của cờ |
| Dạng weight decay | L2 trong loss, trên mọi tham số (kể cả BN và bias), hệ số `0.5·wd·‖w‖²` | SAM §2, công thức (1): *"min_w L^SAM_S(w) + λ‖w‖²₂"*, với `w` là toàn bộ tham số. repo: `l2_reg * 0.5 * weight_l2` | Bài báo đặt L2 trên toàn bộ `w`. Hệ số 0.5 lấy từ code vì số WD trong Table 6 là giá trị `weight_decay` của chính code đó (xem mục 7) |
| Optimizer | SGD, momentum 0.9 (momentum thường, **không** Nesterov) | SAM Algorithm 1 (trang 4): base optimizer là SGD. SAM C.1 (trang 16, thực nghiệm ImageNet): *"Weight decay is set to 0.0001 with SGD optimizer and momentum = 0.9"* | Bài báo không ghi momentum cho CIFAR. 0.9 là số momentum duy nhất bài báo ghi. Bài báo không nhắc Nesterov nên không dùng |
| Lịch LR | Cosine, không warmup | SAM C.1 (trang 15–16, thực nghiệm ImageNet): *"The initial learning rate is set to 1.0 and decayed using a cosine schedule"*. SAM C.2 (fine-tuning): *"cosine learning rate decay"* | Bài báo không ghi lịch LR cho CIFAR. Cosine là lịch LR duy nhất bài báo dùng. Bài báo không nhắc warmup cho các lịch này nên không dùng |
| Gradient clipping | **Không dùng** | Bài báo không nhắc | Không có trong bài báo nên không thêm |
| Label smoothing | **Không dùng** | Bài báo không nhắc | Không có trong bài báo nên không thêm |
| Model báo cáo | Epoch 200 (cuối), không chọn checkpoint | §3.3: *"for 200 epochs"*. Table 4: *"Test accuracy on the clean test set"* | Train đủ 200 epoch rồi đánh giá model đó, không thêm bước chọn checkpoint mà bài báo không mô tả. `val_loss`/`val_error_rate` mỗi epoch đo trên 5k validation sạch, chỉ để theo dõi, không dùng để chọn gì. Tập test chỉ được đo **đúng một lần** trên model epoch cuối |
| Seed | 1 (chạy thêm bằng `--seed 1 2 3 4 5`) | SAM §3.1: *"We run five independent replicas"* | Câu này ở §3.1. Table 4 không có ±, nên không chắc §3.3 chạy 5 lần. Nên chạy 5 seed nếu đủ tài nguyên |

## 5. SAM

| Thông số | Chọn | Nằm ở đâu | Lý do |
|---|---|---|---|
| ρ | 0.1 cho 20/40/60%, 0.05 cho 80% | §3.3: *"we use ρ = 0.1 for all noise levels except 80%, for which we use ρ = 0.05 for more stable convergence"* | Bài báo nói trực tiếp. Khớp Table 8 nếu đọc số trong bảng là error rate (chú thích ghi "accuracy", nhưng số nhỏ nhất rơi đúng vào các ρ này) |
| ρ khi Bootstrap | Giống trên, ở cả hai giai đoạn | §3.3: *"When applying SAM, we use ρ = ..."* | Câu áp cho mọi lần dùng SAM |
| Chuẩn của ε | L2 (p = 2): `ε = ρ·g / ‖g‖₂` | SAM §2 (trang 3): *"p = 2 is typically optimal"* | SAM chuẩn, dùng cho mọi thực nghiệm |
| Xấp xỉ gradient | Bậc một, bỏ số hạng Hessian | SAM §2 (trang 4): *"we drop the second-order terms"* | SAM chuẩn |

## 6. Mixup và Bootstrap

| Thông số | Chọn | Nằm ở đâu | Lý do |
|---|---|---|---|
| α | 1, 8, 16, 32 (mỗi α một run) | §3.3: *"we tried all values of α∈{1, 8, 16, 32}"* | Bài báo nói trực tiếp |
| Báo cáo Mixup | α có accuracy test trung bình cao nhất ở từng mức nhiễu | §3.3: *"conservatively report the best score for each noise level"* | Bài báo nói trực tiếp. Lưu ý: chọn trên tập test nên số hơi lạc quan |
| λ | Một λ ~ Beta(α, α) cho mỗi minibatch 256 ảnh | Mixup `train.py`: `lam = np.random.beta(alpha, alpha)`, gọi một lần mỗi minibatch | §3.3 không mô tả cách trộn, nên lấy từ bài gốc. Minibatch là 256 theo SAM C.1 |
| Cặp trộn | Hoán vị ngẫu nhiên của chính batch | Mixup §2: *"mixup is applied to the same minibatch after random shuffling"*. `train.py`: `torch.randperm(batch_size)` | Như trên |
| Nhãn trộn | One-hot trộn `λ·y_i + (1−λ)·y_j` | Mixup: *"where y_i, y_j are one-hot label encodings"* | Như trên. Cho kết quả giống `λ·CE(y_a) + (1−λ)·CE(y_b)` |
| Khi nào trộn | Chỉ khi train. Đánh giá và dự đoán nhãn bootstrap thì dùng ảnh gốc | Mixup là kỹ thuật lúc train | Đánh giá phải trên dữ liệu thật |
| Bootstrap | (1) train model gốc 200 epoch, (2) model đó dự đoán nhãn (lớp argmax) cho 45k ảnh train, (3) train lại từ đầu trên nhãn đó | §3.3: *"first trained as usual and then retrained from scratch on the labels predicted by the initially trained model"* | Ba bước khớp câu của bài báo. "Labels predicted" là lớp model dự đoán, tức argmax. Dự đoán dùng ảnh gốc, như khi đánh giá |

## 7. Những điểm chưa có bằng chứng

Các điểm sau bắt buộc phải có một giá trị, nhưng bài báo không ghi rõ. Repo dùng số hoặc cách đọc gần nhất với bài báo:

| Điểm | Đang chọn | Tình trạng |
|---|---|---|
| Weight decay cho ResNet-32 ở §3.3 | 5e-4 | §3.3 không ghi. Lấy từ Table 6 (CIFAR-10, 200 epoch), là số gần nhất có trong bài báo |
| Hệ số 0.5 của L2 | `0.5·wd·‖w‖²` | Công thức (1) viết `λ‖w‖²`, không có 0.5. Repo dùng 0.5 vì số WD của Table 6 được tạo ra bởi code dùng 0.5. Nếu đọc đúng chữ công thức (1) thì L2 mạnh gấp đôi |
| §3.3 chạy trên 8 GPU | 8 micro-batch × 32 | Câu về 8 GPU nằm ở §3.1. §3.3 không nhắc phần cứng |
| α khi Bootstrap + Mixup | Cùng α ở hai giai đoạn | Bài báo chỉ ghi "bootstrapped variants of SGD with Mixup" và "tried all values of α" |
| Kích thước ResNet-32 | ResNet-32 chuẩn của He et al. 2016 (16/32/64 kênh, 0.46M tham số, shortcut option A) | Bài báo chỉ ghi "a simple model of similar size (ResNet-32)". Jiang B.2 (trang 16) ghi *"training ResNet-32 on the clean training dataset, the validation accuracy reaches 95.2% on CIFAR-10"*. Con số này cao hơn nhiều so với mức thường thấy của ResNet-32 chuẩn trên CIFAR-10 sạch (khoảng 92–93%). Vì vậy có thể ResNet-32 của Jiang (và của SAM) **rộng hơn** bản chuẩn. Chưa xác minh được kiến trúc chính xác |

## 8. Cách chạy

```bash
# 4 mức nhiễu x 5 phương pháp (Mixup và Bootstrap + Mixup chạy mỗi α trong mixup.alphas).
# bootstrap_* tái dùng run sam / mixup cùng mức nhiễu, seed, α (tự train nếu chưa có).
python -m sam_pytorch.label_noise --config configs/cifar10_noisy_resnet32.yaml \
  --method sgd mixup sam bootstrap_mixup bootstrap_sam --noise-rate 0.2 0.4 0.6 0.8 --seed 1

# Chỉ chạy một α cho Mixup
python -m sam_pytorch.label_noise --config configs/cifar10_noisy_resnet32.yaml \
  --method mixup bootstrap_mixup --mixup-alpha 8

# Chạy thử nhanh (5 bước mỗi run)
python -m sam_pytorch.label_noise --config configs/cifar10_noisy_resnet32.yaml \
  --method sgd mixup sam bootstrap_mixup bootstrap_sam --noise-rate 0.8 --mixup-alpha 8 --steps 5

# Gom kết quả thành bảng giống Table 4 (ghi ra outputs/cifar10_noisy/summary.md)
python -m sam_pytorch.label_noise --config configs/cifar10_noisy_resnet32.yaml --summarize
```

| Mục | Ghi chú |
|---|---|
| Yêu cầu | Python ≥ 3.10, vì code có cú pháp `str \| Path` |
| Output | `outputs/cifar10_noisy/resnet32/noise_{20,40,60,80}/{sgd, sam, bootstrap_sam, mixup_alpha{α}, bootstrap_mixup_alpha{α}}/seed_{s}/` |
| File trong mỗi run | `results.txt` (accuracy test ở epoch cuối), `metrics.csv`, `train.log`, `config.yaml` (giá trị thật đã dùng, kể cả ρ và α) |
| Bootstrap | Có thêm `bootstrap_labels.pt` và `bootstrap_stats.txt`. Nếu train lại run gốc thì phải xóa `bootstrap_labels.pt` để nhãn được dự đoán lại |
| Số run | Mỗi mức nhiễu và mỗi seed: 3 run (SGD, SAM, Bootstrap + SAM) + 4 run Mixup + 4 run Bootstrap + Mixup, mỗi run 200 epoch |
