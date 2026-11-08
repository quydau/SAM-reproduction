# Tái lập thí nghiệm SAM bằng PyTorch

Repo phục vụ đồ án môn Học máy, tái lập và so sánh Sharpness-Aware Minimization (SAM) với SGD theo paper [*Sharpness-Aware Minimization for Efficiently Improving Generalization*](https://arxiv.org/abs/2010.01412) của Foret et al. (ICLR 2021). Mã nguồn tham khảo chính thức: [google-research/sam](https://github.com/google-research/sam). Repo JAX/Flax gốc được giữ riêng tại `../sam`; repo này dùng PyTorch và tự chọn CUDA, Apple MPS (GPU Apple Silicon), hoặc CPU.

## Câu hỏi và phạm vi đồ án

SAM thực hiện cập nhật SGD trên loss tại các trọng số đã bị perturb trong bán kính `rho`, nhằm tìm nghiệm có loss ổn định trong lân cận. Đồ án so sánh SGD và SAM với cùng dữ liệu, seed, khởi tạo, lịch học và các hyperparameter còn lại.

Các nhóm thực nghiệm theo kế hoạch:

- CIFAR-10 và CIFAR-100, WRN-16-8 huấn luyện từ đầu với `basic`, `cutout`, `autoaugment`.
- Fine-tune EfficientNet-B0 với checkpoint ImageNet RandAugment `efficientnet_b0.ra_in1k` trên Flowers102, Oxford-IIIT Pet, FGVC Aircraft và Stanford Cars.
- Fine-tune EfficientNet-B3 với weights `tf_efficientnet_b3.ns_jft_in1k` trên Flowers102, Stanford Cars và Oxford-IIIT Pet.
- Fine-tune thêm EfficientNet-B0 trên CIFAR-100 với cùng recipe EfficientNet.
- Fine-tune EfficientNet-B0 trên CIFAR-10 với cùng recipe và AutoAugment CIFAR.
- Fine-tune EfficientNet-B0 trên Food101 với recipe Flowers102; dùng split train/test chính thức và giữ 10% train làm validation stratified. Ảnh 224×224 dạng uint8 được cache memory-mapped ở `data/food101-cache/` (khoảng 15 GB) để không giữ toàn bộ ảnh trong RAM.
- Ba seed ghép cặp giữa SGD và SAM cho mỗi điều kiện; chọn checkpoint trên validation rồi đánh giá test một lần.

Paper báo cáo năm lần chạy độc lập cho fine-tuning. Ba seed hiện dùng là giới hạn phạm vi đồ án; vì vậy độ bất định thống kê sẽ khác paper. Ngoài ra, paper fine-tune EfficientNet-B7 và EfficientNet-L2 với checkpoint công khai, trong khi cấu hình T4 ở đây dùng EfficientNet-B0 với checkpoint ImageNet RandAugment của timm cho Flowers102, Oxford-IIIT Pet, FGVC Aircraft và Stanford Cars. Đây là tái lập có điều chỉnh về mô hình/phần cứng, không phải kết quả số học tương đương với thí nghiệm gốc.

## Hyperparameter và nguồn gốc

| Thí nghiệm | Giá trị cấu hình | Nguồn |
|---|---|---|
| CIFAR-10 / WRN-16-8 (paper dùng WRN-28-10; ở đây thay bằng WRN-16-8 nhỏ hơn) | Batch hiệu dụng 256, 200 epoch, LR 0.1, WD 0.0005, `rho=0.05`, cosine, SGD momentum 0.9, Nesterov | Paper Appendix C.1, Table 6 nêu batch/LR/WD/rho; repo chính thức dùng cosine, momentum 0.9, Nesterov và clipping mặc định 5 |
| CIFAR augmentation | Basic: reflect-pad 4, random crop + flip; AutoAugment CIFAR rồi basic; Cutout: mask 16×16 chung batch sau normalization | Paper Appendix C; thứ tự/policy/chi tiết mask theo repo chính thức |
| Fine-tune Flowers/Pet theo C.2 | WD 1e-5; LR 0.016 với batch 1024, LR 0.008 với batch 512; cosine; momentum 0.9; đóng băng thống kê BatchNorm; không augmentation. | Paper Appendix C.2. Repo SAM không dispatch Flowers/Pet; preprocessing ở đây dùng center crop, resize và ImageNet normalization, không có phép biến đổi ngẫu nhiên. |
| Fine-tune FGVC Aircraft/Stanford Cars | Dùng cùng recipe EfficientNet-B0, batch hiệu dụng 1024, 5.000 bước, LR 0.016; validation stratified 10% từ train chính thức | Cấu hình theo recipe C.2 của repo; tập Cars lấy từ mirror Hugging Face vì URL gốc không còn hoạt động. |
| EfficientNet-B3 JFT trên Flowers102/Pet/Cars | `tf_efficientnet_b3.ns_jft_in1k`, ảnh 300×300, batch hiệu dụng 1024; config Flowers102 dùng microbatch 32 × tích lũy 32 trên T4, 5.000 bước; giữ LR, optimizer và SAM như recipe B0 | Weights Noisy Student được train trên ImageNet-1k và JFT-300M; kích thước ảnh và normalization lấy từ pretrained config của timm. |
| Fine-tune đề xuất cho T4 | Batch hiệu dụng 256 (microbatch 16 × tích lũy 16), LR 0.004, 20,000 bước, WD 1e-5, `rho=0.05`, EfficientNet-B0 | Batch/LR/steps là suy luận để khớp xấp xỉ số mẫu đã thấy: LR scale tuyến tính từ C.2; 20k × 256 tương đương 5k × 1024. B0 là thay đổi để phù hợp bộ nhớ, không phải model trong paper |

Các giá trị cho CIFAR được nêu trực tiếp trong paper. `rho=0.05` là lựa chọn có căn cứ: paper dùng giá trị này cho CIFAR-10 và cũng báo cáo kết quả fine-tuning với `rho=0.05`; kết quả fine-tuning tốt nhất có thể dùng rho đã tinh chỉnh riêng. Với Flowers/Pet, LR 0.004 và batch hiệu dụng 256 là ngoại suy theo tỷ lệ từ 0.016 @ 1024 và 0.008 @ 512. Paper không xác nhận trực tiếp giá trị 0.004. Bốn cấu hình B0 Flowers/Pet/Aircraft/Cars dùng weights `efficientnet_b0.ra_in1k` từ timm; các cấu hình B0 khác vẫn dùng `IMAGENET1K_V1` từ torchvision. Cả hai checkpoint B0 nhận ảnh 224×224 với ImageNet mean/std; cách crop/resize theo code SAM, không dùng transform inference mặc định của model zoo.

Các cấu hình B3 giữ batch hiệu dụng và hyperparameter của recipe fine-tuning hiện tại. Config Flowers102 dùng microbatch 32 và tích lũy 32 bước cho hai GPU T4; nếu một GPU không đủ VRAM, giảm microbatch xuống 16 và tăng tích lũy lên 64. Ảnh train/validation được lưu dạng uint8 trên CPU; từng microbatch được chuẩn hóa trên CPU rồi chuyển lên GPU khi tính gradient. SAM tính gradient hai lượt bằng cách phát lại cùng microbatch và RNG, giữ phép cập nhật theo batch hiệu dụng. Data loader dùng center crop, bicubic resize và mean/std do `timm` khai báo cho checkpoint; ảnh train/validation/test không augmentation.

`nesterov: true` được đặt rõ trong optimizer SGD của repo SAM. Với CIFAR, `gradient_clip_norm: 5.0` và `label_smoothing: 0.0` giữ mặc định code path của repo gốc. Với fine-tuning EfficientNet, cấu hình dùng `gradient_clip_norm: 1.0` và `label_smoothing: 0.1` theo lệnh fine-tune EfficientNet trong README của [repo SAM gốc](../sam/README.md); đây là cấu hình lệnh ví dụ cho EfficientNet-L2 trên ImageNet, được áp dụng cho fine-tune B0 Flowers/Pet của đồ án, không phải giá trị được báo cáo trực tiếp trong paper C.2.

Trên T4 16 GB, cấu hình CIFAR dùng microbatch 64 × tích lũy 4; các config fine-tuning B0 dùng microbatch 16 × tích lũy 16. B3 Flowers102 có cấu hình riêng 32 × 32 bên dưới. Đây là ước lượng ban đầu, cần giảm microbatch nếu gặp out-of-memory. Tích lũy gradient giữ batch hiệu dụng cho bước tối ưu, nhưng BatchNorm trong WRN vẫn thống kê trên microbatch vật lý; điều này có thể làm kết quả lệch khỏi batch 256 thực sự. Fine-tuning cố định BatchNorm ở chế độ eval như C.2, nên vấn đề này không áp dụng cùng cách cho EfficientNet. Đối với Flowers/Pet, repo gốc không có nhánh dataset; theo C.2, loader dùng cùng preprocessing xác định cho train/validation/test: center crop 87.5% cạnh ngắn, resize bicubic về 224×224 và ImageNet normalization. Không áp dụng random crop, flip hay RandAugment.

Để phù hợp thiết kế đồ án, kế hoạch dùng validation split cố định 10% ở CIFAR và Flowers/Pet, cùng ba seed. Flowers102 gộp split `train` và `val` có sẵn của torchvision rồi chia stratified 90/10; Oxford-IIIT Pet chia stratified 90/10 từ split `trainval`. Đây là quy ước triển khai của repo; paper dùng validation 10% để chọn rho trong một số thí nghiệm nhưng không mô tả đúng split này cho toàn bộ pipeline. Không dùng test set để chọn hyperparameter hoặc checkpoint. Repo SAM khai báo tên Flowers/Pet trong enum nhưng không dispatch các dataset này trong `main`, nên split và preprocessing hiện tại là lựa chọn mới của đồ án, không phải port tương đương từ code SAM.

## Chạy thí nghiệm

Các YAML trong [`configs/`](configs/) là cấu hình mẫu được ghi chú nguồn trực tiếp và suy luận. `--optimizer` là bắt buộc; `--optimizer`, `--seed` và `--augmentation` (chỉ CIFAR) nhận **một hoặc nhiều giá trị**, và một lệnh sẽ chạy **tuần tự** mọi tổ hợp (mỗi tổ hợp là một run độc lập, có thư mục output riêng). Nếu bỏ `--seed`/`--augmentation`, dùng giá trị trong YAML. Ví dụ:

```bash
python -m sam_pytorch.train --config configs/cifar10_wrn16x8.yaml --augmentation basic --optimizer sgd --seed 1
python -m sam_pytorch.train --config configs/cifar10_wrn16x8.yaml --augmentation basic --optimizer sam --seed 1
python -m sam_pytorch.train --config configs/cifar10_wrn16x8.yaml --augmentation cutout --optimizer sam --seed 2
python -m sam_pytorch.train --config configs/cifar10_wrn16x8.yaml --augmentation autoaugment --optimizer sgd --seed 3
python -m sam_pytorch.train --config configs/cifar100_wrn16x8.yaml --augmentation basic --optimizer sam --seed 1
python -m sam_pytorch.finetune --config configs/flowers102_efficientnet_b0.yaml --optimizer sgd --seed 1
python -m sam_pytorch.finetune --config configs/food101_efficientnet_b0.yaml --optimizer sgd sam --seed 1
python -m sam_pytorch.finetune --config configs/oxford_iiit_pet_efficientnet_b0.yaml --optimizer sam --seed 2
python -m sam_pytorch.finetune --config configs/fgvc_aircraft_efficientnet_b0.yaml --optimizer sgd sam --seed 1
python -m sam_pytorch.finetune --config configs/stanford_cars_efficientnet_b0.yaml --optimizer sam --seed 1
python -m sam_pytorch.finetune --config configs/cifar10_efficientnet_b0.yaml --optimizer sgd sam --seed 1
python -m sam_pytorch.finetune --config configs/cifar100_efficientnet_b0.yaml --optimizer sgd sam --seed 1
python -m sam_pytorch.finetune --config configs/flowers102_efficientnet_b3_jft.yaml --optimizer sgd sam --seed 1
python -m sam_pytorch.finetune --config configs/stanford_cars_efficientnet_b3_jft.yaml --optimizer sgd sam --seed 1
python -m sam_pytorch.finetune --config configs/oxford_iiit_pet_efficientnet_b3_jft.yaml --optimizer sgd sam --seed 1

# Nhiều giá trị trong một lệnh (tuần tự): 3 seed x 3 augmentation x 2 optimizer = 18 run
python -m sam_pytorch.train --config configs/cifar10_wrn16x8.yaml \
  --augmentation basic cutout autoaugment --optimizer sgd sam --seed 1 2 3
python -m sam_pytorch.finetune --config configs/flowers102_efficientnet_b0.yaml --optimizer sgd sam --seed 1 2 3
python -m sam_pytorch.finetune --config configs/stanford_cars_efficientnet_b0.yaml --optimizer sgd sam --seed 1 2 3
```

### Chạy song song SGD và SAM trên 2 GPU T4 ở Kaggle (Flowers102 B3)

Trong Kaggle, bật Internet cho notebook nếu đây là lần tải đầu tiên. Flowers102 được tải/giải nén khi tạo data module, còn tensor ảnh được cache lười vào `data/flowers-102-hf/`; hai tiến trình cùng tạo cache lần đầu có thể ghi vào cùng file tạm. Vì vậy hãy chạy lệnh chuẩn bị **một lần và đợi kết thúc thành công** trước khi khởi chạy SGD/SAM. Module này tải/cache train, validation, test và checkpoint B3 tuần tự.

```bash
python -m sam_pytorch.prepare_dataset --dataset flowers102

mkdir -p logs
CUDA_VISIBLE_DEVICES=0 python -u -m sam_pytorch.finetune \
  --config configs/flowers102_efficientnet_b3_jft.yaml \
  --optimizer sgd --seed 1 > logs/flowers102_b3_sgd.log 2>&1 &
SGD_PID=$!
CUDA_VISIBLE_DEVICES=1 python -u -m sam_pytorch.finetune \
  --config configs/flowers102_efficientnet_b3_jft.yaml \
  --optimizer sam --seed 1 > logs/flowers102_b3_sam.log 2>&1 &
SAM_PID=$!
set +e
wait "$SGD_PID"; SGD_STATUS=$?
wait "$SAM_PID"; SAM_STATUS=$?
echo "SGD exit=$SGD_STATUS; SAM exit=$SAM_STATUS"
echo "Logs: logs/flowers102_b3_sgd.log, logs/flowers102_b3_sam.log"
if [ "$SGD_STATUS" -ne 0 ] || [ "$SAM_STATUS" -ne 0 ]; then exit 1; fi
```

`CUDA_VISIBLE_DEVICES=0` và `=1` giả định Kaggle hiển thị hai GPU theo thứ tự đó; kiểm tra bằng `nvidia-smi -L` nếu cần. Mỗi tiến trình chỉ thấy một GPU vật lý, được PyTorch đánh số nội bộ là `cuda:0`. Hai optimizer ghi vào thư mục output riêng. Nếu gặp CUDA out-of-memory, dừng cả hai tiến trình, đổi `batch_size` thành 16 và `gradient_accumulation_steps` thành 64 trong YAML rồi chạy lại; cả hai cách giữ `effective_batch_size: 1024`.

### Chạy song song SGD và SAM trên 2 GPU T4 ở Kaggle (Food101 B0)

Bật Internet trong Kaggle nếu đây là lần tải đầu tiên. Lệnh chuẩn bị tải split train/test và weights ImageNet của B0, sau đó tạo cache ảnh 224×224 ở `data/food101-cache/`. Cache tensor chiếm khoảng 15 GB; chờ lệnh prepare hoàn tất trước khi chạy hai tiến trình để chúng không ghi cache đồng thời.

```bash
python -m sam_pytorch.prepare_dataset --dataset food101

mkdir -p logs
CUDA_VISIBLE_DEVICES=0 python -u -m sam_pytorch.finetune \
  --config configs/food101_efficientnet_b0.yaml \
  --optimizer sgd --seed 1 > logs/food101_b0_sgd.log 2>&1 &
SGD_PID=$!
CUDA_VISIBLE_DEVICES=1 python -u -m sam_pytorch.finetune \
  --config configs/food101_efficientnet_b0.yaml \
  --optimizer sam --seed 1 > logs/food101_b0_sam.log 2>&1 &
SAM_PID=$!
set +e
wait "$SGD_PID"; SGD_STATUS=$?
wait "$SAM_PID"; SAM_STATUS=$?
echo "SGD exit=$SGD_STATUS; SAM exit=$SAM_STATUS"
echo "Logs: logs/food101_b0_sgd.log, logs/food101_b0_sam.log"
if [ "$SGD_STATUS" -ne 0 ] || [ "$SAM_STATUS" -ne 0 ]; then exit 1; fi
```

Các biến `CUDA_VISIBLE_DEVICES` giả định Kaggle hiển thị hai GPU theo thứ tự 0 và 1; kiểm tra bằng `nvidia-smi -L` nếu cần. Mỗi tiến trình dùng một GPU và lưu output riêng. Nếu gặp CUDA out-of-memory, dừng cả hai, đổi `batch_size` thành 32 và `gradient_accumulation_steps` thành 32 trong YAML rồi chạy lại; batch hiệu dụng vẫn là 1024.

### Chạy song song SGD và SAM trên 2 GPU T4 ở Kaggle (EfficientNet-B0 RandAugment)

Các cấu hình Flowers102, Oxford-IIIT Pet, FGVC Aircraft và Stanford Cars khởi tạo từ checkpoint `efficientnet_b0.ra_in1k`. Trong Kaggle bật Internet nếu đây là lần tải đầu tiên. Mỗi lệnh `prepare_dataset` tải/cache dữ liệu và tải checkpoint trước khi chạy hai tiến trình; đợi prepare kết thúc rồi chạy khối tương ứng. GPU 0 chạy SGD, GPU 1 chạy SAM.

#### FGVC Aircraft

```bash
python -m sam_pytorch.prepare_dataset --dataset fgvc_aircraft

mkdir -p logs
CUDA_VISIBLE_DEVICES=0 python -u -m sam_pytorch.finetune \
  --config configs/fgvc_aircraft_efficientnet_b0.yaml \
  --optimizer sgd --seed 1 > logs/aircraft_b0_sgd.log 2>&1 &
SGD_PID=$!
CUDA_VISIBLE_DEVICES=1 python -u -m sam_pytorch.finetune \
  --config configs/fgvc_aircraft_efficientnet_b0.yaml \
  --optimizer sam --seed 1 > logs/aircraft_b0_sam.log 2>&1 &
SAM_PID=$!
set +e
wait "$SGD_PID"; SGD_STATUS=$?
wait "$SAM_PID"; SAM_STATUS=$?
echo "SGD exit=$SGD_STATUS; SAM exit=$SAM_STATUS"
echo "Logs: logs/aircraft_b0_sgd.log, logs/aircraft_b0_sam.log"
if [ "$SGD_STATUS" -ne 0 ] || [ "$SAM_STATUS" -ne 0 ]; then exit 1; fi
```

#### Flowers102

```bash
python -m sam_pytorch.prepare_dataset --dataset flowers102_b0

mkdir -p logs
CUDA_VISIBLE_DEVICES=0 python -u -m sam_pytorch.finetune \
  --config configs/flowers102_efficientnet_b0.yaml \
  --optimizer sgd --seed 1 > logs/flowers102_b0_sgd.log 2>&1 &
SGD_PID=$!
CUDA_VISIBLE_DEVICES=1 python -u -m sam_pytorch.finetune \
  --config configs/flowers102_efficientnet_b0.yaml \
  --optimizer sam --seed 1 > logs/flowers102_b0_sam.log 2>&1 &
SAM_PID=$!
set +e
wait "$SGD_PID"; SGD_STATUS=$?
wait "$SAM_PID"; SAM_STATUS=$?
echo "SGD exit=$SGD_STATUS; SAM exit=$SAM_STATUS"
echo "Logs: logs/flowers102_b0_sgd.log, logs/flowers102_b0_sam.log"
if [ "$SGD_STATUS" -ne 0 ] || [ "$SAM_STATUS" -ne 0 ]; then exit 1; fi
```

#### Oxford-IIIT Pet

```bash
python -m sam_pytorch.prepare_dataset --dataset oxford_iiit_pet

mkdir -p logs
CUDA_VISIBLE_DEVICES=0 python -u -m sam_pytorch.finetune \
  --config configs/oxford_iiit_pet_efficientnet_b0.yaml \
  --optimizer sgd --seed 1 > logs/oxford_pet_b0_sgd.log 2>&1 &
SGD_PID=$!
CUDA_VISIBLE_DEVICES=1 python -u -m sam_pytorch.finetune \
  --config configs/oxford_iiit_pet_efficientnet_b0.yaml \
  --optimizer sam --seed 1 > logs/oxford_pet_b0_sam.log 2>&1 &
SAM_PID=$!
set +e
wait "$SGD_PID"; SGD_STATUS=$?
wait "$SAM_PID"; SAM_STATUS=$?
echo "SGD exit=$SGD_STATUS; SAM exit=$SAM_STATUS"
echo "Logs: logs/oxford_pet_b0_sgd.log, logs/oxford_pet_b0_sam.log"
if [ "$SGD_STATUS" -ne 0 ] || [ "$SAM_STATUS" -ne 0 ]; then exit 1; fi
```

#### Stanford Cars

```bash
python -m sam_pytorch.prepare_dataset --dataset stanford_cars

mkdir -p logs
CUDA_VISIBLE_DEVICES=0 python -u -m sam_pytorch.finetune \
  --config configs/stanford_cars_efficientnet_b0.yaml \
  --optimizer sgd --seed 1 > logs/stanford_cars_b0_sgd.log 2>&1 &
SGD_PID=$!
CUDA_VISIBLE_DEVICES=1 python -u -m sam_pytorch.finetune \
  --config configs/stanford_cars_efficientnet_b0.yaml \
  --optimizer sam --seed 1 > logs/stanford_cars_b0_sam.log 2>&1 &
SAM_PID=$!
set +e
wait "$SGD_PID"; SGD_STATUS=$?
wait "$SAM_PID"; SAM_STATUS=$?
echo "SGD exit=$SGD_STATUS; SAM exit=$SAM_STATUS"
echo "Logs: logs/stanford_cars_b0_sgd.log, logs/stanford_cars_b0_sam.log"
if [ "$SGD_STATUS" -ne 0 ] || [ "$SAM_STATUS" -ne 0 ]; then exit 1; fi
```

Mỗi tiến trình được gán một GPU T4, dùng microbatch và gradient accumulation như YAML, và ghi vào thư mục output riêng. Nếu gặp CUDA out-of-memory, giảm `training.batch_size` rồi tăng `training.gradient_accumulation_steps` tương ứng để giữ `effective_batch_size: 1024`. `flowers102` vẫn là lựa chọn prepare cho cấu hình B3 JFT; dùng `flowers102_b0` cho cấu hình B0 RandAugment.

### Chạy song song SGD và SAM trên 2 GPU T4 ở Kaggle (CIFAR10 EfficientNetB0)

Bật Internet trong Kaggle nếu đây là lần tải đầu tiên. Lệnh prepare tải dữ liệu CIFAR10 và weights ImageNet của EfficientNetB0 trước khi chạy hai tiến trình, tránh tải đồng thời. Augmentation AutoAugment được áp dụng khi train; ảnh không cần tạo cache 224×224.

```bash
python -m sam_pytorch.prepare_dataset --dataset cifar10

mkdir -p logs
CUDA_VISIBLE_DEVICES=0 python -u -m sam_pytorch.finetune \
  --config configs/cifar10_efficientnet_b0.yaml \
  --optimizer sgd --seed 1 > logs/cifar10_b0_sgd.log 2>&1 &
SGD_PID=$!
CUDA_VISIBLE_DEVICES=1 python -u -m sam_pytorch.finetune \
  --config configs/cifar10_efficientnet_b0.yaml \
  --optimizer sam --seed 1 > logs/cifar10_b0_sam.log 2>&1 &
SAM_PID=$!
set +e
wait "$SGD_PID"; SGD_STATUS=$?
wait "$SAM_PID"; SAM_STATUS=$?
echo "SGD exit=$SGD_STATUS; SAM exit=$SAM_STATUS"
echo "Logs: logs/cifar10_b0_sgd.log, logs/cifar10_b0_sam.log"
if [ "$SGD_STATUS" -ne 0 ] || [ "$SAM_STATUS" -ne 0 ]; then exit 1; fi
```

`CUDA_VISIBLE_DEVICES=0` và `=1` giả định Kaggle hiển thị hai GPU theo thứ tự đó. Mỗi tiến trình dùng một GPU riêng. Nếu gặp CUDA out-of-memory, dừng cả hai, đổi `batch_size` thành 32 và `gradient_accumulation_steps` thành 32 trong YAML; batch hiệu dụng vẫn là 1024.

Để chạy thử ngắn, truyền **một** trong `--epochs N` hoặc `--steps N`. Tham số được chọn ghi đè giới hạn tương ứng trong YAML; hai cờ không dùng đồng thời. Nếu không truyền cờ giới hạn, chương trình dùng nguyên `training.epochs` hoặc `training.max_steps` trong YAML. `--steps` là số lần cập nhật optimizer. Ví dụ `--steps 5` cho fine-tuning hoặc `--epochs 2` cho CIFAR. Tiến độ epoch được in ra terminal và append ngay vào `train.log`; `metrics.csv`, `metrics.txt` (báo cáo tiến độ trong lúc chạy) và checkpoint cũng được cập nhật sau mỗi epoch. Khi hoàn tất, `metrics.txt`/`results.txt` chứa báo cáo đầy đủ; test được đánh giá một lần. Báo cáo gồm `test_error`, classification report theo lớp và confusion matrix. Trên Mac, cài PyTorch/torchvision bản macOS; MPS được tự chọn khi có GPU Apple Silicon.

Các lệnh nhiều giá trị ở trên đã thay cho vòng lặp shell; thêm `--epochs 2` hoặc `--steps 5` để smoke-run. Những tổ hợp trong cùng một lệnh fine-tune thường chạy nối tiếp; các ví dụ Kaggle phía trên khởi chạy hai tiến trình độc lập trên hai GPU.

Output có dạng:

```text
outputs/{dataset}/{model}/{augmentation}/{optimizer}/seed_{seed}/
```

Ví dụ: `outputs/cifar10/wrn16x8/basic/sam/seed_1/`.

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
- [Checkpoint timm `efficientnet_b0.ra_in1k`](https://huggingface.co/timm/efficientnet_b0.ra_in1k), dùng cho fine-tune B0 trên Flowers102, Oxford-IIIT Pet, FGVC Aircraft và Stanford Cars.
- [Mirror Stanford Cars trên Hugging Face](https://huggingface.co/datasets/mteb/StanfordCars), giữ hai split train/test chính thức; torchvision ghi nhận URL tải gốc của bộ dữ liệu hiện không còn hoạt động.
