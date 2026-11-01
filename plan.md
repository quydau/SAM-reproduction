# Kế hoạch repo PyTorch để thực nghiệm SAM và SGD

Tài liệu này mô tả thiết kế và cách vận hành repo PyTorch dùng để so sánh SAM/SGD. Repo JAX/Flax gốc được giữ nguyên.

## 1. Mục tiêu và ma trận thực nghiệm

Repo mới thực hiện hai nhóm thí nghiệm độc lập:

1. Huấn luyện từ đầu WRN-16-8 trên CIFAR-10, với ba biến thể augmentation: `basic`, `cutout`, `autoaugment`.
2. Fine-tune EfficientNet-B0 pretrained trên ImageNet cho Flowers102 và Oxford-IIIT Pet.

Mỗi lần gọi entry point chỉ chạy một tổ hợp dataset/augmentation, optimizer và seed. Thực hiện ba seed riêng biệt; ghép cùng seed giữa SGD và SAM để giảm nhiễu khi đối chiếu. Tổng số lần gọi:

| Nhóm | Cấu hình | Số lần chạy |
|---|---|---:|
| CIFAR-10 | 3 augmentation × 2 optimizer × 3 seed | 18 |
| Fine-tuning | 2 dataset × 2 optimizer × 3 seed | 12 |
| **Tổng** |  | **30** |

Các siêu tham số huấn luyện như epoch, batch size, scheduler, weight decay và cách khởi tạo được khai báo trong config, cố định trong mỗi phép so sánh SGD/SAM và được lưu lại cho từng run.

## 2. Cấu trúc repo

```text
sam-pytorch/
├── README.md
├── requirements.txt
├── configs/
│   ├── cifar10_wrn16x8.yaml
│   ├── flowers102_efficientnet_b0.yaml
│   └── oxford_iiit_pet_efficientnet_b0.yaml
├── sam_pytorch/
│   ├── train.py                 # WRN-16-8 trên CIFAR-10
│   ├── finetune.py              # EfficientNet-B0 trên Flowers/Pet
│   ├── datasets.py              # loader và split dữ liệu
│   ├── augmentation.py          # các pipeline augmentation
│   ├── models.py                # WRN và EfficientNet-B0
│   ├── optimizers.py            # SGD và SAM
│   ├── trainer.py               # train/evaluate loop, scheduler, checkpoint
│   └── reporting.py             # metric, classification report, results.txt
└── outputs/                 # kết quả cục bộ, không commit vào git
```

Thiết kế giữ một package chính và một thư mục config. Hai tác vụ có file entry point riêng là `train.py` và `finetune.py`; data và augmentation cũng tách thành file riêng để dễ tìm. Vòng lặp, scheduler và checkpoint dùng chung nằm trong `trainer.py`; metric và ghi kết quả nằm trong `reporting.py`. Không chia thành nhiều package con hoặc tạo framework tổng quát.

## 3. Trách nhiệm từng phần

- **`datasets.py`**: tải CIFAR-10, Flowers102 và Oxford-IIIT Pet; tạo train/validation/test split cố định và DataLoader.
- **`augmentation.py`**: định nghĩa ba pipeline CIFAR-10 cùng preprocessing/augmentation ảnh cho EfficientNet-B0.
- **`models.py`**: định nghĩa WRN-16-8 gọn cho ảnh CIFAR 32×32 và tạo EfficientNet-B0 từ torchvision weights đã chọn, thay classifier theo dataset. WRN-50-2 của torchvision không tương đương WRN-16-8 nên phần WRN cần module riêng.
- **`optimizers.py`**: tạo SGD và adapter SAM hai bước dựa trên `torch.optim.SGD`, gồm perturb, gradient bước hai và khôi phục tham số.
- **`trainer.py`**: loop train/evaluate dùng chung, scheduler và lưu checkpoint.
- **`train.py`**: entry point riêng cho huấn luyện WRN/CIFAR-10.
- **`finetune.py`**: entry point riêng cho fine-tuning EfficientNet-B0 trên Flowers/Pet.
- **`reporting.py`**: tính accuracy, precision, recall, F1, confusion matrix, `classification_report`; ghi `results.txt`.
- Người dùng gọi `train.py` hoặc `finetune.py` một lần cho mỗi optimizer/seed/augmentation; không có grid runner tự chạy hàng loạt.

## 4. Quy ước augmentation và dữ liệu

- **Basic:** random crop 32×32 có padding 4 và random horizontal flip.
- **Cutout:** basic cộng Cutout. Cần ghi rõ kích thước mask và cách chọn tâm mask. Repo cũ lấy một mask dùng chung cho batch; nếu muốn giữ đúng hành vi đó thì cần mô tả rõ, nếu chọn mask riêng từng ảnh thì cũng phải xem đây là thay đổi protocol.
- **AutoAugment:** dùng policy CIFAR-10 có sẵn của torchvision. Repo cũ chạy AutoAugment rồi áp dụng crop/flip basic; nếu giữ ý đồ đó, pipeline mới cũng cần cố định cùng thứ tự transform và ghi trong config.
- **Flowers102/Pet:** repo SAM khai báo nhưng không dispatch hai dataset này, nên theo Appendix C.2: không augmentation, dùng nhánh preprocessing ImageNet không augmentation cho train/validation/test. Nhánh ImageNet train trong repo (random distorted crop area 0.08–1.0, aspect 0.75–1.33, rồi horizontal flip; RandAugment chỉ khi bật `autoaugment`) được giữ tùy chọn nhưng không bật cho Flowers/Pet. Preprocessing đánh giá center crop với vùng giữa bằng 87.5% cạnh ngắn, resize bicubic về 224 và ImageNet normalization. Flowers102 gộp split train/val chính thức thành pool 2.040 ảnh rồi chia stratified 90/10 theo seed; Oxford-IIIT Pet chia stratified 90/10 từ split trainval chính thức. Split test chính thức được giữ nguyên và không dùng để chọn checkpoint/hyperparameter. Pet giữ thứ tự class của torchvision; Flowers dùng nhãn chỉ số 0–101 (torchvision trả nhãn gốc 1–102).
- Ghi phiên bản PyTorch/torchvision, pretrained weights chính xác, seed, split và preprocessing vào config lưu cùng run để kết quả có thể truy nguyên.

Repo cũ khai báo tên `Flowers` và `Oxford_IIIT_Pets`, nhưng `main` trong `sam_jax/train.py` chỉ dispatch CIFAR-10/100, Fashion-MNIST, SVHN và ImageNet. Do đó loader Flowers/Pet là phần cần thiết kế mới, không có pipeline cũ tương ứng để chuyển nguyên trạng.

## 5. Định dạng output cho từng lần chạy

Mỗi run phải lưu model và **toàn bộ kết quả sau huấn luyện vào file văn bản**, ngoài log theo epoch. Ví dụ:

```text
outputs/
├── cifar10/wrn16x8/cutout/sam/seed_1/
│   ├── config.yaml
│   ├── metrics.csv
│   ├── train.log
│   ├── results.txt
│   ├── last.pt
│   └── best.pt
└── flowers102/efficientnet_b0/sgd/seed_1/
    ├── config.yaml
    ├── metrics.csv
    ├── results.txt
    ├── last.pt
    └── best.pt
```

`results.txt` là bản tổng kết đầy đủ, có tối thiểu:

1. **Thông tin lần chạy:** tên thí nghiệm, dataset, model, augmentation, optimizer, seed, thời điểm bắt đầu/kết thúc, trạng thái hoàn tất. Trong khi chạy, `metrics.txt` được cập nhật trạng thái/metric mới nhất sau mỗi epoch; `train.log` ghi nối tiếp từng dòng tiến độ và thời điểm bắt đầu/kết thúc.
2. **Môi trường:** phiên bản Python/PyTorch/torchvision, thiết bị (CPU/GPU), tên GPU nếu có, pretrained weights và đường dẫn dữ liệu.
3. **Toàn bộ hyperparameter:** số epoch, batch size, learning rate ban đầu, lịch học và tham số lịch, momentum, Nesterov, weight decay, SAM `rho`, loss/label smoothing, gradient clipping nếu dùng, kích thước ảnh, normalization, augmentation và mọi tham số liên quan khác. Ghi cả hyperparameter không áp dụng bằng `N/A`, tránh bỏ sót trường.
4. **Split và lớp:** số mẫu train/validation/test, cách tạo split, danh sách class theo thứ tự index và số lớp.
5. **Kết quả theo epoch:** train/validation loss và accuracy, learning rate, thời gian epoch; có thể lưu dạng bảng trong `metrics.csv` và ghi metric cuối/best trong `results.txt`.
6. **Kết quả phân loại:** epoch được chọn theo validation, validation metrics và classification report; sau khi chốt checkpoint, đánh giá test đúng một lần và ghi test loss/accuracy, precision, recall, F1, `classification_report` theo từng lớp, macro average, weighted average, support và confusion matrix.
7. **Checkpoint:** tên file best/last, epoch tương ứng, tiêu chí chọn best và trạng thái optimizer/scheduler nếu cần resume.

Classification report nên được xuất dạng văn bản dễ đọc (ví dụ `sklearn.metrics.classification_report`) và kèm confusion matrix có nhãn class. Với CIFAR-10, Flowers102 và Pet, report test thể hiện precision/recall/F1 theo từng class; báo cáo tổng hợp macro/weighted để thấy cả chất lượng trung bình và ảnh hưởng của mất cân bằng lớp.

Sau khi đủ ba seed, kết quả từng run có thể được tổng hợp để tính mean ± std theo dataset/model/augmentation/optimizer. Mỗi `results.txt` là báo cáo đầy đủ của riêng một run.

Mỗi entry point yêu cầu `--optimizer sgd|sam`, chấp nhận `--seed` để ghi đè seed YAML, và có hai cờ chạy thử loại trừ lẫn nhau: `--epochs N` hoặc `--steps N`. Khi không truyền cờ giới hạn, giữ nguyên `training.epochs`/`training.max_steps` trong YAML. Cấu hình hiệu lực được lưu cùng output.

## 6. Các bước implement và file cũ để đối chiếu

| Bước | Phần triển khai | Đối chiếu repo cũ |
|---|---|---|
| 1 | Tạo repo PyTorch riêng, khai báo dependency gọn, YAML mẫu và quy tắc đặt tên output. Giữ repo cũ nguyên trạng. | `requirements.txt` liệt kê stack JAX/Flax/TensorFlow cũ; `sam_jax/train.py` dòng 86–99 tạo output theo siêu tham số và seed. |
| 2 | Viết loader CIFAR-10, split train/validation/test và augmentation độc lập. | `sam_jax/datasets/dataset_source.py` dòng 87–145 xử lý augmentation, batch, normalization và test loader; dòng 148–215 quy định split và normalization CIFAR. |
| 3 | Viết loader Flowers102/Pet bằng torchvision và cố định preprocessing/split. | `sam_jax/train.py` khai báo dataset ở dòng 35–39 nhưng nhánh dispatch dòng 108–141 không hỗ trợ Flowers/Pet; `sam_jax/datasets/dataset_source_imagenet.py` dòng 279–352 tham khảo cách tách train/eval preprocessing cho ảnh lớn. |
| 4 | Viết WRN-16-8 nhỏ (thay WRN-28-10 của paper) bằng PyTorch và dùng torchvision EfficientNet-B0/pretrained weights. | `sam_jax/models/wide_resnet.py` dòng 182–237 mô tả các group, multiplier và classifier; `sam_jax/models/load_model.py` dòng 77–80 chọn WRN-16-8. EfficientNet JAX nằm trong `sam_jax/efficientnet/efficientnet.py`; bản mới dùng torchvision thay vì port module Flax. |
| 5 | Viết SAM wrapper trên SGD, tách rõ `first_step`/`second_step` hoặc một bước cập nhật dễ đọc; xác định cách xử lý BatchNorm giữa hai forward. | `sam_jax/training_utils/flax_training.py` dòng 496–568 tính loss, gradient perturbation và nhánh SAM/SGD. |
| 6 | **Đã triển khai lại bằng PyTorch một GPU:** `trainer.py`: `make_scheduler` (71), `train_step` (234), `train_for_one_epoch` (280), `eval_step` (88), `eval_on_dataset` (99), `train` (317). `optimizers.py`: SAM `first_step` (66), `second_step` (91), L2 gradient (109). Entry point `train.py` và `finetune.py` gọi `train`. Cả SGD và SAM áp dụng L2 trước clipping; SAM tính gradient L2 ở cả hai lượt, chỉ cập nhật BatchNorm ở lượt đầu (hoặc đóng băng theo config). Không có pmap, collective hay distributed sampler. | Đối chiếu file SAM gốc đính kèm: `get_cosine_schedule` dòng 377–396 → `make_scheduler`; `train_step` dòng 466–568 → `train_step` và `SAM.first_step`/`second_step`; `eval_step` dòng 599–625 → `eval_step`; `eval_on_dataset` dòng 627–692 → `eval_on_dataset`; `train_for_one_epoch` dòng 701–760 → `train_for_one_epoch`; `train` dòng 763–916 → `train`. Không chuyển phần phân tán. |
| 7 | **Đã triển khai:** evaluator phân loại chọn `best.pt` bằng validation accuracy, chỉ đánh giá test một lần, ghi `classification_report`, confusion matrix có tên lớp và `test_error` vào cả `metrics.txt` và `results.txt`; lưu `metrics.csv`, `train.log`, `last.pt`, `best.pt`, `config.yaml`. Metrics CSV, báo cáo tiến độ và checkpoint được cập nhật sau mỗi epoch; epoch log đồng thời in ra màn hình và append vào file. Tự chọn CUDA/MPS/CPU để hỗ trợ GPU Apple Silicon. | `sam_jax/training_utils/flax_training.py` dòng 650–692 đánh giá dataset; dòng 170–195 lưu checkpoint; dòng 899–916 lưu tiến độ/checkpoint cuối. Repo cũ không ghi report phân loại dạng đầy đủ theo lớp, nên đây là bổ sung mới. |
| 8 | **Cách chạy:** gọi riêng entry point cho từng optimizer, seed và augmentation; hỗ trợ giới hạn thử nghiệm qua `--epochs` hoặc `--steps`, mặc định đọc giới hạn từ YAML. Tổng hợp ba seed sau khi chạy xong. | `sam_jax/train.py` dòng 86–99 gợi ý định danh theo hyperparameter/seed; dùng `flax_training.py` phần log/evaluation để đối chiếu nội dung metric. |

## 7. Nguyên tắc so sánh và hoàn tất

- Dùng ba seed cố định và ghép cùng seed giữa SGD/SAM; giữ split dữ liệu giống nhau giữa các optimizer.
- Trong mỗi cặp so sánh, giữ nguyên model initialization, batch order/seed, augmentation protocol, số epoch, scheduler, loss và weight decay; chỉ thay optimizer mechanism.
- Chọn `best.pt` dựa trên validation metric đã quy định trước. Chỉ evaluate test sau khi hoàn tất chọn checkpoint.
- Fine-tuning dùng cùng pretrained weights cho SGD/SAM trong một seed và cùng quy tắc freeze/unfreeze classifier/backbone.
- Mỗi run chỉ được xem là hoàn tất khi có `config.yaml`, `results.txt`, `metrics.csv`, `best.pt` và `last.pt`; cấu trúc output cho phép chạy lại riêng một tổ hợp bị thiếu.
- Không nhân bản chức năng đã có trong PyTorch/torchvision/sklearn. Chỉ tự viết phần kiến trúc WRN-16-8, thuật toán SAM nếu không dùng thư viện ngoài, và phần điều phối/lưu kết quả đặc thù cho thí nghiệm.
