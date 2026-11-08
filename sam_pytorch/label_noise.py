"""SAM paper §3.3: ResNet-32 trên CIFAR-10 với nhãn nhiễu (Table 4).

Năm phương pháp như trong bài báo:
  sgd              SGD trên nhãn nhiễu
  mixup            SGD + Mixup trên nhãn nhiễu, chạy với từng alpha trong mixup.alphas
  sam              SAM trên nhãn nhiễu (rho lấy từ sam.rho_by_noise_rate)
  bootstrap_mixup  lấy model ``mixup`` (cùng alpha) dự đoán lại nhãn train,
                   rồi train lại SGD + Mixup từ đầu trên nhãn dự đoán
  bootstrap_sam    lấy model ``sam`` dự đoán lại nhãn train, rồi train lại SAM
                   từ đầu trên nhãn dự đoán

Bài báo: "For the Mixup baselines, we tried all values of alpha in {1, 8, 16, 32}
and conservatively report the best score for each noise level." Vì vậy với hai
phương pháp Mixup, bảng tổng kết chọn alpha có accuracy test trung bình cao nhất
ở từng mức nhiễu.

Ví dụ:
  python -m sam_pytorch.label_noise --config configs/cifar10_noisy_resnet32.yaml \\
      --method sgd mixup sam bootstrap_mixup bootstrap_sam \\
      --noise-rate 0.2 0.4 0.6 0.8 --seed 1
  python -m sam_pytorch.label_noise --config configs/cifar10_noisy_resnet32.yaml \\
      --method mixup --mixup-alpha 8
  python -m sam_pytorch.label_noise --config configs/cifar10_noisy_resnet32.yaml --summarize
"""

import argparse
import copy
import logging
import re
from pathlib import Path
from typing import Optional

import torch

from sam_pytorch.datasets import CIFAR10DataModuleForNoise
from sam_pytorch.models import create_model
from sam_pytorch.noise_train_step import use_micro_batch_train_step
from sam_pytorch.train_utils import load_config, seed_everything, train

# Thứ tự các hàng trong bảng tổng kết, giống Table 4.
METHODS = ("sgd", "mixup", "sam", "bootstrap_mixup", "bootstrap_sam")

# Phương pháp nào dùng Mixup, phương pháp nào dùng optimizer SAM.
MIXUP_METHODS = ("mixup", "bootstrap_mixup")
SAM_METHODS = ("sam", "bootstrap_sam")

# Bootstrap train lại trên nhãn dự đoán bởi model của phương pháp nào.
BOOTSTRAP_SOURCE_METHOD = {
    "bootstrap_mixup": "mixup",
    "bootstrap_sam": "sam",
}


# ----------------------------------------------------------------------------
# Tên run, thư mục output, rho
# ----------------------------------------------------------------------------

def _noise_rate_key(noise_rate: float) -> float:
    """Làm tròn mức nhiễu để dùng làm khóa dict (tránh lỗi 0.6000000001)."""
    return round(float(noise_rate), 4)


def _run_name(method: str, mixup_alpha: Optional[float]) -> str:
    """Tên thư mục của một run, ví dụ ``sam`` hoặc ``mixup_alpha8``."""
    if method in MIXUP_METHODS:
        return f"{method}_alpha{mixup_alpha:g}"
    return method


def run_dir_for(config: dict, noise_rate: float, method: str, seed: int,
                mixup_alpha: Optional[float] = None) -> Path:
    """outputs/cifar10_noisy/<model>/noise_<rate>/<run name>/seed_<seed>"""
    noise_folder = f"noise_{round(noise_rate * 100)}"
    return (Path(config.get("output_root", "outputs")) / "cifar10_noisy"
            / config["model"] / noise_folder / _run_name(method, mixup_alpha)
            / f"seed_{seed}")


def _rho_for(config: dict, noise_rate: float) -> float:
    """rho của SAM cho mức nhiễu này, đọc từ config."""
    rho_by_noise_rate = {_noise_rate_key(rate): float(rho)
                         for rate, rho in config["sam"]["rho_by_noise_rate"].items()}
    key = _noise_rate_key(noise_rate)
    if key not in rho_by_noise_rate:
        raise ValueError(f"No rho configured for noise rate {noise_rate}")
    return rho_by_noise_rate[key]


def _mixup_alphas(config: dict, alphas_from_cli: Optional[list[float]]) -> list[float]:
    """Danh sách alpha sẽ chạy: lấy từ dòng lệnh nếu có, không thì từ config."""
    if alphas_from_cli:
        return [float(alpha) for alpha in alphas_from_cli]
    return [float(alpha) for alpha in config["mixup"]["alphas"]]


def _is_complete(run_dir: Path) -> bool:
    return (run_dir / "results.txt").exists() and (run_dir / "last.pt").exists()


# ----------------------------------------------------------------------------
# Dữ liệu và config cho một run
# ----------------------------------------------------------------------------

def _make_data(config: dict, noise_rate: float, seed: int,
               train_labels: Optional[list[int]] = None,
               mixup_alpha: Optional[float] = None) -> CIFAR10DataModuleForNoise:
    training = config["training"]
    augmentation_params = config.get("augmentation_params", {})
    split = config.get("split", {})
    # Cùng (noise_rate, seed) thì mọi phương pháp dùng chung một bộ nhãn nhiễu.
    noise_seed = int(config["label_noise"].get("noise_seed_offset", 1000)) + seed

    return CIFAR10DataModuleForNoise(
        data_root=config["data_root"],
        batch_size=int(training["batch_size"]),
        augmentation=config["augmentation"],
        validation_fraction=float(split.get("validation_fraction", 0.1)),
        validation_seed=int(split.get("seed", seed)),
        num_workers=int(training.get("num_workers", 0)),
        crop_padding=int(augmentation_params.get("crop_padding", 4)),
        horizontal_flip_probability=float(
            augmentation_params.get("horizontal_flip_probability", 0.5)),
        label_noise_rate=noise_rate,
        noise_seed=noise_seed,
        train_labels_override=train_labels,
        mixup_alpha=mixup_alpha,
    )


def _run_config(base_config: dict, noise_rate: float, method: str, seed: int,
                mixup_alpha: Optional[float], args) -> dict:
    """Bản sao config, ghi rõ các giá trị thật sự dùng cho run này."""
    config = copy.deepcopy(base_config)
    config["seed"] = seed
    config["method"] = method
    config["label_noise"]["rate"] = noise_rate
    config["label_noise"]["noise_seed"] = (
        int(config["label_noise"].get("noise_seed_offset", 1000)) + seed)

    # Chỉ ghi rho cho phương pháp SAM, alpha cho phương pháp Mixup; còn lại để null.
    config.setdefault("sam", {})["rho"] = (
        _rho_for(config, noise_rate) if method in SAM_METHODS else None)
    config.setdefault("mixup", {})["alpha"] = (
        mixup_alpha if method in MIXUP_METHODS else None)

    # Ghi đè số epoch / số bước khi chạy thử nhanh.
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
        config["training"]["max_steps"] = None
    if args.steps is not None:
        config["training"]["max_steps"] = args.steps
        config["training"]["epochs"] = None
    return config


def _optimizer_name(method: str) -> str:
    """SAM cho các phương pháp SAM, SGD cho phần còn lại (kể cả Mixup)."""
    return "sam" if method in SAM_METHODS else "sgd"


def _train_model(model, data, config: dict, method: str, run_dir: Path) -> None:
    """Gọi ``train`` chung, với mỗi batch chia thành training.num_micro_batches
    micro-batch như 8 GPU trong bài báo (xem noise_train_step.py)."""
    num_micro_batches = int(config["training"].get("num_micro_batches", 1))
    with use_micro_batch_train_step(num_micro_batches):
        train(model, data, config, _optimizer_name(method), run_dir)


# ----------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------

def _train_on_noisy(base_config: dict, noise_rate: float, method: str, seed: int,
                    args, mixup_alpha: Optional[float] = None) -> Path:
    """Train ``sgd``, ``mixup`` hoặc ``sam`` trên nhãn nhiễu; trả về thư mục run."""
    config = _run_config(base_config, noise_rate, method, seed, mixup_alpha, args)
    run_dir = run_dir_for(config, noise_rate, method, seed, mixup_alpha)

    data = _make_data(config, noise_rate, seed, mixup_alpha=mixup_alpha)
    config["label_noise"]["num_wrong_train_labels"] = data.num_wrong_train_labels
    config["label_noise"]["num_train_labels"] = len(data.train_labels)

    seed_everything(seed)
    model = create_model(config["model"], int(config["num_classes"]))
    _train_model(model, data, config, method, run_dir)
    return run_dir


@torch.no_grad()
def _predict_train_labels(model, data) -> list[int]:
    """Nhãn argmax của model trên ảnh train (không augmentation, thứ tự cố định)."""
    device = next(model.parameters()).device
    model.eval()
    predicted_labels = []
    for images, _ in data.get_train_prediction_loader():
        logits = model(images.to(device))
        predicted_labels.extend(logits.argmax(dim=-1).cpu().tolist())
    return predicted_labels


def _load_finished_model(config: dict, checkpoint_path: Path):
    """Nạp model ở epoch cuối của một run đã xong (EMA nếu có)."""
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    model = create_model(config["model"], int(config["num_classes"])).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["ema"] or state["model"])
    return model


def _bootstrap(base_config: dict, noise_rate: float, method: str, seed: int,
               args, mixup_alpha: Optional[float] = None) -> Path:
    """Bootstrap (§3.3): "the model is first trained as usual and then retrained
    from scratch on the labels predicted by the initially trained model".

    Bước 1: lấy run gốc (``mixup`` cùng alpha, hoặc ``sam``); chưa có thì train.
    Bước 2: model gốc ở epoch cuối dự đoán argmax nhãn cho ảnh train.
    Bước 3: khởi tạo model mới và train lại từ đầu trên nhãn dự đoán,
            cùng optimizer / rho / alpha với run gốc.
    """
    source_method = BOOTSTRAP_SOURCE_METHOD[method]

    # ----- Bước 1: run gốc -----
    source_dir = run_dir_for(base_config, noise_rate, source_method, seed, mixup_alpha)
    source_checkpoint = source_dir / "last.pt"
    if not _is_complete(source_dir):
        print(f"  {method}: run {source_method} chưa có ở {source_dir}; train trước",
              flush=True)
        _train_on_noisy(base_config, noise_rate, source_method, seed, args, mixup_alpha)

    config = _run_config(base_config, noise_rate, method, seed, mixup_alpha, args)
    run_dir = run_dir_for(config, noise_rate, method, seed, mixup_alpha)
    run_dir.mkdir(parents=True, exist_ok=True)
    labels_path = run_dir / "bootstrap_labels.pt"

    # ----- Bước 2: dự đoán lại nhãn train (lưu lại để không phải dự đoán lần nữa) -----
    noisy_data = _make_data(config, noise_rate, seed)
    if labels_path.exists():
        predicted_labels = torch.load(labels_path)["predicted_labels"]
    else:
        source_model = _load_finished_model(config, source_checkpoint)
        predicted_labels = _predict_train_labels(source_model, noisy_data)

        num_labels = len(predicted_labels)
        agree_with_clean = sum(
            predicted == clean
            for predicted, clean in zip(predicted_labels, noisy_data.clean_train_labels))
        agree_with_noisy = sum(
            predicted == noisy
            for predicted, noisy in zip(predicted_labels, noisy_data.train_labels))

        torch.save({"predicted_labels": predicted_labels,
                    "train_indices": noisy_data.train_indices,
                    "source_checkpoint": str(source_checkpoint)}, labels_path)
        (run_dir / "bootstrap_stats.txt").write_text(
            f"source_checkpoint: {source_checkpoint}\n"
            f"noise_rate: {noise_rate}\n"
            f"train_labels: {num_labels}\n"
            f"noisy_labels_wrong_vs_clean: {noisy_data.num_wrong_train_labels / num_labels:.6f}\n"
            f"predicted_agree_with_clean: {agree_with_clean / num_labels:.6f}\n"
            f"predicted_agree_with_noisy: {agree_with_noisy / num_labels:.6f}\n")
        noisy_label_accuracy = 1 - noisy_data.num_wrong_train_labels / num_labels
        print(f"  bootstrap labels: {agree_with_clean / num_labels:.4f} agree with clean "
              f"labels (noisy labels: {noisy_label_accuracy:.4f})", flush=True)

    # ----- Bước 3: train lại từ đầu trên nhãn dự đoán -----
    data = _make_data(config, noise_rate, seed, train_labels=predicted_labels,
                      mixup_alpha=mixup_alpha)
    config["label_noise"]["num_wrong_train_labels"] = data.num_wrong_train_labels
    config["label_noise"]["num_train_labels"] = len(data.train_labels)
    config["bootstrap_source"] = str(source_checkpoint)

    seed_everything(seed)
    model = create_model(config["model"], int(config["num_classes"]))
    _train_model(model, data, config, method, run_dir)
    return run_dir


# ----------------------------------------------------------------------------
# Bảng tổng kết
# ----------------------------------------------------------------------------

def _test_accuracy(results_path: Path) -> Optional[float]:
    """Đọc accuracy test (%) từ mục [test_metrics] của results.txt."""
    text = results_path.read_text()
    sections = text.split("[test_metrics]", 1)
    if len(sections) < 2:
        return None
    match = re.search(r"^accuracy:\s*([0-9.]+)", sections[1], re.MULTILINE)
    return float(match.group(1)) * 100 if match else None


def _seed_accuracies(run_folder: Path) -> list[float]:
    """Accuracy test của mọi seed đã chạy xong trong một thư mục run."""
    accuracies = []
    for results_file in sorted(run_folder.glob("seed_*/results.txt")):
        accuracy = _test_accuracy(results_file)
        if accuracy is not None:
            accuracies.append(accuracy)
    return accuracies


def _mean_and_std(values: list[float]) -> tuple[float, float]:
    """Trung bình và độ lệch chuẩn mẫu (std = 0 khi chỉ có một giá trị)."""
    num_values = len(values)
    mean = sum(values) / num_values
    if num_values > 1:
        variance = sum((value - mean) ** 2 for value in values) / (num_values - 1)
        std = variance ** 0.5
    else:
        std = 0.0
    return mean, std


def summarize(config: dict) -> str:
    """Bảng accuracy test (%) kiểu Table 4: hàng = phương pháp, cột = mức nhiễu.

    Mỗi ô là trung bình ± độ lệch chuẩn qua các seed đã chạy xong (n = số seed).
    Với Mixup và Bootstrap + Mixup, ô lấy alpha có trung bình cao nhất ở mức
    nhiễu đó, như bài báo ("report the best score for each noise level").
    """
    results_root = Path(config.get("output_root", "outputs")) / "cifar10_noisy" / config["model"]
    noise_rates = [float(rate) for rate in config["label_noise"]["rates"]]
    mixup_alphas = [float(alpha) for alpha in config["mixup"]["alphas"]]

    header = "| Method | " + " | ".join(f"{round(rate * 100)}%" for rate in noise_rates) + " |"
    separator = "|" + "---|" * (len(noise_rates) + 1)
    table_lines = ["Test accuracy (%) on the clean CIFAR-10 test set.",
                   "Mean ± std over finished seeds; n = number of seeds.",
                   "Mixup rows: best alpha per noise level.", "",
                   header, separator]

    for method in METHODS:
        cells = []
        for noise_rate in noise_rates:
            noise_dir = results_root / f"noise_{round(noise_rate * 100)}"

            # Mỗi "lựa chọn" là một thư mục run: 1 thư mục, hoặc 1 thư mục cho mỗi alpha.
            if method in MIXUP_METHODS:
                candidates = [(alpha, noise_dir / _run_name(method, alpha))
                              for alpha in mixup_alphas]
            else:
                candidates = [(None, noise_dir / method)]

            best_cell, best_mean = "–", None
            for alpha, run_folder in candidates:
                accuracies = _seed_accuracies(run_folder)
                if not accuracies:
                    continue
                mean_accuracy, std_accuracy = _mean_and_std(accuracies)
                if best_mean is None or mean_accuracy > best_mean:
                    best_mean = mean_accuracy
                    best_cell = f"{mean_accuracy:.1f} ± {std_accuracy:.1f}, n={len(accuracies)}"
                    if alpha is not None:
                        best_cell += f", α={alpha:g}"
            cells.append(best_cell)
        table_lines.append(f"| {method} | " + " | ".join(cells) + " |")

    table = "\n".join(table_lines) + "\n"
    results_root.parent.mkdir(parents=True, exist_ok=True)
    (results_root.parent / "summary.md").write_text(table)
    return table


# ----------------------------------------------------------------------------
# Dòng lệnh
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/cifar10_noisy_resnet32.yaml")
    parser.add_argument("--method", nargs="+", choices=METHODS)
    parser.add_argument("--noise-rate", nargs="+", type=float,
                        help="Noise rates, e.g. 0.2 0.4 (default: label_noise.rates)")
    parser.add_argument("--seed", nargs="+", type=int, help="Seeds (default: config.seed)")
    parser.add_argument("--mixup-alpha", nargs="+", type=float,
                        help="Mixup alphas for mixup methods (default: mixup.alphas)")
    parser.add_argument("--summarize", action="store_true",
                        help="Only build the Table-4 style summary from finished runs")
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument("--epochs", type=int, help="Override training.epochs (quick run)")
    limits.add_argument("--steps", type=int, help="Override training.max_steps (quick run)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    config = load_config(args.config)
    if args.summarize:
        print(summarize(config))
        return
    if not args.method:
        parser.error("--method is required unless --summarize is given")

    noise_rates = args.noise_rate or [float(rate) for rate in config["label_noise"]["rates"]]
    seeds = args.seed or [int(config["seed"])]
    mixup_alphas = _mixup_alphas(config, args.mixup_alpha)
    methods = list(dict.fromkeys(args.method))  # bỏ trùng, giữ thứ tự

    # Liệt kê mọi run cần chạy: (seed, mức nhiễu, phương pháp, alpha).
    runs = []
    for seed in dict.fromkeys(seeds):
        for noise_rate in dict.fromkeys(noise_rates):
            for method in methods:
                if method in MIXUP_METHODS:
                    for alpha in mixup_alphas:
                        runs.append((seed, noise_rate, method, alpha))
                else:
                    runs.append((seed, noise_rate, method, None))

    for run_number, (seed, noise_rate, method, alpha) in enumerate(runs, start=1):
        rho_text = _rho_for(config, noise_rate) if method in SAM_METHODS else "N/A"
        alpha_text = f"{alpha:g}" if alpha is not None else "N/A"
        print(f"[{run_number}/{len(runs)}] seed={seed} noise={noise_rate} method={method} "
              f"rho={rho_text} mixup_alpha={alpha_text}", flush=True)
        if method in BOOTSTRAP_SOURCE_METHOD:
            _bootstrap(config, noise_rate, method, seed, args, alpha)
        else:
            _train_on_noisy(config, noise_rate, method, seed, args, alpha)
    print(summarize(config))


if __name__ == "__main__":
    main()
