"""Vẽ đường loss (train / validation) theo epoch từ các metrics.csv.

Xuất 2 hình riêng (mỗi hình 1 hàng x 4 cột): train loss và validation error;
mỗi cột là một mức nhiễu nhãn (20/40/60/80%), mỗi đường là một phương pháp.

Ví dụ:
  python visualize/plot_loss.py --results-dir "../KẾT QUẢ THỰC NGHIỆM" --out-dir outputs
"""
import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NOISE_RATES = (20, 40, 60, 80)
STYLE = {
    "sgd": ("SGD", "#5B7FD6"),
    "sam": ("SAM", "#D2693A"),
    "bootstrap_sam": ("Boot+SAM", "#1F2A6B"),
}


def load_runs(results_dir: Path):
    """Trả về {(method, noise%): {"epoch": [...], "train": [...], "val": [...]}}."""
    runs = {}
    for path in results_dir.rglob("seed_*/metrics.csv"):
        match = re.search(r"noise_(\d+)[/\\]([^/\\]+)[/\\]seed_", str(path))
        if not match:
            continue
        noise, method = int(match.group(1)), match.group(2)
        if method not in STYLE:
            continue
        with path.open() as f:
            rows = list(csv.DictReader(f))
        runs[(method, noise)] = {
            "epoch": [int(r["epoch"]) for r in rows],
            "train": [float(r["train_loss"]) for r in rows],
            "val": [float(r["validation_loss"]) for r in rows],
            "val_err": [float(r["val_error_rate"]) * 100 for r in rows],
        }
    return runs


def plot_grid(runs, key, ylabel, title, out: Path):
    """Một hàng 4 cột (mỗi cột một mức nhiễu), mỗi đường một phương pháp."""
    fig, axes = plt.subplots(1, len(NOISE_RATES), figsize=(16, 4.2), squeeze=False)
    for col, noise in enumerate(NOISE_RATES):
        ax = axes[0][col]
        for method, (label, color) in STYLE.items():
            run = runs.get((method, noise))
            if run:
                ax.plot(run["epoch"], run[key], color=color, label=label, linewidth=1.6)
        ax.grid(alpha=0.3)
        ax.set_title(f"Nhiễu nhãn {noise}%")
        ax.set_xlabel("Epoch")
        if col == 0:
            ax.set_ylabel(ylabel)
    axes[0][0].legend()
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print("Đã lưu:", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    runs = load_runs(args.results_dir)
    print("Số run:", len(runs))
    plot_grid(runs, "train", "Train loss*",
              "Train loss theo epoch, ResNet-32 trên CIFAR-10 nhiễu nhãn\n"
              "*Train loss của Boot+SAM tính trên nhãn đã dự đoán lại, không so trực tiếp với SGD/SAM",
              args.out_dir / "train_loss_curves.png")
    plot_grid(runs, "val_err", "Validation error (%)",
              "Validation error theo epoch, ResNet-32 trên CIFAR-10 nhiễu nhãn",
              args.out_dir / "val_error_curves.png")


if __name__ == "__main__":
    main()
