"""Vẽ biểu đồ so sánh bài báo và tái lập cho §3.3 (Table 4, nhãn nhiễu CIFAR-10).

Đường liền + chấm tròn đặc = số trong bài báo. Đường nét đứt + hình thoi rỗng
= số tái lập (trung bình qua các seed đã chạy xong).

Ví dụ:
  # Quét đệ quy mọi .../noise_*/<method>/seed_*/results.txt trong thư mục kết quả
  python visualize/plot_noise.py --results-dir "../KẾT QUẢ THỰC NGHIỆM"

  # Hoặc nhập số tái lập tay từ CSV (cột: method,20,40,60,80)
  python visualize/plot_noise.py --csv reproduced.csv --out outputs/noise_chart.png
"""
import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NOISE_RATES = (0.2, 0.4, 0.6, 0.8)

# Table 4 (SAM paper, trang 7), dòng "our implementation".
PAPER = {
    "sgd": (84.8, 68.8, 48.2, 26.2),
    "mixup": (93.0, 90.0, 83.8, 70.2),
    "bootstrap_mixup": (93.3, 92.0, 87.6, 72.0),
    "sam": (95.1, 93.4, 90.5, 77.9),
    "bootstrap_sam": (95.4, 94.2, 91.8, 79.9),
}

# Phương pháp được vẽ (đúng như slide), nhãn và màu.
STYLE = {
    "sgd": ("SGD", "#5B7FD6"),
    "sam": ("SAM", "#D2693A"),
    "bootstrap_sam": ("Boot+SAM", "#1F2A6B"),
    "mixup": ("Mixup", "#3A9E6F"),
    "bootstrap_mixup": ("Boot+Mixup", "#8E5BB5"),
}
DEFAULT_METHODS = ("sgd", "sam", "bootstrap_sam")


def _test_accuracy(results_path: Path):
    """Accuracy test (%) trong mục [test_metrics] của results.txt."""
    parts = results_path.read_text().split("[test_metrics]", 1)
    if len(parts) < 2:
        return None
    match = re.search(r"^accuracy:\s*([0-9.]+)", parts[1], re.MULTILINE)
    return float(match.group(1)) * 100 if match else None


def reproduced_from_runs(results_dir: Path, methods) -> dict:
    """Quét đệ quy results_dir tìm .../noise_<XX>/<method>/seed_*/results.txt.

    Kết quả có thể nằm rải rác trong nhiều thư mục con (mỗi máy chạy một phần).
    Mỗi ô = trung bình các seed; Mixup lấy α có trung bình cao nhất.
    """
    found = {}  # (method, noise %) -> {run name: [acc, ...]}
    for results_file in results_dir.rglob("seed_*/results.txt"):
        run_name = results_file.parent.parent.name
        noise_name = results_file.parent.parent.parent.name
        if not noise_name.startswith("noise_"):
            continue
        method = re.sub(r"_alpha[0-9.]+$", "", run_name)
        accuracy = _test_accuracy(results_file)
        if accuracy is not None:
            key = (method, int(noise_name[len("noise_"):]))
            found.setdefault(key, {}).setdefault(run_name, []).append(accuracy)

    result = {}
    for method in methods:
        row = []
        for rate in NOISE_RATES:
            runs = found.get((method, round(rate * 100)), {})
            means = [sum(v) / len(v) for v in runs.values()]
            row.append(max(means) if means else None)
        result[method] = row
    return result


def reproduced_from_csv(path: Path) -> dict:
    """CSV: method,20,40,60,80 (ô trống = chưa có)."""
    result = {}
    with open(path, newline="") as handle:
        for record in csv.DictReader(handle):
            method = record["method"].strip()
            result[method] = [float(record[str(round(r * 100))]) if record.get(str(round(r * 100)), "").strip()
                              else None for r in NOISE_RATES]
    return result


def plot(reproduced: dict, methods, out_path: Path) -> None:
    x = list(range(len(NOISE_RATES)))
    fig, ax = plt.subplots(figsize=(8.5, 3.8), dpi=200)

    # Thứ tự legend như slide: với mỗi phương pháp, "bài báo" rồi "tái lập".
    for method in methods:
        label, color = STYLE[method]
        ax.plot(x, PAPER[method], color=color, linewidth=2.2, marker="o",
                markersize=8, label=f"{label} · bài báo", zorder=3)
        values = reproduced.get(method)
        if values and any(v is not None for v in values):
            points = [(i, v) for i, v in zip(x, values) if v is not None]
            ax.plot([p[0] for p in points], [p[1] for p in points], color=color,
                    linewidth=2.2, linestyle=(0, (4, 2.5)), marker="D", markersize=6.5,
                    markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.4,
                    label=f"{label} · tái lập", zorder=4)

    ax.set_xticks(x, [f"{round(r * 100)}%" for r in NOISE_RATES])
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_ylim(20, 100)
    ax.set_yticks(range(20, 101, 10))
    ax.set_xlabel("Tỉ lệ nhãn nhiễu", fontsize=11, color="#404040", labelpad=8)
    ax.set_ylabel("Độ chính xác trên tập test sạch (%)", fontsize=11, color="#404040", labelpad=10)
    ax.yaxis.grid(True, color="#D9D9D9", linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#BFBFBF")
    ax.tick_params(axis="both", length=0, labelsize=11, colors="#595959", pad=8)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
              fontsize=11, labelcolor="#404040", handlelength=2.6)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--results-dir", type=Path,
                        help="Thư mục chứa kết quả (quét đệ quy), ví dụ 'KẾT QUẢ THỰC NGHIỆM'")
    source.add_argument("--csv", type=Path, help="CSV số tái lập: method,20,40,60,80")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS),
                        choices=list(STYLE), help="Phương pháp cần vẽ")
    parser.add_argument("--out", type=Path, default=None,
                        help="File ảnh (.png/.pdf/.svg). Mặc định <results-dir>/noise_chart.png")
    args = parser.parse_args()

    if args.results_dir:
        reproduced = reproduced_from_runs(args.results_dir, args.methods)
        default_out = args.results_dir / "noise_chart.png"
    else:
        reproduced = reproduced_from_csv(args.csv)
        default_out = Path("outputs/noise_chart.png")

    for method in args.methods:
        print(f"{method:>15}: paper={PAPER[method]}  reproduced={reproduced.get(method)}")
    plot(reproduced, args.methods, args.out or default_out)


if __name__ == "__main__":
    main()
