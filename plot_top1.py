#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot Top-1 accuracy curves from federated-learning log files.

Example:
python plot_top1.py \
  --method "HeteroFL=/path/to/heterofl/log.txt" \
  --method "FedRoLeX=/path/to/fedrolex/log.txt" \
  --method "FedDropout=/path/to/feddropout/log.txt" \
  --method "FedCMR (Ours)=/path/to/fedour/log.txt" \
  --title "CIFAR-100 (Dirichlet: 0.1)" \
  --max-round 200 \
  --smooth-window 7 \
  --output "./figures/cifar100_dirichlet_0.1"
"""

import argparse
import csv
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt


def parse_top1_from_log(log_path):
    """
    Extract (comm_round, top1) pairs from logs.

    Supported patterns:
    1) Master starting one round ... (comm_round=12)
       The validation performance = {'loss': ..., 'top1': 23.45, ...}

    2) runtime: {'comm_round': 12, 'loss': ..., 'top1': 23.45, ...}
    """
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    round_pat = re.compile(r"comm_round[=\':\s]+(\d+)")
    top1_pat = re.compile(r"['\"]?top1['\"]?\s*:\s*([-+]?\d+(?:\.\d+)?)")

    pairs = []
    current_round = None

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Pattern 2: comm_round and top1 in the same line.
            if "top1" in line and "comm_round" in line:
                r_match = round_pat.search(line)
                t_match = top1_pat.search(line)
                if r_match and t_match:
                    pairs.append((int(r_match.group(1)), float(t_match.group(1))))
                    continue

            # Pattern 1: first read current round, then read validation top1.
            r_match = round_pat.search(line)
            if r_match:
                current_round = int(r_match.group(1))

            if "The validation performance" in line and "top1" in line:
                t_match = top1_pat.search(line)
                if current_round is not None and t_match:
                    pairs.append((current_round, float(t_match.group(1))))

    # Remove duplicated rounds: keep the last top1 for each round.
    round_to_top1 = {}
    for r, t in pairs:
        round_to_top1[r] = t

    pairs = sorted(round_to_top1.items(), key=lambda x: x[0])
    return pairs


def normalize_top1(values):
    """
    Convert percentage values such as 45.6 to ratio 0.456.
    If the curve is already in [0, 1], keep it unchanged.
    """
    if not values:
        return values

    vmax = max(values)
    if vmax > 1.5:
        return [v / 100.0 for v in values]
    return values


def moving_average(values, window):
    """
    Causal moving average. Keeps the same length as input.
    window=1 means no smoothing.
    """
    if window <= 1:
        return values

    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed.append(sum(values[start:i + 1]) / (i - start + 1))
    return smoothed


def method_style(method_name, index):
    """
    Make Ours red and bold. Other methods use publication-style dashed lines.
    """
    name_lower = method_name.lower()

    if ("ours" in name_lower) or ("fedcmr" in name_lower) or ("fedara" in name_lower) or ("fedour" in name_lower):
        return {
            "color": "red",
            "linestyle": "-",
            "linewidth": 3.2,
            "alpha": 0.95,
            "zorder": 10,
        }

    colors = [
        "#1f77b4",  # blue
        "#2ca02c",  # green
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#17becf",  # cyan
        "#bcbd22",  # olive
        "#e377c2",  # pink
        "#7f7f7f",  # gray
    ]
    linestyles = [
        (0, (6, 3)),
        (0, (2, 2)),
        (0, (8, 3, 2, 3)),
        (0, (4, 2)),
        (0, (3, 1, 1, 1)),
        (0, (1, 2)),
        (0, (5, 2, 1, 2)),
        "--",
    ]

    return {
        "color": colors[index % len(colors)],
        "linestyle": linestyles[index % len(linestyles)],
        "linewidth": 2.2,
        "alpha": 0.9,
        "zorder": 5,
    }


def save_extracted_csv(curves, output_prefix):
    csv_path = f"{output_prefix}_top1.csv"
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    max_len = max((len(v["rounds"]) for v in curves.values()), default=0)
    headers = []
    for method in curves:
        headers.extend([f"{method}_round", f"{method}_top1"])

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for i in range(max_len):
            row = []
            for method, data in curves.items():
                if i < len(data["rounds"]):
                    row.extend([data["rounds"][i], data["top1_raw"][i]])
                else:
                    row.extend(["", ""])
            writer.writerow(row)

    return csv_path


def plot_curves(curves, title, output_prefix, y_min=None, y_max=None):
    os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)

    # Paper-like figure size.
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=160)

    for idx, (method, data) in enumerate(curves.items()):
        style = method_style(method, idx)
        ax.plot(
            data["rounds"],
            data["top1_plot"],
            label=method,
            solid_capstyle="round",
            dash_capstyle="round",
            **style,
        )

    ax.set_title(title, fontsize=20, fontweight="bold", pad=12)
    ax.set_xlabel("Communication rounds", fontsize=18, labelpad=8)
    ax.set_ylabel("Top-1 Accuracy", fontsize=18, labelpad=8)

    # Unified x-axis range, usually 1--200.
    all_rounds = []
    for data in curves.values():
        all_rounds.extend(data["rounds"])
    if all_rounds:
        ax.set_xlim(0, max(all_rounds))

    if y_min is not None or y_max is not None:
        ax.set_ylim(y_min, y_max)
    else:
        # Auto y range with small margin.
        all_y = []
        for data in curves.values():
            all_y.extend(data["top1_plot"])
        if all_y:
            ymin = max(0.0, min(all_y) - 0.03)
            ymax = min(1.0, max(all_y) + 0.04)
            ax.set_ylim(ymin, ymax)

    ax.grid(True, linestyle=(0, (1.5, 3.0)), linewidth=0.8, alpha=0.65)
    ax.tick_params(axis="both", labelsize=14, direction="in", length=5, width=1.1)

    for spine in ax.spines.values():
        spine.set_linewidth(1.6)

    legend = ax.legend(
        loc="lower right",
        fontsize=11,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=0.88,
    )
    legend.get_frame().set_linewidth(1.0)

    fig.tight_layout()

    png_path = f"{output_prefix}.png"
    pdf_path = f"{output_prefix}.pdf"

    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


def parse_method_argument(item):
    if "=" not in item:
        raise ValueError(
            f"Invalid --method format: {item}\n"
            f'Expected format: --method "MethodName=/path/to/log.txt"'
        )
    name, path = item.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name:
        raise ValueError(f"Empty method name in: {item}")
    if not path:
        raise ValueError(f"Empty log path in: {item}")
    return name, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        help='Method and log path. Example: --method "FedAvg=/path/to/log.txt"',
    )
    parser.add_argument(
        "--title",
        default="CIFAR-100 (Dirichlet: 0.1)",
        help="Figure title.",
    )
    parser.add_argument(
        "--output",
        default="./figures/top1_curve",
        help="Output prefix without extension.",
    )
    parser.add_argument(
        "--max-round",
        type=int,
        default=400,
        help="Only keep rounds <= max-round. Default: 400.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=7,
        help="Moving-average window. Use 1 for raw noisy curves. Default: 7.",
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=None,
        help="Optional y-axis minimum, e.g., 0.0.",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Optional y-axis maximum, e.g., 0.55.",
    )
    args = parser.parse_args()

    curves = {}

    for item in args.method:
        method, log_path = parse_method_argument(item)
        pairs = parse_top1_from_log(log_path)

        if args.max_round is not None:
            pairs = [(r, t) for r, t in pairs if r <= args.max_round]

        if not pairs:
            print(f"[Warning] No top1 data found for {method}: {log_path}")
            continue

        rounds = [r for r, _ in pairs]
        top1_raw = normalize_top1([t for _, t in pairs])
        top1_plot = moving_average(top1_raw, args.smooth_window)

        curves[method] = {
            "rounds": rounds,
            "top1_raw": top1_raw,
            "top1_plot": top1_plot,
        }

        print(
            f"[OK] {method}: {len(rounds)} points, "
            f"round {min(rounds)}-{max(rounds)}, "
            f"best top1={max(top1_raw):.4f}, "
            f"last top1={top1_raw[-1]:.4f}"
        )

    if not curves:
        raise RuntimeError("No valid curves to plot. Please check log paths and log format.")

    csv_path = save_extracted_csv(curves, args.output)
    png_path, pdf_path = plot_curves(
        curves=curves,
        title=args.title,
        output_prefix=args.output,
        y_min=args.y_min,
        y_max=args.y_max,
    )

    print(f"[Saved] {png_path}")
    print(f"[Saved] {pdf_path}")
    print(f"[Saved] {csv_path}")


if __name__ == "__main__":
    main()

"""
模型异构 cifar100 迪利克雷
python plot_top1.py \
  --method "HeteroFL=/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/federated_averaging_cnn/heterofl_noiid_0.1中模型p=0.2/1781693912_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-heterofl/0/log.txt" \
  --method "FedRoLeX=/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/federated_averaging_cnn/fedrolex_noiid_0.1中模型p=0.2/1781701541_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedrolex/0/log.txt" \
  --method "FedDropout=/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/federated_averaging_cnn/feddropout_noiid_0.1中模型p=0.2/1781868763_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-feddropout/0/log.txt" \
  --method "FedCMR (Ours)=/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/federated_averaging_cnn/fed_our_noiid_0.1中模型异构p=0.2/1781673175_l2-0.0_lr-0.01_n_comm_rounds-400_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedour/0/log.txt" \
  --title "CIFAR-100 (Dirichlet: 0.1)" \
  --max-round 200 \
  --smooth-window 7 \
  --y-min 0.0 \
  --y-max 0.55 \
  --output "./figures/cifar100_dirichlet_0.1_smooth"






模型异构 cifar100 病理
python plot_top1.py \
    --method "HeteroFL=/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/federated_averaging_cnn/heterofl_病理分布中模型p=0.2/1781701133_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-heterofl/0/log.txt" \
    --method "FedRoLeX=/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/federated_averaging_cnn/fedrolex_病理分布中模型p=0.2/1781706582_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedrolex/0/log.txt" \
    --method "FedDropout=/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/federated_averaging_cnn/feddropout_病理分布中模型p=0.2/1781869054_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-feddropout/0/log.txt" \
    --method "FedCMR (Ours)=/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/federated_averaging_cnn/fed_our_病理分布中模型异构p=0.2/1781696269_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedour/0/log.txt" \
    --method "FedDF=/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/federated_averaging_cnn/feddf_病理分布中模型p=0.2/1781872330_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-feddf/0/log.txt" \
    --method "FedMD=/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/federated_averaging_cnn/fedmd_病理分布中模型p=0.2/1781921686_l2-0.0_lr-0.01_n_comm_rounds-400_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedmd/0/log.txt" \
    --method "Fjord=/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/federated_averaging_cnn/fjord_cifar100__病理分布pr0.2/1781935456_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fjord/0/log.txt" \
    --method "Fiarse=/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/federated_averaging_cnn/fiarse_cifar100_病理分布_p02_mild/1781938504_l2-0.0_lr-0.01_n_comm_rounds-400_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fiarse/0/log.txt" \
    --method "FedDse=/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/federated_averaging_cnn/feddse_cifar100_病理分布p02_rates_085_09_095_10/1782157726_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-feddse/0/log.txt" \
    --title "CIFAR-100 (Pathological: 10/100)" \
    --max-round 200 \
    --smooth-window 7 \
    --y-min 0.0 \
    --y-max 0.55 \
    --output "./figures/cifar100_病理分布_smooth"


模型异构 tiny 迪利克雷

python plot_top1.py \
    --method "HeteroFL=/data/zyz/Feds/FedGKD-main/data/checkpoint/imagenet_tiny/federated_averaging_cnn/heterofl_noiid_0.1中模型p=0.2/1781946938_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-heterofl/0/log.txt" \
    --method "FedRoLeX=/data/zyz/Feds/FedGKD-main/data/checkpoint/imagenet_tiny/federated_averaging_cnn/fedrolex_noiid_0.1中模型p=0.2/1781955476_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedrolex/0/log.txt" \
    --method "FedDropout=/data/zyz/Feds/FedGKD-main/data/checkpoint/imagenet_tiny/federated_averaging_cnn/feddropout_noiid_0.1中模型p=0.2/1781956160_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-feddropout/0/log.txt" \
    --method "FedCMR (Ours)=/data/zyz/Feds/FedGKD-main/data/checkpoint/imagenet_tiny/federated_averaging_cnn/fed_our_noiid_0.1中模型异构p=0.2/1781702353_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedour/0/log.txt" \
    --method "FedDF=/data/zyz/Feds/FedGKD-main/data/checkpoint/imagenet_tiny/federated_averaging_cnn/feddf_noiid_0.1中模型p=0.2/1781962953_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-feddf/0/log.txt" \
    --method "FedMD=/data/zyz/Feds/FedGKD-main/data/checkpoint/imagenet_tiny/federated_averaging_cnn/fedmd_noiid_0.1中模型p=0.2/1782007798_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedmd/0/log.txt" \
    --method "Fjord=/data/zyz/Feds/FedGKD-main/data/checkpoint/imagenet_tiny/federated_averaging_cnn/fjord_imagenet_tiny_dirichlet0.1_pr0.2/1782008085_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fjord/0/log.txt" \
    --method "Fiarse=/data/zyz/Feds/FedGKD-main/data/checkpoint/imagenet_tiny/federated_averaging_cnn/fiarse_dirichlet01_p02_mild/1781954207_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fiarse/0/log.txt" \
    --method "FedDse=/data/zyz/Feds/FedGKD-main/data/checkpoint/imagenet_tiny/federated_averaging_cnn/feddse_tiny_dirichlet01_p02_rates_085_09_095_10/1782138656_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-feddse/0/log.txt" \
    --title "Tiny-ImageNet (Practical: 0.1)" \
    --max-round 200 \
    --smooth-window 7 \
    --y-min 0.0 \
    --y-max 0.55 \
    --output "./figures/异构tiny迪利克雷top1"

模型异构 tiny 病理分布

python plot_top1.py \
    --method "HeteroFL=/data/zyz/Feds/FedGKD-main/data/checkpoint/imagenet_tiny/federated_averaging_cnn/heterofl_病理分布中模型p=0.2/1781948032_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-heterofl/0/log.txt" \
    --method "FedRoLeX=/data/zyz/Feds/FedGKD-main/data/checkpoint/imagenet_tiny/federated_averaging_cnn/fedrolex_病理分布中模型p=0.2/1781955882_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedrolex/0/log.txt" \
    --method "FedDropout=/data/zyz/Feds/FedGKD-main/data/checkpoint/imagenet_tiny/federated_averaging_cnn/feddropout_病理分布中模型p=0.2/1781962856_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-feddropout/0/log.txt" \
    --method "FedCMR (Ours)=/data/zyz/Feds/FedGKD-main/data/checkpoint/imagenet_tiny/federated_averaging_cnn/fed_our_病理分布中模型p=0.2/1780625846_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedour/0/log.txt" \
    --method "FedDF=/data/zyz/Feds/FedGKD-main/data/checkpoint/imagenet_tiny/federated_averaging_cnn/feddf_病理分布中模型p=0.2/1781963234_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-feddf/0/log.txt" \
    --method "FedMD=/data/zyz/Feds/FedGKD-main/data/checkpoint/imagenet_tiny/federated_averaging_cnn/fedmd_病理分布中模型p=0.2/1782007714_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedmd/0/log.txt" \
    --method "Fjord=/data/zyz/Feds/FedGKD-main/data/checkpoint/imagenet_tiny/federated_averaging_cnn/fjord_imagenet_tiny__病理分布pr0.2/1782007975_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fjord/0/log.txt" \
    --method "Fiarse=/data/zyz/Feds/FedGKD-main/data/checkpoint/imagenet_tiny/federated_averaging_cnn/fiarse_病理分布_p02_mild/1781954372_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fiarse/0/log.txt" \
    --method "FedDse=/data/zyz/Feds/FedGKD-main/data/checkpoint/imagenet_tiny/federated_averaging_cnn/feddse_tiny_病理分布p02_rates_085_09_095_10/1782180066_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-feddse/0/log.txt" \
    --title "Tiny-ImageNet (Pathological: 20/200)" \
    --max-round 200 \
    --smooth-window 7 \
    --y-min 0.0 \
    --y-max 0.55 \
    --output "./figures/异构tiny病理分布top1"

同构 cifar100_dirichlet_0.1
python plot_top1.py \
    --method "FedAvg=" \
    --method "FedAdam=" \
    --method "FedDistill=" \
    --method "FedCMR (Ours)=" \
    --method "FedDyn=" \
    --method "FedDensemble=" \
    --method "FedGen=" \
    --method "FEdGKD=" \
    --method "FedMoSWA=" \
    --method "FedSWA=" \
    --method "FedProx=" \
    --method "FedSAM=" \
    --method "Moon=" \
    --title "CIFAR-100 (Dirichlet: 0.1)" \
    --max-round 400 \
    --smooth-window 7 \
    --y-min 0.0 \
    --y-max 0.55 \
    --output "./figures/cifar100_dirichlet_0.1_smooth同构"

同构 cifar100_病理_0.1
python plot_top1.py \
    --method "FedAvg=/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/federated_averaging_cnn/avg_Niid0.1中模型病理分布p=0.2/1778197927_l2-0.0001_lr-0.01_n_comm_rounds-400_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedavg/0/log.txt" \
    --method "FedAdam=/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/federated_averaging_cnn/fedadam中模型niid0.1病理分布p=0.2/1778198144_l2-0.0001_lr-0.01_n_comm_rounds-400_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedadam/0/log.txt" \
    --method "FedDistill=/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/federated_averaging_cnn/feddistill中模型病理分布p=0.2/1778210775_l2-0.0001_lr-0.01_n_comm_rounds-400_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-feddistill/0/log.txt" \
    --method "FedCMR (Ours)=/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/federated_averaging_cnn/fed_our_noiid_中模型病理分布p=0.2——new/1778224997_l2-0.0_lr-0.01_n_comm_rounds-400_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedour/0/log.txt" \
    --method "FedDyn=/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/federated_averaging_cnn/feddyn中模型niid0.1病理分布p=0.2/1778240126_l2-0.0001_lr-0.01_n_comm_rounds-400_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-feddyn/0/log.txt" \
    --method "FedDensemble=/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/federated_averaging_cnn/fedensemble中模型niid0.1病理分布p=0.2/1778198211_l2-0.0001_lr-0.01_n_comm_rounds-400_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedensemble/0/log.txt" \
    --method "FedGen=/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/federated_averaging_cnn/fedgen中模型病理分布p=0.2/1778224839_l2-1e-05_lr-0.01_n_comm_rounds-100_local_n_epochs-20.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedgen/0/log.txt" \
    --method "FEdGKD=/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/federated_averaging_cnn/fedgkd中模型niid0.1病理分布p=0.2/1778198018_l2-0.0001_lr-0.01_n_comm_rounds-400_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedgkd/0/log.txt" \
    --method "FedMoSWA=/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/federated_averaging_cnn/fedmoswa_病理中模型p=0.2/1782030393_l2-0.0001_lr-0.01_n_comm_rounds-400_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedmoswa/0/log.txt" \
    --method "FedSWA=/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/federated_averaging_cnn/fedswa_病理中模型p=0.2/1781935275_l2-0.0001_lr-0.01_n_comm_rounds-400_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedswa/0/log.txt" \
    --method "FedProx=/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/federated_averaging_cnn/fedprox中模型niid0.1病理分布p=0.2/1778211644_l2-0.0001_lr-0.01_n_comm_rounds-400_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedprox/0/log.txt" \
    --method "FedSAM=/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/federated_averaging_cnn/fedsam_病理中模型p=0.2/1780968505_l2-0.0_lr-0.01_n_comm_rounds-400_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedsam/0/log.txt" \
    --method "Moon=/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/federated_averaging_cnn/moon中模型病理分布p=0.2/1778210907_l2-0.0001_lr-0.01_n_comm_rounds-400_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-moon/0/log.txt" \
    --title "CIFAR-100 (Pathological: 10/100)" \
    --max-round 400 \
    --smooth-window 7 \
    --y-min 0.0 \
    --y-max 0.55 \
    --output "./figures/cifar100_病理分布_smooth同构"
  
同构 tiny 病理分布
python plot_top1.py \
    --method "FedAvg=/data/zyz/Feds/FedGKD-main/data/checkpoint/imagenet_tiny/federated_averaging_cnn/fedavg中模型病理分布p=0.2/1780543887_l2-0.0001_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedavg/0/log.txt" \
    --method "FedAdam=/data/zyz/Feds/FedGKD-main/data/checkpoint/imagenet_tiny/federated_averaging_cnn/fedadam中模型病理分布p=0.2/1780547171_l2-0.0001_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedadam/0/log.txt" \
    --method "FedDistill=/data/zyz/Feds/FedGKD-main/data/checkpoint/imagenet_tiny/federated_averaging_cnn/feddistill_病理分布中模型p=0.2/1780570731_l2-0.0001_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-feddistill/0/log.txt" \
    --method "FedCMR (Ours)=/data/zyz/Feds/FedGKD-main/data/checkpoint/imagenet_tiny/federated_averaging_cnn/fed_our_病理分布中模型p=0.2/1780625846_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedour/0/log.txt" \
    --method "FedDyn=/data/zyz/Feds/FedGKD-main/data/checkpoint/imagenet_tiny/federated_averaging_cnn/feddyn中模型病理0.2/1780802855_l2-0.0001_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-feddyn/0/log.txt" \
    --method "FedDensemble=/data/zyz/Feds/FedGKD-main/data/checkpoint/imagenet_tiny/federated_averaging_cnn/fedensemble中模型病理分布p=0.2/1780547021_l2-0.0001_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedensemble/0/log.txt" \
    --method "FedGen=/data/zyz/Feds/FedGKD-main/data/checkpoint/imagenet_tiny/federated_averaging_cnn/fedgen——病理p=0.2/1783040337_l2-1e-05_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedgen/0/log.txt" \
    --method "FedGKD=/data/zyz/Feds/FedGKD-main/data/checkpoint/imagenet_tiny/federated_averaging_cnn/fedgkd中模型病理分布p=0.2/1780546948_l2-0.0001_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedgkd/0/log.txt" \
    --method "FedMoSWA=/data/zyz/Feds/FedGKD-main/data/checkpoint/imagenet_tiny/federated_averaging_cnn/fedmoswa_病理中模型p=0.2/1782455232_l2-0.0001_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedmoswa/0/log.txt" \
    --method "FedSWA=/data/zyz/Feds/FedGKD-main/data/checkpoint/imagenet_tiny/federated_averaging_cnn/fedswa_病理中模型p=0.2/1782346123_l2-0.0001_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedswa/0/log.txt" \
    --method "FedProx=/data/zyz/Feds/FedGKD-main/data/checkpoint/imagenet_tiny/federated_averaging_cnn/fedprox中模型病理分布p=0.2/1780570149_l2-0.0001_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedprox/0/log.txt" \
    --method "FedSAM=/data/zyz/Feds/FedGKD-main/data/checkpoint/imagenet_tiny/federated_averaging_cnn/fedsam_病理分布中模型p=0.2/1781573712_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedsam/0/log.txt" \
    --method "Moon=/data/zyz/Feds/FedGKD-main/data/checkpoint/imagenet_tiny/federated_averaging_cnn/moonn_病理分布中模型p=0.2/1780547280_l2-0.0001_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-moon/0/log.txt" \
    --title "Tiny-ImageNet (Pathological: 20/200)" \
    --max-round 400 \
    --smooth-window 7 \
    --y-min 0.0 \
    --y-max 0.35 \
    --output "./figures/Tiny-ImageNet_病理分布_smooth同构"




python plot_top1.py \
    --method "FedAvg=" \
    --method "FedAdam=" \
    --method "FedDistill=" \
    --method "FedCMR (Ours)=" \
    --method "FedDyn=" \
    --method "FedDensemble=" \
    --method "FedGen=" \
    --method "FEdGKD=" \
    --method "FedMoSWA=" \
    --method "FedSWA=" \
    --method "FedProx=" \
    --method "FedSAM=" \
    --method "Moon=" \
    --title "CIFAR-100 (Pathological: 10/100)" \
    --max-round 400 \
    --smooth-window 7 \
    --y-min 0.0 \
    --y-max 0.55 \
    --output "./figures/cifar100_病理分布_smooth同构"


"""

