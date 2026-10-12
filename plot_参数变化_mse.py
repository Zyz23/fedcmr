# -*- coding: utf-8 -*-
"""
从多个联邦学习日志中提取每一通信轮次的 validation top1，
并绘制不同 lambda 参数的准确率变化曲线。

日志格式示例：

Master starting one round of federated learning: (comm_round=1).
...
The validation performance = {
    'loss': 4.5842,
    'top1': 2.91,
    'top5': 10.81,
    'loss2': 0.0
}.

注意：
1. 只读取 "The validation performance" 行中的 top1。
2. 不读取 "Best performance of top1"。
3. 日志中的 top1 是百分数，例如 52.31，绘图时转换为 0.5231。
4. 不补齐缺失轮次，只连接日志中真实存在的数据点。
"""


# ============================================================
# 1. 实验配置
# ============================================================
# 每个实验的路径、颜色和标记绑定在一起，避免样式错位



# -*- coding: utf-8 -*-
"""
从多个联邦学习日志中提取每一通信轮次的 validation top1，
绘制不同 lambda 参数的准确率变化曲线，并进行平滑和纵轴优化。

优化点：
1. 只提取 The validation performance 中的 top1
2. 支持平滑（EMA / Moving Average）
3. y 轴不从 0 开始，自动聚焦后期精度差异
4. 图整体更接近论文风格

作者建议：
- 平滑方法：EMA
- y轴模式：late_stage_auto
"""

import re
from pathlib import Path

import numpy as np
import matplotlib

# 如果服务器没有图形界面，设为 True
HEADLESS = True
if HEADLESS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


# ============================================================
# 1. 实验配置
# ============================================================
EXPERIMENTS = [
    {
        "lambda": 0,
        "path": (
            "/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/"
            "federated_averaging_cnn/"
            "fed_our_noiid_0.1p=0.2参数mse_0v4/"
            "1783075932_l2-0.0_lr-0.01_n_comm_rounds-200_"
            "local_n_epochs-10.0_batchsize-64_n_clients_20_"
            "n_participated-4_optim-sgd_method-fedour/0/log.txt"
        ),
        "color": "#B7B7B7",
        "marker": "s",
    },
    {
        "lambda": 0.5,
        "path": (
            "/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/"
            "federated_averaging_cnn/"
            "fed_our_noiid_0.1p=0.2参数mse_0.5v3/"            
            "1783074109_l2-0.0_lr-0.01_n_comm_rounds-200_"
            "local_n_epochs-10.0_batchsize-64_n_clients_20_"
            "n_participated-4_optim-sgd_method-fedour/0/log.txt"
        ),
        "color": "#3B75AF",
        "marker": "o",
    },
    {
        "lambda": 1,
        "path": (
            "/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/"
            "federated_averaging_cnn/"
            "fed_our_noiid_0.1p=0.2参数mse_1v4/"
            "1783076056_l2-0.0_lr-0.01_n_comm_rounds-200_"
            "local_n_epochs-10.0_batchsize-64_n_clients_20_"
            "n_participated-4_optim-sgd_method-fedour/0/log.txt"
        ),
        "color": "#E68636",
        "marker": "^",
    },
    {
        "lambda": 5,
        "path": (
            "/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/"
            "federated_averaging_cnn/"
            "fed_our_noiid_0.1p=0.2参数mse_5v3/"
            "1783073846_l2-0.0_lr-0.01_n_comm_rounds-200_"
            "local_n_epochs-10.0_batchsize-64_n_clients_20_"
            "n_participated-4_optim-sgd_method-fedour/0/log.txt"
        ),
        "color": "#4EA43B",
        "marker": "D",
    },
    {
        "lambda": 10,
        "path": (
            "/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/"
            "federated_averaging_cnn/"
            "fed_our_noiid_0.1p=0.2参数mse_10v2/"
            "1783070670_l2-0.0_lr-0.01_n_comm_rounds-200_"
            "local_n_epochs-10.0_batchsize-64_n_clients_20_"
            "n_participated-4_optim-sgd_method-fedour/0/log.txt"
        ),
        "color": "#9567BD",
        "marker": "P",
    },
]


# ============================================================
# 2. 全局绘图配置
# ============================================================
MAX_ROUND = 200

# ---------- 平滑配置 ----------
# 可选: "ema", "moving_average", "none"
SMOOTH_METHOD = "ema"

# EMA 平滑参数：越小越平滑，推荐 0.20 ~ 0.35
EMA_ALPHA = 0.05

# Moving Average 窗口：奇数更合适
MA_WINDOW = 7

# ---------- 是否显示原始曲线 ----------
SHOW_RAW_CURVE = False

# ---------- marker 显示频率 ----------
MARK_EVERY = 10

# ---------- x 轴 ----------
X_LIM = (40, 200)
X_TICK_INTERVAL = 20

# ---------- y 轴模式 ----------
# "all_auto"         : 根据全部数据自动设定
# "late_stage_auto"  : 只根据后期轮次自动设定，突出差异（推荐）
# "manual"           : 手动指定 Y_LIM
Y_AXIS_MODE = "late_stage_auto"

# 从第几轮开始作为“后期”来决定 y 轴范围
FOCUS_START_ROUND_FOR_Y = 70

# 手动 y 轴范围（仅 Y_AXIS_MODE="manual" 时生效）
Y_LIM = (0.40, 0.48)

# 自动 y 轴的最小跨度，防止范围太窄
MIN_Y_SPAN = 0.04

# 自动 y 轴上下边界留白比例
Y_MARGIN_RATIO = 0.12

# ---------- 标题 ----------
TITLE = r"$\mathbf{CIFAR100}$ $(\mathit{Practical}: \alpha=0.1)$"

# ---------- 输出 ----------
SAVE_DIR = Path("/data/zyz/Feds/FedGKD-main/参数消融研究")
SAVE_PDF_PATH = SAVE_DIR / "cifar100_alpha0.1_lambda_ablation_smooth.pdf"
SAVE_PNG_PATH = SAVE_DIR / "cifar100_alpha0.1_lambda_ablation_smooth.png"


# ============================================================
# 3. 日志正则表达式
# ============================================================
ROUND_PATTERN = re.compile(
    r"\bcomm_round\s*=\s*(\d+)\b",
    re.IGNORECASE,
)

FLOAT_PATTERN = (
    r"[+-]?"
    r"(?:"
    r"\d+(?:\.\d*)?"
    r"|"
    r"\.\d+"
    r")"
    r"(?:[eE][+-]?\d+)?"
)

VALIDATION_TOP1_PATTERN = re.compile(
    rf"The\s+validation\s+performance\s*="
    rf".*?"
    rf"['\"]top1['\"]\s*:\s*"
    rf"({FLOAT_PATTERN})",
    re.IGNORECASE,
)


# ============================================================
# 4. 提取 validation top1
# ============================================================
def extract_validation_top1(file_path, max_round=None):
    """
    只提取每轮 validation performance 中的 top1
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"日志文件不存在：{file_path}")

    round_to_top1 = {}
    current_round = None
    duplicate_rounds = set()

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            round_match = ROUND_PATTERN.search(line)
            if round_match:
                current_round = int(round_match.group(1))
                continue

            if current_round is None:
                continue

            if max_round is not None and current_round > max_round:
                continue

            top1_match = VALIDATION_TOP1_PATTERN.search(line)
            if top1_match is None:
                continue

            top1 = float(top1_match.group(1))

            # 日志里一般是百分数，如 43.27
            if top1 > 1.0:
                top1 /= 100.0

            if not (0.0 <= top1 <= 1.0):
                current_round = None
                continue

            if current_round in round_to_top1:
                duplicate_rounds.add(current_round)

            round_to_top1[current_round] = top1

            # 避免同一轮后续 Best performance 的干扰
            current_round = None

    if len(round_to_top1) == 0:
        raise RuntimeError(f"没有解析到 validation top1：{file_path}")

    rounds = np.asarray(sorted(round_to_top1.keys()), dtype=int)
    accuracies = np.asarray([round_to_top1[r] for r in rounds], dtype=float)

    if duplicate_rounds:
        print(
            f"[提示] 检测到重复轮次 {sorted(duplicate_rounds)}，"
            f"已保留最后一次验证结果。"
        )

    return rounds, accuracies


# ============================================================
# 5. 平滑函数
# ============================================================
def exponential_moving_average(values, alpha=0.22):
    """
    EMA 平滑：适合训练曲线，视觉上更顺
    """
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values.copy()

    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]

    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]

    return smoothed


def moving_average(values, window=7):
    """
    中心滑动平均
    """
    values = np.asarray(values, dtype=float)

    if window <= 1 or len(values) < 3:
        return values.copy()

    window = min(window, len(values))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return values.copy()

    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window

    return np.convolve(padded, kernel, mode="valid")


def smooth_curve(values, method="ema", ema_alpha=0.22, ma_window=7):
    if method == "none":
        return np.asarray(values, dtype=float).copy()
    elif method == "ema":
        return exponential_moving_average(values, alpha=ema_alpha)
    elif method == "moving_average":
        return moving_average(values, window=ma_window)
    else:
        raise ValueError(f"不支持的平滑方法: {method}")


# ============================================================
# 6. 自动计算 y 轴范围
# ============================================================
def calculate_ylim(experiments, mode="late_stage_auto"):
    """
    根据配置计算 y 轴范围
    """
    if mode == "manual":
        return Y_LIM

    values_for_ylim = []

    for exp in experiments:
        rounds = exp["rounds"]
        ys = exp["smoothed_accuracies"]

        if mode == "all_auto":
            values_for_ylim.append(ys)
        elif mode == "late_stage_auto":
            mask = rounds >= FOCUS_START_ROUND_FOR_Y
            if np.any(mask):
                values_for_ylim.append(ys[mask])
            else:
                values_for_ylim.append(ys)
        else:
            raise ValueError(f"未知的 Y_AXIS_MODE: {mode}")

    values = np.concatenate(values_for_ylim)

    y_min = float(np.min(values))
    y_max = float(np.max(values))

    span = y_max - y_min
    span = max(span, MIN_Y_SPAN)

    lower = y_min - span * Y_MARGIN_RATIO
    upper = y_max + span * Y_MARGIN_RATIO

    lower = np.floor(lower * 100.0) / 100.0
    upper = np.ceil(upper * 100.0) / 100.0

    lower = max(0.0, lower)
    upper = min(1.0, upper)

    return lower, upper


# ============================================================
# 7. 控制台打印
# ============================================================
def print_experiment_summary(lambda_value, rounds, accuracies):
    best_idx = int(np.argmax(accuracies))
    best_round = int(rounds[best_idx])
    best_acc = float(accuracies[best_idx])

    print("=" * 72)
    print(f"lambda             : {lambda_value:g}")
    print(f"提取数据点数量     : {len(rounds)}")
    print(f"通信轮次范围       : {rounds.min()} ~ {rounds.max()}")
    print(f"第一轮记录         : round={rounds[0]}, top1={accuracies[0] * 100:.2f}%")
    print(f"最后一轮记录       : round={rounds[-1]}, top1={accuracies[-1] * 100:.2f}%")
    print(f"最高准确率         : round={best_round}, top1={best_acc * 100:.2f}%")
    print("前 5 个数据点：")
    for r, a in zip(rounds[:5], accuracies[:5]):
        print(f"  round={r:3d}, top1={a:.4f} ({a * 100:.2f}%)")


# ============================================================
# 8. 读取全部实验
# ============================================================
def load_all_experiments():
    loaded = []

    for exp in EXPERIMENTS:
        lambda_value = exp["lambda"]
        file_path = exp["path"]

        print(f"\n正在读取 lambda={lambda_value:g}")
        print(f"日志路径：{file_path}")

        rounds, raw_accuracies = extract_validation_top1(
            file_path=file_path,
            max_round=MAX_ROUND,
        )

        smoothed_accuracies = smooth_curve(
            raw_accuracies,
            method=SMOOTH_METHOD,
            ema_alpha=EMA_ALPHA,
            ma_window=MA_WINDOW,
        )

        print_experiment_summary(
            lambda_value=lambda_value,
            rounds=rounds,
            accuracies=raw_accuracies,
        )

        loaded.append(
            {
                **exp,
                "rounds": rounds,
                "raw_accuracies": raw_accuracies,
                "smoothed_accuracies": smoothed_accuracies,
            }
        )

    return loaded


# ============================================================
# 9. 绘图
# ============================================================
def plot_experiments(experiments):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.labelsize": 16,
            "axes.titlesize": 22,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.linewidth": 1.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(8.4, 5.8), dpi=180)

    for exp in experiments:
        rounds = exp["rounds"]
        raw_y = exp["raw_accuracies"]
        y = exp["smoothed_accuracies"]

        color = exp["color"]
        marker = exp["marker"]
        lambda_value = exp["lambda"]

        if SHOW_RAW_CURVE:
            ax.plot(
                rounds,
                raw_y,
                color=color,
                linewidth=0.8,
                alpha=0.18,
                zorder=1,
            )

        ax.plot(
            rounds,
            y,
            label=rf"$\lambda={lambda_value:g}$",
            color=color,
            marker=marker,
            markevery=MARK_EVERY,
            linewidth=2.2,
            markersize=6.5,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.7,
            solid_capstyle="round",
            zorder=3,
        )

    # 标题和坐标轴
    ax.set_title(TITLE, pad=12)
    ax.set_xlabel("Communication Rounds")
    ax.set_ylabel("Test Accuracy")

    # x 轴
    ax.set_xlim(*X_LIM)
    ax.set_xticks(np.arange(X_LIM[0], X_LIM[1] + 1, X_TICK_INTERVAL))

    # y 轴
    ylim = calculate_ylim(experiments, mode=Y_AXIS_MODE)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    # 网格
    ax.grid(
        True,
        which="major",
        axis="both",
        linestyle=(0, (2, 3)),
        linewidth=0.8,
        alpha=0.6,
    )

    # 坐标轴边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
        spine.set_color("black")

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=5,
        width=1.1,
    )

    # 图例
    legend = ax.legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor="black",
        borderpad=0.45,
        labelspacing=0.3,
        handlelength=1.7,
        handletextpad=0.5,
    )
    legend.get_frame().set_linewidth(1.0)

    fig.tight_layout()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    fig.savefig(SAVE_PDF_PATH, format="pdf", bbox_inches="tight")
    fig.savefig(SAVE_PNG_PATH, format="png", dpi=600, bbox_inches="tight")

    if not HEADLESS:
        plt.show()

    plt.close(fig)

    print("\n" + "=" * 72)
    print(f"当前平滑方法：{SMOOTH_METHOD}")
    if SMOOTH_METHOD == "ema":
        print(f"EMA_ALPHA = {EMA_ALPHA}")
    elif SMOOTH_METHOD == "moving_average":
        print(f"MA_WINDOW = {MA_WINDOW}")

    print(f"当前 Y 轴模式：{Y_AXIS_MODE}")
    if Y_AXIS_MODE == "late_stage_auto":
        print(f"后期聚焦起始轮次：round >= {FOCUS_START_ROUND_FOR_Y}")

    print(f"实际 y 轴范围：{ax.get_ylim()}")
    print(f"PDF 已保存：{SAVE_PDF_PATH}")
    print(f"PNG 已保存：{SAVE_PNG_PATH}")


# ============================================================
# 10. 主函数
# ============================================================
def main():
    experiments = load_all_experiments()
    plot_experiments(experiments)


if __name__ == "__main__":
    main()