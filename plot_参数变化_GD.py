# -*- coding: utf-8 -*-
"""
从多个联邦学习日志中提取每一通信轮次的 validation top1，
并绘制不同参数设置下的准确率变化曲线。

修复内容：
1. 修复 print_experiment_summary() 中 param_value 未定义的问题。
2. 将 lambda_value 统一重命名为 param_value。
3. 日志路径是 CIFAR10，因此同步修正图标题和输出文件名。
4. 默认使用中心移动平均，避免 EMA_ALPHA 太小时产生明显滞后。
5. 只提取 The validation performance 行中的 top1。
6. 不补齐不存在的通信轮次。
"""

import re
from pathlib import Path

import matplotlib
import numpy as np

# 服务器没有图形界面时保持 True
HEADLESS = True
if HEADLESS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


# ============================================================
# 1. 实验配置
# ============================================================
EXPERIMENTS = [
    {
        "value": 0,
        "path": (
            "/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar10/"
            "federated_averaging_cnn/GD超惨分析-0/"
            "1783756955_l2-0.0_lr-0.1_n_comm_rounds-200_"
            "local_n_epochs-10.0_batchsize-64_n_clients_20_"
            "n_participated-4_optim-sgd_method-fedour/0/log.txt"
        ),
        "color": "#B7B7B7",
        "marker": "s",
    },
    {
        "value": 1,
        "path": (
            "/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar10/"
            "federated_averaging_cnn/GD超惨分析-1/"
            "1783757669_l2-0.0_lr-0.1_n_comm_rounds-200_"
            "local_n_epochs-10.0_batchsize-64_n_clients_20_"
            "n_participated-4_optim-sgd_method-fedour/0/log.txt"
        ),
        "color": "#3B75AF",
        "marker": "o",
    },
    {
        "value": 3,
        "path": (
            "/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar10/"
            "federated_averaging_cnn/GD超惨分析-3/"
            "1783758083_l2-0.0_lr-0.1_n_comm_rounds-200_"
            "local_n_epochs-10.0_batchsize-64_n_clients_20_"
            "n_participated-4_optim-sgd_method-fedour/0/log.txt"
        ),
        "color": "#E68636",
        "marker": "^",
    },
    {
        "value": 5,
        "path": (
            "/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar10/"
            "federated_averaging_cnn/GD超惨分析-5/"
            "1783758509_l2-0.0_lr-0.1_n_comm_rounds-200_"
            "local_n_epochs-10.0_batchsize-64_n_clients_20_"
            "n_participated-4_optim-sgd_method-fedour/0/log.txt"
        ),
        "color": "#4EA43B",
        "marker": "D",
    },
    {
        "value": 9,
        "path": (
            "/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar10/"
            "federated_averaging_cnn/GD超惨分析-9/"
            "1783763353_l2-0.0_lr-0.1_n_comm_rounds-200_"
            "local_n_epochs-10.0_batchsize-64_n_clients_20_"
            "n_participated-4_optim-sgd_method-fedour/0/log.txt"
        ),
        "color": "#9567BD",
        "marker": "P",
    },
]


# ============================================================
# 2. 参数名称与绘图配置
# ============================================================
# 根据参数真实含义修改下面两项即可。
# 例如：
# PARAM_SYMBOL = r"\beta_{\mathrm{align}}"
# PARAM_NAME = "alignment weight"
PARAM_SYMBOL = r"\beta"
PARAM_NAME = "alignment weight"

DATASET_NAME = "CIFAR10"
NON_IID_ALPHA = 0.1

MAX_ROUND = 200

# 平滑方式："moving_average"、"ema" 或 "none"
# 推荐中心移动平均，它不会像较小 alpha 的 EMA 那样产生明显滞后。
SMOOTH_METHOD = "ema"
MA_WINDOW = 9
EMA_ALPHA = 0.03

SHOW_RAW_CURVE = False
RAW_CURVE_ALPHA = 0.15

MARK_EVERY = 10

X_LIM = (40, 200)
X_TICK_INTERVAL = 20

# y 轴模式：
# "late_stage_auto"：根据后期曲线自动缩放，突出参数差异
# "all_auto"：根据所有轮次自动缩放
# "manual"：使用 Y_LIM
Y_AXIS_MODE = "late_stage_auto"
FOCUS_START_ROUND_FOR_Y = 50
Y_LIM = (0, 0.7)

MIN_Y_SPAN = 0.04
Y_MARGIN_RATIO = 0.10
Y_TICK_INTERVAL = 0.02

TITLE = (
    rf"$\mathbf{{{DATASET_NAME}}}$ "
    rf"$(\mathit{{Practical}}: \alpha={NON_IID_ALPHA:g})$"
)

SAVE_DIR = Path("/data/zyz/Feds/FedGKD-main/参数消融研究")
SAVE_PDF_PATH = SAVE_DIR / "cifar10_alpha0.1_GD_ablation_smooth.pdf"
SAVE_PNG_PATH = SAVE_DIR / "cifar10_alpha0.1_GD_ablation_smooth.png"


# ============================================================
# 3. 日志正则表达式
# ============================================================
# 只匹配每轮正式开始的行，避免误读其他 comm_round 字段。
ROUND_PATTERN = re.compile(
    r"Master\s+starting\s+one\s+round\s+of\s+federated\s+learning:"
    r".*?\(\s*comm_round\s*=\s*(\d+)\s*\)",
    re.IGNORECASE,
)

FLOAT_PATTERN = (
    r"[+-]?"
    r"(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?"
)

# 只匹配 validation performance 中的 top1。
VALIDATION_TOP1_PATTERN = re.compile(
    rf"The\s+validation\s+performance\s*="
    rf".*?['\"]top1['\"]\s*:\s*({FLOAT_PATTERN})",
    re.IGNORECASE,
)


# ============================================================
# 4. 配置检查
# ============================================================
def validate_config():
    if not EXPERIMENTS:
        raise ValueError("EXPERIMENTS 不能为空。")

    values = [exp["value"] for exp in EXPERIMENTS]
    if len(values) != len(set(values)):
        raise ValueError(f"EXPERIMENTS 中存在重复参数值：{values}")

    if MAX_ROUND <= 0:
        raise ValueError("MAX_ROUND 必须大于 0。")

    if X_TICK_INTERVAL <= 0:
        raise ValueError("X_TICK_INTERVAL 必须大于 0。")

    if SMOOTH_METHOD not in {"moving_average", "ema", "none"}:
        raise ValueError(
            "SMOOTH_METHOD 只能是 moving_average、ema 或 none。"
        )

    if not 0.0 < EMA_ALPHA <= 1.0:
        raise ValueError("EMA_ALPHA 必须位于 (0, 1]。")

    if MA_WINDOW < 1:
        raise ValueError("MA_WINDOW 必须至少为 1。")

    if Y_AXIS_MODE not in {"late_stage_auto", "all_auto", "manual"}:
        raise ValueError(
            "Y_AXIS_MODE 只能是 late_stage_auto、all_auto 或 manual。"
        )

    if Y_AXIS_MODE == "manual" and Y_LIM[0] >= Y_LIM[1]:
        raise ValueError("Y_LIM 下界必须小于上界。")


# ============================================================
# 5. 提取 validation top1
# ============================================================
def extract_validation_top1(file_path, max_round=None):
    """
    从日志中提取每一通信轮次 validation performance 的 top1。

    日志中的 top1 按百分数处理：
        43.27 -> 0.4327
    """
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"日志文件不存在：{file_path}")

    round_to_top1 = {}
    current_round = None
    duplicate_rounds = set()

    with file_path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            round_match = ROUND_PATTERN.search(line)
            if round_match:
                current_round = int(round_match.group(1))

                if max_round is not None and current_round > max_round:
                    current_round = None

                continue

            if current_round is None:
                continue

            top1_match = VALIDATION_TOP1_PATTERN.search(line)
            if top1_match is None:
                continue

            # 该日志格式中的 top1 是百分数。
            top1 = float(top1_match.group(1)) / 100.0

            if not 0.0 <= top1 <= 1.0:
                print(
                    f"[警告] 跳过异常数据："
                    f"round={current_round}, top1={top1}"
                )
                current_round = None
                continue

            if current_round in round_to_top1:
                duplicate_rounds.add(current_round)

            # 若恢复训练导致同一轮重复出现，保留最后一次记录。
            round_to_top1[current_round] = top1
            current_round = None

    if not round_to_top1:
        raise RuntimeError(
            f"没有从日志中解析到 validation top1：{file_path}\n"
            "请确认日志中存在类似：\n"
            "The validation performance = {'top1': 43.27, ...}"
        )

    rounds = np.asarray(sorted(round_to_top1), dtype=int)
    accuracies = np.asarray(
        [round_to_top1[round_id] for round_id in rounds],
        dtype=float,
    )

    if duplicate_rounds:
        print(
            f"[提示] 检测到重复轮次 {sorted(duplicate_rounds)}，"
            "已保留每轮最后一次验证结果。"
        )

    return rounds, accuracies


# ============================================================
# 6. 平滑函数
# ============================================================
def exponential_moving_average(values, alpha=0.20):
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return values.copy()

    smoothed = np.empty_like(values)
    smoothed[0] = values[0]

    for index in range(1, len(values)):
        smoothed[index] = (
            alpha * values[index]
            + (1.0 - alpha) * smoothed[index - 1]
        )

    return smoothed


def moving_average(values, window=9):
    """
    中心移动平均。

    使用边缘填充，返回长度与输入完全一致。
    """
    values = np.asarray(values, dtype=float)

    if values.size < 3 or window <= 1:
        return values.copy()

    window = min(int(window), len(values))

    # 使用奇数窗口以保证曲线居中。
    if window % 2 == 0:
        window -= 1

    if window < 3:
        return values.copy()

    padding = window // 2
    padded = np.pad(values, (padding, padding), mode="edge")
    kernel = np.ones(window, dtype=float) / window

    return np.convolve(padded, kernel, mode="valid")


def smooth_curve(values):
    if SMOOTH_METHOD == "none":
        return np.asarray(values, dtype=float).copy()

    if SMOOTH_METHOD == "ema":
        return exponential_moving_average(values, alpha=EMA_ALPHA)

    if SMOOTH_METHOD == "moving_average":
        return moving_average(values, window=MA_WINDOW)

    # 正常情况下 validate_config() 会提前拦截。
    raise ValueError(f"不支持的平滑方法：{SMOOTH_METHOD}")


# ============================================================
# 7. 自动计算 y 轴范围
# ============================================================
def calculate_ylim(experiments):
    if Y_AXIS_MODE == "manual":
        return Y_LIM

    values_for_ylim = []

    for experiment in experiments:
        rounds = experiment["rounds"]
        curve = experiment["smoothed_accuracies"]

        if Y_AXIS_MODE == "late_stage_auto":
            mask = rounds >= FOCUS_START_ROUND_FOR_Y
            selected = curve[mask] if np.any(mask) else curve
        else:
            selected = curve

        if selected.size:
            values_for_ylim.append(selected)

    if not values_for_ylim:
        raise RuntimeError("无法根据空曲线计算 y 轴范围。")

    values = np.concatenate(values_for_ylim)

    y_min = float(np.min(values))
    y_max = float(np.max(values))

    actual_span = y_max - y_min
    display_span = max(actual_span, MIN_Y_SPAN)

    center = (y_min + y_max) / 2.0
    lower = center - display_span / 2.0
    upper = center + display_span / 2.0

    margin = display_span * Y_MARGIN_RATIO
    lower -= margin
    upper += margin

    # 调整到 0.01 边界。
    lower = np.floor(lower * 100.0) / 100.0
    upper = np.ceil(upper * 100.0) / 100.0

    lower = max(0.0, lower)
    upper = min(1.0, upper)

    if lower >= upper:
        raise RuntimeError(
            f"自动计算得到无效 y 轴范围：({lower}, {upper})"
        )

    return lower, upper


# ============================================================
# 8. 控制台摘要
# ============================================================
def print_experiment_summary(param_value, rounds, accuracies):
    """
    打印一个参数设置对应的实验摘要。

    这里统一使用 param_value，修复原代码中引用未定义
    param_value 导致的 NameError。
    """
    best_index = int(np.argmax(accuracies))
    best_round = int(rounds[best_index])
    best_accuracy = float(accuracies[best_index])

    print("=" * 72)
    print(f"{PARAM_NAME:<19}: {param_value:g}")
    print(f"提取数据点数量     : {len(rounds)}")
    print(f"通信轮次范围       : {rounds.min()} ~ {rounds.max()}")
    print(
        f"第一轮记录         : round={rounds[0]}, "
        f"top1={accuracies[0] * 100:.2f}%"
    )
    print(
        f"最后一轮记录       : round={rounds[-1]}, "
        f"top1={accuracies[-1] * 100:.2f}%"
    )
    print(
        f"最高准确率         : round={best_round}, "
        f"top1={best_accuracy * 100:.2f}%"
    )
    print("前 5 个数据点：")

    for round_id, accuracy in zip(rounds[:5], accuracies[:5]):
        print(
            f"  round={round_id:3d}, "
            f"top1={accuracy:.4f} ({accuracy * 100:.2f}%)"
        )


# ============================================================
# 9. 读取全部实验
# ============================================================
def load_all_experiments():
    loaded = []

    for experiment in EXPERIMENTS:
        param_value = experiment["value"]
        file_path = experiment["path"]

        print(f"\n正在读取 {PARAM_NAME}={param_value:g}")
        print(f"日志路径：{file_path}")

        rounds, raw_accuracies = extract_validation_top1(
            file_path=file_path,
            max_round=MAX_ROUND,
        )

        smoothed_accuracies = smooth_curve(raw_accuracies)

        print_experiment_summary(
            param_value=param_value,
            rounds=rounds,
            accuracies=raw_accuracies,
        )

        loaded.append(
            {
                **experiment,
                "rounds": rounds,
                "raw_accuracies": raw_accuracies,
                "smoothed_accuracies": smoothed_accuracies,
            }
        )

    if not loaded:
        raise RuntimeError("没有成功读取任何实验。")

    return loaded


# ============================================================
# 10. 绘图
# ============================================================
def plot_experiments(experiments):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.labelsize": 15,
            "axes.titlesize": 20,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axis = plt.subplots(figsize=(7.4, 5.2), dpi=180)

    for experiment in experiments:
        rounds = experiment["rounds"]
        raw_curve = experiment["raw_accuracies"]
        smooth_curve_values = experiment["smoothed_accuracies"]

        color = experiment["color"]
        marker = experiment["marker"]
        param_value = experiment["value"]

        if SHOW_RAW_CURVE:
            axis.plot(
                rounds,
                raw_curve,
                color=color,
                linewidth=0.8,
                alpha=RAW_CURVE_ALPHA,
                zorder=1,
            )

        axis.plot(
            rounds,
            smooth_curve_values,
            label=rf"${PARAM_SYMBOL}={param_value:g}$",
            color=color,
            marker=marker,
            markevery=MARK_EVERY,
            linewidth=2.0,
            markersize=5.8,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.6,
            solid_capstyle="round",
            zorder=3,
        )

    axis.set_title(TITLE, pad=10)
    axis.set_xlabel("Communication Rounds")
    axis.set_ylabel("Test Accuracy")

    axis.set_xlim(*X_LIM)
    axis.set_xticks(
        np.arange(
            X_LIM[0],
            X_LIM[1] + X_TICK_INTERVAL,
            X_TICK_INTERVAL,
        )
    )

    ylim = calculate_ylim(experiments)
    axis.set_ylim(*ylim)
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    if Y_TICK_INTERVAL is not None and Y_TICK_INTERVAL > 0:
        axis.yaxis.set_major_locator(MultipleLocator(Y_TICK_INTERVAL))

    axis.grid(
        True,
        which="major",
        axis="both",
        linestyle=(0, (2, 3)),
        linewidth=0.75,
        alpha=0.60,
    )

    for spine in axis.spines.values():
        spine.set_linewidth(1.4)
        spine.set_color("black")

    axis.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=4.5,
        width=1.0,
    )

    legend = axis.legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor="black",
        borderpad=0.40,
        labelspacing=0.28,
        handlelength=1.65,
        handletextpad=0.45,
    )
    legend.get_frame().set_linewidth(0.9)

    figure.tight_layout()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(SAVE_PDF_PATH, format="pdf", bbox_inches="tight")
    figure.savefig(SAVE_PNG_PATH, format="png", dpi=600, bbox_inches="tight")

    actual_ylim = axis.get_ylim()

    if not HEADLESS:
        plt.show()

    plt.close(figure)

    print("\n" + "=" * 72)
    print(f"当前平滑方法：{SMOOTH_METHOD}")

    if SMOOTH_METHOD == "moving_average":
        print(f"MA_WINDOW = {MA_WINDOW}")
    elif SMOOTH_METHOD == "ema":
        print(f"EMA_ALPHA = {EMA_ALPHA}")

    print(f"当前 Y 轴模式：{Y_AXIS_MODE}")

    if Y_AXIS_MODE == "late_stage_auto":
        print(
            "后期聚焦起始轮次："
            f"round >= {FOCUS_START_ROUND_FOR_Y}"
        )

    print(f"实际 y 轴范围：{actual_ylim}")
    print(f"PDF 已保存：{SAVE_PDF_PATH}")
    print(f"PNG 已保存：{SAVE_PNG_PATH}")


# ============================================================
# 11. 主函数
# ============================================================
def main():
    validate_config()
    experiments = load_all_experiments()
    plot_experiments(experiments)


if __name__ == "__main__":
    main()