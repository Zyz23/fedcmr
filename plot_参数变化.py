# -*- coding: utf-8 -*-
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

# -------------------------- 配置区（修改这里）--------------------------
# 每个lambda对应的日志路径，按图例顺序：λ=0,0.1,1,2,5,10,20
log_paths = {
    r"$\lambda=0$": "/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/federated_averaging_cnn/fed_our_noiid_0.1p=0.2参数mse_0v4/1783075932_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedour/0/log.txt",
    r"$\lambda=0.5$": "/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/federated_averaging_cnn/fed_our_noiid_0.1p=0.2参数mse_0.5v4/1783148567_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedour/0/log.txt",
    r"$\lambda=1$": "/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/federated_averaging_cnn/fed_our_noiid_0.1p=0.2参数mse_1v4/1783076056_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedour/0/log.txt",
    # r"$\lambda=2$": "/path/to/lambda_2/log.txt",
    r"$\lambda=5$": "/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/federated_averaging_cnn/fed_our_noiid_0.1p=0.2参数mse_5v3/1783073846_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedour/0/log.txt",
    r"$\lambda=10$": "/data/zyz/Feds/FedGKD-main/data/checkpoint/cifar100/federated_averaging_cnn/fed_our_noiid_0.1p=0.2参数mse_10v2/1783070670_l2-0.0_lr-0.01_n_comm_rounds-200_local_n_epochs-10.0_batchsize-64_n_clients_20_n_participated-4_optim-sgd_method-fedour/0/log.txt",
    # r"$\lambda=20$": "/path/to/lambda_20/log.txt",
}
max_round = 100        # 横轴最大通信轮次（图中是100）
smooth_window = 3      # 平滑窗口大小，消除抖动
fig_save_path = "./cifar10_practical_alpha0.1_lambda_ablation.pdf"
# 配色+标记 严格匹配原图
style_cfg = [
    {"color":"#cccccc", "marker":"s"},  # λ=0 灰色方块
    {"color":"#3377bb", "marker":"o"},  # λ=0.1 蓝色圆
    {"color":"#ee8833", "marker":"^"},  # λ=1 橙色三角
    {"color":"#885533", "marker":"D"},  # λ=2 棕色菱形
    {"color":"#55aa33", "marker":"D"},  # λ=5 绿色菱形
    {"color":"#9966cc", "marker":"o"},  # λ=10 紫色圆
    {"color":"#dd2222", "marker":"*"},  # λ=20 红色星号
]
# -----------------------------------------------------------------------

def extract_top1_from_log(file_path, max_comm=200):
    """从日志提取 通信轮次 - Top1 Acc"""
    round_pattern = re.compile(r"comm_round=(\d+)")
    top1_pattern = re.compile(r"'top1': ([0-9.]+)")
    rounds = []
    top1_accs = []
    current_round = None

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # 匹配轮次
            r_match = round_pattern.search(line)
            if r_match:
                current_round = int(r_match.group(1))
                if current_round > max_comm:
                    break
            # 匹配验证精度
            t_match = top1_pattern.search(line)
            if t_match and current_round is not None:
                acc = float(t_match.group(1)) / 100.0  # 转0~1小数
                rounds.append(current_round)
                top1_accs.append(acc)
    # 补齐到完整轮次，缺失值就近填充
    full_rounds = np.arange(1, max_round+1)
    full_acc = np.zeros(max_round)
    ptr = 0
    for i, r in enumerate(full_rounds):
        if ptr < len(rounds) and rounds[ptr] == r:
            full_acc[i] = top1_accs[ptr]
            ptr += 1
        else:
            full_acc[i] = full_acc[i-1] if i>0 else top1_accs[0]
    # 平滑
    smooth_acc = uniform_filter1d(full_acc, size=smooth_window, mode="nearest")
    return full_rounds, smooth_acc

# 1. 读取所有lambda数据
data_dict = {}
labels_list = list(log_paths.keys())
for label, path in log_paths.items():
    x, y = extract_top1_from_log(path, max_round)
    data_dict[label] = (x, y)

# 2. 绘图
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(12,7), dpi=150)

for idx, label in enumerate(labels_list):
    x, y = data_dict[label]
    cfg = style_cfg[idx]
    plt.plot(x, y, label=label, color=cfg["color"], marker=cfg["marker"],
             markevery=10, linewidth=1.4, markersize=6)

# 3. 图表标注（完全对齐原图）
plt.title(r"CIFAR10 ($Practical: \alpha=0.1$)", fontsize=16, pad=12)
plt.xlabel("Communication Rounds", fontsize=13)
plt.ylabel("Test Accuracy", fontsize=13)
plt.xlim(0, max_round)
plt.ylim(0.4, 0.48)
plt.xticks([0,40,80,120,160,200])
plt.grid(linestyle="--", alpha=0.5)
plt.legend(loc="lower right", framealpha=0.9)
plt.tight_layout()
plt.savefig(fig_save_path, bbox_inches="tight")
plt.show()
print(f"图片已保存至: {fig_save_path}")