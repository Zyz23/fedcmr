import matplotlib.pyplot as plt
import numpy as np

# 1. 数据
# participate_rate = ["0.1", "0.05", "0.01", "10/100"]
# cifar10_acc   = [59.52, 59.78, 58.66, 58.56]
# fedavg_acc    = [53.23, 50.23, 46.43, 45.97]
# cifar100_acc  = [53.15, 51.23, 46.32, 52.51]
# tiny_acc      = [50.71, 47.35, 44.23, 43.38]
participate_rate = ["0.2", "0.5", "1"]
cifar10_acc   = [47.43, 47.54, 47.17]
fedavg_acc    = [40.05, 42.49, 43.58]
cifar100_acc  = [39.75, 41.83, 42.36]
tiny_acc      = [38.54, 39.93, 41.29]

# 2. 移除Times New Roman，使用系统默认字体，消除字体警告
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示

plt.figure(figsize=(8, 5), dpi=150)

# 绘图
plt.plot(participate_rate, cifar10_acc,  marker='o', linewidth=1.2, label='FedCMR(our)')
plt.plot(participate_rate, fedavg_acc,   marker='d', linewidth=1.2, label='FedAvg')
plt.plot(participate_rate, cifar100_acc, marker='s', linewidth=1.2, label='FedDensemble')
plt.plot(participate_rate, tiny_acc,     marker='^', linewidth=1.2, label='FedAlign')

# 坐标轴
plt.xlabel(' participation Rate $P$')
plt.ylabel('Test Accuracy (%)')
plt.xticks(participate_rate)
plt.legend(loc='best', fontsize=8)
plt.grid(alpha=0.3, linestyle='--')

# 修复：$lpha → \alpha
# plt.title('Accuracy Changes ')

plt.tight_layout()
plt.savefig('mixup随参与率变化.pdf', bbox_inches='tight')
plt.show()