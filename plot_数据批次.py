import matplotlib.pyplot as plt
import numpy as np

# 1. 数据
participate_rate = [8,16,32,64]
cifar10_acc   = [72.92, 72.84, 74.72, 71.02]
cifar100_acc  = [32.13, 33.2, 33.51, 43.86]
tiny_acc      = [26.6, 30.06, 31.71, 31.57]

# 2. 移除Times New Roman，使用系统默认字体，消除字体警告
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示

plt.figure(figsize=(8, 5), dpi=150)

# 绘图
plt.plot(participate_rate, cifar10_acc,  marker='o', linewidth=1.2, label='CIFAR-10')
plt.plot(participate_rate, cifar100_acc, marker='s', linewidth=1.2, label='CIFAR-100')
plt.plot(participate_rate, tiny_acc,     marker='^', linewidth=1.2, label='Tiny-ImageNet')

# 坐标轴
plt.xlabel(' Batch Size')
plt.ylabel('Test Accuracy (%)')
plt.xticks(participate_rate)
plt.legend(loc='best', fontsize=8)
plt.grid(alpha=0.3, linestyle='--')

# 修复：$lpha → \alpha
plt.title('Accuracy Changes with $BS$')

plt.tight_layout()
plt.savefig('本地批次分析.pdf', bbox_inches='tight')
plt.show()