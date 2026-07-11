import matplotlib.pyplot as plt
import numpy as np

# 1. 数据
participate_rate = [0.2, 0.4, 0.6, 0.8, 1.0]
cifar10_acc   = [71.02, 77.54, 78.14, 78.73, 78.62]
cifar100_acc  = [43.86, 42.37, 45.90, 46.56, 48.40]
tiny_acc      = [29.68, 32.88,	33.17,	33.08,	33.05]

# 2. 移除Times New Roman，使用系统默认字体，消除字体警告
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示

plt.figure(figsize=(8, 5), dpi=150)

# 绘图
plt.plot(participate_rate, cifar10_acc,  marker='o', linewidth=1.2, label='CIFAR-10')
plt.plot(participate_rate, cifar100_acc, marker='s', linewidth=1.2, label='CIFAR-100')
plt.plot(participate_rate, tiny_acc,     marker='^', linewidth=1.2, label='Tiny-ImageNet')

# 坐标轴
plt.xlabel(' Participation Rate $P$')
plt.ylabel('Test Accuracy (%)')
plt.xticks(participate_rate)
plt.legend(loc='best', fontsize=8)
plt.grid(alpha=0.3, linestyle='--')

# 修复：$lpha → \alpha
plt.title('Accuracy Changes with $P$')

plt.tight_layout()
plt.savefig('参与率超惨分析.pdf', bbox_inches='tight')
plt.show()