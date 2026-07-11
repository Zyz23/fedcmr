import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

# 1. 原始通信开销数据 (请替换为你的真实 CSV 数据)
csv_data = """setting,method,upload_mb,download_mb,reach_round,n_participated,target_top1
CIFAR100-α0.1,FedDSE,2536900,2536900,307,20,35
CIFAR100-α0.1,FedDroupt,2536900,2536900,87,20,35
CIFAR100-α0.1,FedCMR,2536900,2536900,43,20,35
CIFAR10-α0.1,FedDSE,2536900,2536900,189,20,78
CIFAR10-α0.1,FedDroupt,2536900,2536900,162,20,78
CIFAR10-α0.1,FedCMR,2536900,2536900,153,20,78
tiny_Img-α0.5,FedDSE,9040193,9040193,60,20,20
tiny_Img-α0.5,FedDroupt,9040193,9040193,40,20,20
tiny_Img-α0.5,FedCMR,9040193,9040193,9,20,20
"""

df = pd.read_csv(io.StringIO(csv_data))

# 2. 计算总通信开销 (Total GB)
# 公式: (Upload + Download) * 轮次 * 每轮参与客户端数
# 最后除以 1024 将 MB 转换为 GB
df['total_comm_gb'] = (
    (df['upload_mb'] + df['download_mb']) * df['reach_round'] * df['n_participated']
) / 1024

# 3. 设置顶会风格
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

fig, ax = plt.subplots(figsize=(9, 6))

settings = df['setting'].unique()
methods = ['FedDSE', 'FedDroupt', 'FedCMR'] 

# 配色、宽度和纹理字典
palette = {'FedDSE': '#b9c0f1', 'FedDroupt': '#44b39e', 'FedCMR': '#dec76c'}
widths = {'FedDSE': 0.5, 'FedDroupt': 0.4, 'FedCMR': 0.3}  
hatches = {'FedDSE': '', 'FedDroupt': '', 'FedCMR': ''}

x_positions = np.arange(len(settings))

# 4. 原生 Matplotlib 绘制嵌套柱状图
for method in methods:
    y_values = []
    for setting in settings:
        val = df[(df['setting'] == setting) & (df['method'] == method)]['total_comm_gb'].values[0]
        y_values.append(val)
    
    ax.bar(
        x_positions, 
        y_values, 
        width=widths[method], 
        label=method, 
        color=palette[method], 
        edgecolor='black', 
        linewidth=1.5,
        hatch=hatches[method],
        zorder=3  
    )

# 5. 自动计算降低百分比并标注
baseline_method = 'FedDSE'
ours_method = 'FedCMR'

for i, setting in enumerate(settings):
    val_baseline = df[(df['setting'] == setting) & (df['method'] == baseline_method)]['total_comm_gb'].values[0]
    val_ours = df[(df['setting'] == setting) & (df['method'] == ours_method)]['total_comm_gb'].values[0]
    
    # 计算通信开销降低百分比
    reduction = (val_baseline - val_ours) / val_baseline * 100
    max_height_in_group = df[df['setting'] == setting]['total_comm_gb'].max()
    
    sign = '↓' if reduction > 0 else '↑'
    color = '#d62728' if reduction > 0 else 'black'
    
    # 添加箭头和百分比标注
    ax.annotate(
        f'{sign} {abs(reduction):.1f}%', 
        (x_positions[i], max_height_in_group),
        xytext=(0, 8), 
        textcoords='offset points', 
        ha='center', 
        va='bottom',
        color=color, 
        fontsize=14,
        fontweight='bold'
    )

# 6. 图表美化与布局调整
ax.set_title('Total Communication Overhead to Target Accuracy', fontweight='bold', pad=20, fontsize=17)
# 纵轴标签更新为 Communication (GB)
ax.set_ylabel('Total Communication (GB)', fontweight='bold', labelpad=12, fontsize=14)

formatted_labels = [s.replace('-', '\n') for s in settings]
ax.set_xticks(x_positions)
ax.set_xticklabels(formatted_labels, fontsize=14)

ax.legend(loc='upper right', frameon=True, edgecolor='black', fontsize=12)

ax.set_axisbelow(True)
ax.grid(axis='y', linestyle='--', alpha=0.7, color='gray', zorder=0)

y_max = df['total_comm_gb'].max()
ax.set_ylim(0, y_max * 1.15)

plt.tight_layout()
plt.savefig('FL_communication_nested_bar.pdf', format='pdf', bbox_inches='tight', transparent=True)
plt.show()