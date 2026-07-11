import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
'''
CIFAR100-α0.1,FedGen,0.555,73,4,2250,10,50
CIFAR100-α0.1,FedCMR,0.436,74,4,2250,10,50
CIFAR100-α0.1,FedCMR*,0.172,73,4,2250,10,50
CIFAR10-α0.1,FedGen,0.555,132,4,2250,5,80
CIFAR10-α0.1,FedCMR,0.436,151,4,2250,5,80
CIFAR10-α0.1,FedCMR*,0.172,151,4,2250,5,80
tiny_Img-α0.1,FedGen,0.555,37,4,2250,10,20
tiny_Img-α0.1,FedCMR,0.436,14,4,2250,10,20
tiny_Img-α0.1,FedCMR*,0.172,24,4,2250,10,20
'''

'''
CIFAR100-α0.1,FedDSE,0.013774608,37,4,2250,10,35
CIFAR100-α0.1,FedDroupt,0.013774608,67,4,2250,10,35
CIFAR100-α0.1,FedCMR,0.015302656,29,4,2250,10,35
CIFAR10-α0.5,FedDSE,0.013774608,85,4,2250,5,78
CIFAR10-α0.5,FedDroupt,0.013774608,80,4,2250,5,78
CIFAR10-α0.5,FedCMR,0.015302656,20,4,2250,5,78
tiny_Img-α0.1,FedDSE,0.013774608,60,4,2250,10,20
tiny_Img-α0.1,FedDroupt,0.013774608,44,4,2250,10,20
tiny_Img-α0.1,FedCMR,0.015302656,24,4,2250,10,20'''

"""
CIFAR100-10/100,Fedadam,2536900,2536900,307,20,40
CIFAR100-10/100,FedCMR,2748737,2748737,87,20,40
CIFAR100-10/100,FedCMR*,1241409,1241409,43,20,40
CIFAR10-2/10,Fedadam,2536900,2536900,189,20,50
CIFAR10-2/10,FedCMR,2748737,2748737,162,20,50
CIFAR10-2/10,FedCMR*,1241409,1241409,153,20,50
tiny_Img-20/200,Fedadam,8879656,8879656,145,20,20
tiny_Img-20/200,FedCMR,9040193,9040193,27,20,20
tiny_Img-20/200,FedCMR*, 2814273, 2814273,45,20,20
"""
# 1. 原始数据
csv_data = """setting,method,forward_gmacs_per_sample,reach_round,n_participated,samples_per_client_per_epoch,local_n_epochs,target_top1

CIFAR100-10/100,Fedadam,2536900,2536900,307,20,40
CIFAR100-10/100,FedCMR,2748737,2748737,87,20,40
CIFAR100-10/100,FedCMR*,1241409,1241409,43,20,40
CIFAR10-2/10,Fedadam,2536900,2536900,189,20,50
CIFAR10-2/10,FedCMR,2748737,2748737,162,20,50
CIFAR10-2/10,FedCMR*,1241409,1241409,153,20,50
tiny_Img-20/200,Fedadam,8879656,8879656,145,20,20
tiny_Img-20/200,FedCMR,9040193,9040193,27,20,20
tiny_Img-20/200,FedCMR*, 2814273, 2814273,45,20,20"""

df = pd.read_csv(io.StringIO(csv_data))

# 2. 计算总计算开销 (Tera-MACs)
df['total_gmacs'] = (
    df['forward_gmacs_per_sample'] * df['reach_round'] * df['n_participated'] * df['samples_per_client_per_epoch'] * df['local_n_epochs']
)
df['total_tmacs'] = df['total_gmacs'] / 1000

# 3. 设置顶会风格
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

fig, ax = plt.subplots(figsize=(9, 6))

settings = df['setting'].unique()
# 必须按计算量从大到小排列，确保宽柱子在下，窄柱子在上
# 【修改点1】：将方法名称更新为你数据里的新方法
methods = ['Fedadam', 'FedCMR', 'FedCMR'] 

# 【修改点2】：同步更新配色、宽度和纹理字典的 Key
palette = {'Fedadam': '#b9c0f1', 'FedCMR': '#44b39e', 'FedCMR*': '#dec76c'}
# 宽度从宽到窄递减，实现嵌套效果 (宽的画在最下面，窄的画在最上面)
widths = {'Fedadam': 0.6, 'FedCMR': 0.4, 'FedCMR*': 0.2}  
hatches = {'Fedadam': '', 'FedCMR': '', 'FedCMR*': ''}

x_positions = np.arange(len(settings))

# 4. 原生 Matplotlib 绘制嵌套柱状图
for method in methods:
    y_values = []
    for setting in settings:
        val = df[(df['setting'] == setting) & (df['method'] == method)]['total_tmacs'].values[0]
        y_values.append(val)
    
    # 所有的柱子使用完全相同的 x_positions，实现“在同一列”的效果
    ax.bar(
        x_positions, 
        y_values, 
        width=widths[method], 
        label=method, 
        color=palette[method], 
        edgecolor='black', 
        linewidth=1.5,
        hatch=hatches[method],
        zorder=3  # 确保柱子画在网格线上方
    )

# 5. 自动计算降低百分比并标注在最高处
for i, setting in enumerate(settings):
    baseline_method = 'Fedadam'
    ours_method = 'FedCMR'
    
    val_baseline = df[(df['setting'] == setting) & (df['method'] == baseline_method)]['total_tmacs'].values[0]
    val_ours = df[(df['setting'] == setting) & (df['method'] == ours_method)]['total_tmacs'].values[0]
    # 计算降低百分比: (Baseline - Ours) / Baseline
    reduction = (val_baseline - val_ours) / val_baseline * 100
    
    # 找到这三个柱子中的最高点，以便把文字标注在最上方，防止文字和柱子重叠
    max_height_in_group = df[df['setting'] == setting]['total_tmacs'].max()
    
    # 如果我们的方法比 Baseline 开销还大，就显示增加 (这是容错处理)
    sign = '↓' if reduction > 0 else '↑'
    color = '#d62728' if reduction > 0 else 'black'
    
    # 添加箭头和百分比标注
    ax.annotate(
        f'{sign} {abs(reduction):.1f}%', 
        (x_positions[i], max_height_in_group), # 定位在该组柱子的最高处
        xytext=(0, 8),             # 向上偏移 8 像素
        textcoords='offset points', 
        ha='center', 
        va='bottom',
        color=color, 
        fontsize=14,
        fontweight='bold'
    )

# 6. 图表美化与布局调整
ax.set_title('Total Computational Overhead to Target Accuracy', fontweight='bold', pad=20, fontsize=17)
ax.set_ylabel('Total Computation (Tera-MACs)', fontweight='bold', labelpad=12, fontsize=14)

# 格式化 X 轴标签
formatted_labels = [s.replace('-', '\n') for s in settings]
ax.set_xticks(x_positions)
ax.set_xticklabels(formatted_labels, fontsize=14)

# 设置图例
ax.legend(loc='upper right', frameon=True, edgecolor='black', fontsize=12)

# 设置背景横向网格线
ax.set_axisbelow(True)
ax.grid(axis='y', linestyle='--', alpha=0.7, color='gray', zorder=0)

# 自动拉高 Y 轴上限，防止文字标注被截断
y_max = df['total_tmacs'].max()
ax.set_ylim(0, y_max * 1.15)

plt.tight_layout()
plt.savefig('FL_computational_nested_bar.pdf', format='pdf', bbox_inches='tight', transparent=True)
# plt.show()