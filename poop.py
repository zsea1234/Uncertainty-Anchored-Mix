import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

# ================= 1. 数据准备 =================
data = [
    # --- Weakly-supervised (Baselines) ---
    {'Method': 'PA-Seg',        'Dice': 0.638, 'GFLOPs': 18.5, 'Params': 12},
    {'Method': 'BoxInst',       'Dice': 0.581, 'GFLOPs': 22.5, 'Params': 25},
    {'Method': 'LC-MIL',        'Dice': 0.593, 'GFLOPs': 21.5, 'Params': 11},
    {'Method': 'CycleMix',      'Dice': 0.800, 'GFLOPs': 28.0, 'Params': 30},
    {'Method': 'ScribFormer',   'Dice': 0.839, 'GFLOPs': 96.15, 'Params': 114.6}, 
    {'Method': 'CIM',           'Dice': 0.833, 'GFLOPs': 25.96, 'Params': 24},
    
    # --- Fully Supervised (Upper Bound) ---
    {'Method': 'nnUNet',        'Dice': 0.902, 'GFLOPs': 35.0, 'Params': 40},

    # --- Your Method (Highlight) ---
    {'Method': 'Ours',          'Dice': 0.859, 'GFLOPs': 19.0, 'Params': 18},
]

df = pd.DataFrame(data)

# ================= 2. 样式设置 =================
plt.figure(figsize=(12, 8)) # 调整画布比例，使其更紧凑
plt.style.use('seaborn-v0_8-whitegrid')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# --- 颜色映射 ---
color_dict = {
    'Ours': '#D62728',          # 红
    'nnUNet': '#1F77B4',        # 蓝
    'ScribFormer': '#9467BD'    # 紫
}
others = [m for m in df['Method'] if m not in color_dict]
palette = sns.color_palette("husl", len(others))
for i, m in enumerate(others):
    color_dict[m] = palette[i]

SCALE_FACTOR = 18 

# ================= 3. 绘制主图 =================
# 全监督基准线
plt.axhline(y=0.902, color='#1F77B4', linestyle='--', alpha=0.4, linewidth=1.5)
# 调整 Upper Bound 文字位置，避免被气泡遮挡
plt.text(90, 0.905, 'Upper Bound (0.902)', color='#1F77B4', fontsize=11, fontstyle='italic', ha='center')

for i, row in df.iterrows():
    method = row['Method']
    is_ours = method == 'Ours'
    
    # 样式配置
    color = color_dict[method]
    size = row['Params'] * SCALE_FACTOR
    alpha = 0.95 if is_ours else 0.65
    edgecolor = 'black' if is_ours else 'white'
    zorder = 100 if is_ours else 10
    
    # 1. 画气泡
    plt.scatter(
        row['GFLOPs'], row['Dice'], s=size, c=color, 
        alpha=alpha, edgecolors=edgecolor, linewidth=1.5, zorder=zorder
    )
    
    # 2. 标注数值 (Params)
    y_offset = 0.012
    x_offset = 0
    ha = 'center'
    
    # 位置微调
    if 'ScribFormer' in method:
        y_offset = -0.028 # 放在大球下方
    elif 'CycleMix' in method:
        y_offset = -0.018 
    elif 'BoxInst' in method:
        y_offset = -0.018
    elif 'Ours' in method:
        y_offset = 0.018  
    
    fw = 'bold'
    fs = 14 if is_ours else 11
    fc = '#8B0000' if is_ours else '#333333'
    
    # 格式化参数量
    params_val = row['Params']
    params_str = f"{int(params_val)}M" if params_val == int(params_val) else f"{params_val}M"

    if is_ours:
        label_text = f"Ours\n{params_str}"
    else:
        label_text = f"{params_str}"
    
    plt.text(
        row['GFLOPs'] + x_offset, row['Dice'] + y_offset, 
        label_text, 
        ha=ha, va='center',
        fontsize=fs, fontweight=fw, color=fc, zorder=zorder+1
    )

# ================= 4. 构建图例 (放入右下角) =================

method_handles = []
sorted_methods = ['Ours'] + [m for m in df['Method'] if m != 'Ours']

for method in sorted_methods:
    h = Line2D(
        [0], [0], 
        marker='o', 
        color='w', 
        label=method,
        markerfacecolor=color_dict[method], 
        markersize=11, 
        markeredgecolor='white'
    )
    method_handles.append(h)

# ★★★ 修改图例位置：放入右下角空白处 ★★★
plt.legend(
    handles=method_handles,
    title='Methods',
    title_fontsize=13,
    fontsize=11,
    loc='lower right',      # 自动定位到右下角
    bbox_to_anchor=(0.98, 0.02), # 微调边距 (x, y)，确保不贴边
    frameon=True,
    framealpha=0.9,         # 增加背景不透明度，防止遮挡网格线
    edgecolor='gray',       # 加个淡边框更精致
    labelspacing=1.1, 
    borderpad=0.8
)

# ================= 5. 坐标轴与输出 =================
plt.title('Performance vs. Efficiency on MSCMR Dataset', fontsize=24, fontweight='bold', pad=20)
plt.xlabel('Computational Cost (GFLOPs)', fontsize=18, fontweight='bold')
plt.ylabel('Dice Similarity Coefficient (DSC)', fontsize=18, fontweight='bold')

plt.ylim(0.55, 0.95)
plt.xlim(10, 105)
plt.grid(True, which='major', linestyle='--', alpha=0.5)

plt.tight_layout()
save_name = 'mscmr_final_layout_optimized.png'
plt.savefig(save_name, dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ 图例已移至右下角，图表已生成: {save_name}")