import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import rcParams
import matplotlib.patches as mpatches

# 设置中文字体和学术期刊风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# 学术期刊配色方案
colors_academic = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#4CAF50', '#9C27B0']
colors_land = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

def create_land_distribution_analysis():
    """
    生成地块分布与作物适应性分析图（图5.1）
    专业美观，适合学术论文插入
    """
    
    # 创建图形布局：2行2列
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('图5.1 地块分布与作物适应性分析', fontsize=16, fontweight='bold', y=0.95)
    
    # ============ 子图1：地块分布饼图 ============
    land_types = ['平旱地', '梯田', '山坡地', '水浇地', '普通大棚', '智慧大棚']
    land_counts = [6, 14, 6, 8, 16, 4]  # 根据论文数据
    land_areas = [480, 840, 360, 480, 9.6, 2.4]  # 假设面积数据（可调整）
    
    # 饼图配色和样式
    wedges, texts, autotexts = ax1.pie(land_counts, labels=land_types, autopct='%1.1f%%',
                                       colors=colors_land, startangle=90,
                                       explode=(0.05, 0.05, 0.05, 0.1, 0.1, 0.1))
    
    # 设置文字样式
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    
    ax1.set_title('(a) 54个地块类型分布', fontsize=12, fontweight='bold', pad=20)
    
    # ============ 子图2：地块类型与最大季数分析 ============
    land_groups = ['粮食地块组\n(A/B/C类)', '水浇地组\n(D类)', '大棚组\n(E/F类)']
    land_counts_group = [26, 8, 20]
    max_seasons = [1, 2, 2]
    
    x_pos = np.arange(len(land_groups))
    bars1 = ax2.bar(x_pos - 0.2, land_counts_group, 0.4, label='地块数量', 
                    color=colors_academic[0], alpha=0.8)
    bars2 = ax2.bar(x_pos + 0.2, [s*10 for s in max_seasons], 0.4, label='最大季数×10', 
                    color=colors_academic[1], alpha=0.8)
    
    # 添加数值标签
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        ax2.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.5,
                str(land_counts_group[i]), ha='center', va='bottom', fontweight='bold')
        ax2.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.5,
                f"γ={max_seasons[i]}", ha='center', va='bottom', fontweight='bold')
    
    ax2.set_xlabel('地块组类型', fontsize=11)
    ax2.set_ylabel('数量', fontsize=11)
    ax2.set_title('(b) 地块组与最大种植季数', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(land_groups)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # ============ 子图3：作物适应性统计 ============
    crop_categories = ['粮食类\n(1-15号)', '水稻\n(16号)', '蔬菜类\n(17-37号)', '食用菌\n(38-41号)']
    crop_counts = [15, 1, 21, 4]
    
    # 各地块类型适应的作物数量
    land_adaptation = {
        '平旱地': [15, 0, 0, 0],  # 只适应粮食类
        '梯田': [15, 0, 0, 0],
        '山坡地': [15, 0, 0, 0],
        '水浇地': [0, 1, 21, 0],  # 适应水稻和蔬菜
        '普通大棚': [0, 0, 21, 4],  # 适应蔬菜和食用菌
        '智慧大棚': [0, 0, 21, 0]   # 只适应蔬菜
    }
    
    # 创建堆叠柱状图
    bottom = np.zeros(len(land_types))
    for i, category in enumerate(crop_categories):
        values = [land_adaptation[land][i] for land in land_types]
        bars = ax3.bar(land_types, values, bottom=bottom, label=category,
                      color=colors_academic[i], alpha=0.8)
        bottom += values
    
    ax3.set_xlabel('地块类型', fontsize=11)
    ax3.set_ylabel('适宜作物数量', fontsize=11)
    ax3.set_title('(c) 各地块类型适宜作物统计', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # ============ 子图4：豆类作物轮作适应性 ============
    bean_data = {
        '地块类型': ['粮食地块', '水浇地', '普通大棚', '智慧大棚'],
        '粮食豆类(1-5号)': [5, 0, 0, 0],
        '蔬菜豆类(17-19号)': [0, 3, 3, 3],
        '总豆类适应数': [5, 3, 3, 3]
    }
    
    x_pos = np.arange(len(bean_data['地块类型']))
    width = 0.25
    
    bars1 = ax4.bar(x_pos - width, bean_data['粮食豆类(1-5号)'], width,
                   label='粮食豆类(1-5号)', color=colors_academic[4], alpha=0.8)
    bars2 = ax4.bar(x_pos, bean_data['蔬菜豆类(17-19号)'], width,
                   label='蔬菜豆类(17-19号)', color=colors_academic[5], alpha=0.8)
    bars3 = ax4.bar(x_pos + width, bean_data['总豆类适应数'], width,
                   label='总计', color='gray', alpha=0.6)
    
    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax4.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                        str(int(height)), ha='center', va='bottom', fontweight='bold')
    
    ax4.set_xlabel('地块类型', fontsize=11)
    ax4.set_ylabel('豆类作物品种数', fontsize=11)
    ax4.set_title('(d) 豆类作物适应性分析', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(bean_data['地块类型'])
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # 只保存PNG图片，不显示窗口，不生成PDF
    plt.savefig('图5.1_地块分布与作物适应性分析.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    # 关闭图形释放内存
    plt.close()
    
    # 输出关键统计信息
    print("="*60)
    print("图5.1 地块分布与作物适应性分析 - 关键数据")
    print("="*60)
    print(f"📊 地块分布：")
    print(f"   - 粮食地块组：{sum(land_counts[:3])}个 (平旱地{land_counts[0]}+梯田{land_counts[1]}+山坡地{land_counts[2]})")
    print(f"   - 水浇地组：{land_counts[3]}个")
    print(f"   - 大棚组：{sum(land_counts[4:])}个 (普通{land_counts[4]}+智慧{land_counts[5]})")
    print(f"   - 总计：{sum(land_counts)}个地块")
    print()
    print(f"🌾 作物适应性：")
    print(f"   - 粮食类作物：15种，适宜粮食地块")
    print(f"   - 水稻：1种，仅适宜水浇地")
    print(f"   - 蔬菜类作物：21种，适宜水浇地和大棚")
    print(f"   - 食用菌：4种，仅适宜普通大棚")
    print()
    print(f"🫘 豆类轮作：")
    print(f"   - 粮食豆类(1-5号)：5种，适宜粮食地块")
    print(f"   - 蔬菜豆类(17-19号)：3种，适宜水浇地和大棚")
    print(f"   - 3年轮作覆盖：确保54个地块均满足豆类种植要求")
    print("="*60)
    print("✅ 图片已生成：图5.1_地块分布与作物适应性分析.png")

# 运行函数生成图像
if __name__ == "__main__":
    create_land_distribution_analysis()
