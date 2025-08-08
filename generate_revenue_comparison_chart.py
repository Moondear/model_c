import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import rcParams
import matplotlib.patches as mpatches
import json

# 设置中文字体和学术期刊风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# 专业学术期刊配色方案
colors_academic = [
    '#1f77b4',  # 专业蓝
    '#ff7f0e',  # 学术橙
    '#2ca02c',  # 自然绿
    '#d62728',  # 科学红
    '#9467bd',  # 紫罗兰
    '#8c564b',  # 棕褐色
]

# 收益对比专用配色
scenario_colors = ['#3182bd', '#fd8d3c']  # 深蓝、橙色
improvement_colors = ['#2ca02c', '#74c476', '#a1d99b']  # 绿色渐变
cost_revenue_colors = ['#d62728', '#1f77b4', '#2ca02c']  # 红、蓝、绿

def create_revenue_comparison_analysis():
    """
    生成图5.6：两情景收益对比与经济效益分析
    使用从实际程序运行结果提取的真实数据
    """
    
    # 加载真实收益数据
    try:
        with open('real_revenue_data.json', 'r', encoding='utf-8') as f:
            real_data = json.load(f)
        print("✅ 已加载真实收益数据")
    except FileNotFoundError:
        print("❌ 未找到真实收益数据文件，请先运行 simplified_real_data.py")
        return
    
    # 创建图形布局：2行2列
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('图5.6 两情景收益对比与经济效益分析', fontsize=16, fontweight='bold', y=0.95)
    
    # ============ 子图1：7年收益变化趋势对比 ============
    years = real_data['yearly_revenues']['years']
    
    # 使用真实的年度收益数据（万元）
    scenario1_revenue = real_data['yearly_revenues']['scenario1']  # 情景一：超产滞销
    scenario2_revenue = real_data['yearly_revenues']['scenario2']  # 情景二：50%折价销售
    
    # 计算累积收益
    scenario1_cumulative = np.cumsum(scenario1_revenue)
    scenario2_cumulative = np.cumsum(scenario2_revenue)
    
    # 绘制年度收益趋势
    ax1.plot(years, scenario1_revenue, 'o-', color=scenario_colors[0], linewidth=3, 
             markersize=8, label='情景一：超产滞销', markerfacecolor='white', 
             markeredgecolor=scenario_colors[0], markeredgewidth=2)
    ax1.plot(years, scenario2_revenue, 's-', color=scenario_colors[1], linewidth=3, 
             markersize=8, label='情景二：50%折价销售', markerfacecolor='white',
             markeredgecolor=scenario_colors[1], markeredgewidth=2)
    
    # 填充收益差异区域
    ax1.fill_between(years, scenario1_revenue, scenario2_revenue, 
                     alpha=0.3, color=improvement_colors[1], label='收益提升区域')
    
    # 添加年度收益提升百分比标注
    for i, (year, rev1, rev2) in enumerate(zip(years, scenario1_revenue, scenario2_revenue)):
        improvement_pct = (rev2 - rev1) / rev1 * 100
        if i % 2 == 0:  # 只标注部分年份避免拥挤
            ax1.annotate(f'+{improvement_pct:.1f}%', 
                        xy=(year, rev2), xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom', fontweight='bold', 
                        color=improvement_colors[0], fontsize=9)
    
    ax1.set_xlabel('年份', fontsize=12)
    ax1.set_ylabel('年度净收益 (万元)', fontsize=12)
    ax1.set_title('(a) 7年收益变化趋势对比', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ============ 子图2：收益提升分解瀑布图 ============
    # 使用真实的收益提升分解数据
    waterfall = real_data['waterfall_breakdown']
    improvement_sources = ['基础收益', '正常销售\n超产收益', '折价销售\n收益', '总收益']
    values = [
        waterfall['base_revenue'], 
        waterfall['normal_sales_improvement'], 
        waterfall['discount_sales_improvement'], 
        waterfall['total_revenue']
    ]
    
    # 计算瀑布图的累积值
    cumulative = [0, values[0], values[0] + values[1], values[0] + values[1] + values[2]]
    
    # 绘制瀑布图
    for i in range(len(improvement_sources)-1):
        if i == 0:
            # 基础收益柱
            ax2.bar(i, values[i], color=scenario_colors[0], alpha=0.8, 
                   edgecolor='white', linewidth=1.5)
        else:
            # 增量收益柱
            ax2.bar(i, values[i], bottom=cumulative[i], color=improvement_colors[i-1], 
                   alpha=0.8, edgecolor='white', linewidth=1.5)
        
        # 添加数值标签
        if i == 0:
            y_pos = values[i] / 2
        else:
            y_pos = cumulative[i] + values[i] / 2
        
        ax2.text(i, y_pos, f'{values[i]}万元', ha='center', va='center', 
                fontweight='bold', color='white', fontsize=10)
    
    # 总收益柱
    ax2.bar(len(improvement_sources)-1, values[-1], color=scenario_colors[1], 
           alpha=0.8, edgecolor='white', linewidth=1.5)
    ax2.text(len(improvement_sources)-1, values[-1]/2, f'{values[-1]}万元', 
            ha='center', va='center', fontweight='bold', color='white', fontsize=10)
    
    # 连接线显示累积效果
    for i in range(len(improvement_sources)-2):
        ax2.plot([i+0.4, i+0.6], [cumulative[i+1], cumulative[i+1]], 
                'k--', alpha=0.5, linewidth=1)
    
    ax2.set_ylabel('收益 (万元)', fontsize=12)
    ax2.set_title('(b) 收益提升来源分解分析', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(improvement_sources)))
    ax2.set_xticklabels(improvement_sources, fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ============ 子图3：成本收入结构对比 ============
    # 使用真实的成本收入结构数据
    cost_revenue = real_data['cost_revenue_structure']
    categories = ['总收入', '总成本', '净收益']
    scenario1_values = [
        cost_revenue['scenario1']['total_revenue'],
        cost_revenue['scenario1']['total_cost'],
        cost_revenue['scenario1']['net_profit']
    ]
    scenario2_values = [
        cost_revenue['scenario2']['total_revenue'],
        cost_revenue['scenario2']['total_cost'],
        cost_revenue['scenario2']['net_profit']
    ]
    
    x_pos = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax3.bar(x_pos - width/2, scenario1_values, width, label='情景一：超产滞销',
                   color=scenario_colors[0], alpha=0.8, edgecolor='white', linewidth=1.5)
    bars2 = ax3.bar(x_pos + width/2, scenario2_values, width, label='情景二：50%折价销售',
                   color=scenario_colors[1], alpha=0.8, edgecolor='white', linewidth=1.5)
    
    # 添加数值标签和提升标注
    for i, (bar1, bar2, val1, val2) in enumerate(zip(bars1, bars2, scenario1_values, scenario2_values)):
        # 情景一数值
        ax3.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 50,
                f'{val1}万元', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 情景二数值
        ax3.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 50,
                f'{val2}万元', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 提升标注（除了成本）
        if i != 1:  # 成本相同，不标注提升
            improvement = val2 - val1
            improvement_pct = improvement / val1 * 100
            ax3.annotate(f'+{improvement}万元\n(+{improvement_pct:.1f}%)', 
                        xy=(i, max(val1, val2) + 200), ha='center', va='bottom',
                        fontweight='bold', color=improvement_colors[0], fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax3.set_ylabel('金额 (万元)', fontsize=12)
    ax3.set_title('(c) 7年总成本收入结构对比', fontsize=13, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(categories, fontsize=11)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # ============ 子图4：不同作物类型收益贡献分析 ============
    # 使用真实的作物类型收益贡献数据
    crop_contributions = real_data['crop_type_contributions']
    crop_types = ['粮食类\n作物', '蔬菜类\n作物', '食用菌\n类']
    revenue_contribution_s1 = [
        crop_contributions['scenario1']['粮食类'],
        crop_contributions['scenario1']['蔬菜类'],
        crop_contributions['scenario1']['食用菌']
    ]
    revenue_contribution_s2 = [
        crop_contributions['scenario2']['粮食类'],
        crop_contributions['scenario2']['蔬菜类'],
        crop_contributions['scenario2']['食用菌']
    ]
    
    # 计算收益提升
    improvements = [s2 - s1 for s1, s2 in zip(revenue_contribution_s1, revenue_contribution_s2)]
    
    # 创建双轴图
    ax4_twin = ax4.twinx()
    
    x_pos = np.arange(len(crop_types))
    width = 0.25
    
    # 收益贡献柱状图
    bars1 = ax4.bar(x_pos - width, revenue_contribution_s1, width, label='情景一收益',
                    color=scenario_colors[0], alpha=0.8, edgecolor='white', linewidth=1)
    bars2 = ax4.bar(x_pos, revenue_contribution_s2, width, label='情景二收益',
                    color=scenario_colors[1], alpha=0.8, edgecolor='white', linewidth=1)
    
    # 收益提升折线图
    line = ax4_twin.plot(x_pos, improvements, 'o-', color=improvement_colors[0], 
                        linewidth=3, markersize=8, label='收益提升', 
                        markerfacecolor='white', markeredgecolor=improvement_colors[0], 
                        markeredgewidth=2)
    
    # 添加数值标签
    for i, (imp, s1, s2) in enumerate(zip(improvements, revenue_contribution_s1, revenue_contribution_s2)):
        if imp > 0:
            ax4_twin.text(i, imp + 20, f'+{imp}万元', ha='center', va='bottom', 
                         fontweight='bold', color=improvement_colors[0], fontsize=9)
        
        # 柱状图数值标签
        if i % 2 == 0:  # 避免标签过密
            ax4.text(i - width, s1 + 30, f'{s1}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=9)
            ax4.text(i, s2 + 30, f'{s2}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=9)
    
    ax4.set_xlabel('作物类型', fontsize=12)
    ax4.set_ylabel('收益贡献 (万元)', fontsize=12)
    ax4_twin.set_ylabel('收益提升 (万元)', fontsize=12, color=improvement_colors[0])
    ax4.set_title('(d) 不同作物类型收益贡献分析', fontsize=13, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(crop_types, fontsize=10)
    
    # 合并图例
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    
    ax4.grid(True, alpha=0.3)
    
    # 添加关键指标文本框（使用真实数据）
    total_improvement = sum(improvements)
    improvement_rate = real_data['key_metrics']['improvement_rate']
    key_metrics = ("关键指标:\n"
                  f"• 总收益提升: {total_improvement:.0f}万元\n"
                  f"• 提升率: {improvement_rate:.1f}%\n"
                  f"• 主要贡献: 蔬菜类+食用菌")
    
    ax4.text(0.98, 0.95, key_metrics, transform=ax4.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=improvement_colors[2], alpha=0.8),
            fontsize=9, ha='right', va='top', fontweight='bold')
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)
    
    # 只保存PNG图片，不显示窗口
    plt.savefig('图5.6_两情景收益对比与经济效益分析.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 输出关键信息（使用真实数据）
    metrics = real_data['key_metrics']
    print("="*60)
    print("图5.6 两情景收益对比与经济效益分析 - 真实数据")
    print("="*60)
    print("📈 收益对比总览：")
    print(f"   - 情景一（超产滞销）7年总净收益: {metrics['scenario1_total']:.1f}万元")
    print(f"   - 情景二（50%折价销售）7年总净收益: {metrics['scenario2_total']:.1f}万元")
    print(f"   - 绝对收益提升: {metrics['absolute_improvement']:.1f}万元")
    print(f"   - 相对收益提升: {metrics['improvement_rate']:.1f}%")
    print()
    print("💰 收益来源分析：")
    print(f"   - 基础收益（情景一水平）: {waterfall['base_revenue']:.0f}万元")
    print(f"   - 正常销售超产收益: {waterfall['normal_sales_improvement']:.0f}万元")
    print(f"   - 折价销售额外收益: {waterfall['discount_sales_improvement']:.0f}万元")
    print(f"   - 总收益（情景二）: {waterfall['total_revenue']:.0f}万元")
    print()
    print("📊 成本收入结构：")
    revenue_increase = cost_revenue['scenario2']['total_revenue'] - cost_revenue['scenario1']['total_revenue']
    revenue_increase_pct = revenue_increase / cost_revenue['scenario1']['total_revenue'] * 100
    net_increase = cost_revenue['scenario2']['net_profit'] - cost_revenue['scenario1']['net_profit']
    net_increase_pct = net_increase / cost_revenue['scenario1']['net_profit'] * 100
    print(f"   - 7年总收入提升: {revenue_increase:.0f}万元 (+{revenue_increase_pct:.1f}%)")
    print(f"   - 7年总成本保持: {cost_revenue['scenario1']['total_cost']:.0f}万元 (基本不变)")
    print(f"   - 7年净收益提升: {net_increase:.0f}万元 (+{net_increase_pct:.1f}%)")
    print()
    print("🌾 作物类型贡献：")
    for i, crop_type in enumerate(['粮食类', '蔬菜类', '食用菌']):
        improvement = improvements[i]
        print(f"   - {crop_type}作物收益提升: +{improvement:.0f}万元")
    print()
    print("🎯 经济效益结论：")
    print(f"   - 情景二相比情景一收益提升{metrics['improvement_rate']:.1f}%")
    print(f"   - 年均额外收益约{metrics['absolute_improvement']/7:.0f}万元")
    print("   - 超产作物通过折价销售实现价值最大化")
    print("   - 高价值作物（蔬菜、食用菌）贡献最大")
    print("="*60)
    print("✅ 图片已生成：图5.6_两情景收益对比与经济效益分析.png (基于真实数据)")

# 运行函数生成图像
if __name__ == "__main__":
    create_revenue_comparison_analysis()
