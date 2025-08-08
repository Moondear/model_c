#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成图5.6的分割版本：将四个子图分别保存为独立PNG文件
基于真实的程序运行结果数据
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import rcParams
import matplotlib.patches as mpatches
import json
from matplotlib import patheffects

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

def create_separate_revenue_charts():
    """
    生成图5.6的分割版本：四个独立的子图
    """
    
    # 加载真实收益数据
    try:
        with open('real_revenue_data.json', 'r', encoding='utf-8') as f:
            real_data = json.load(f)
        print("✅ 已加载真实收益数据")
    except FileNotFoundError:
        print("❌ 未找到真实收益数据文件，请先运行 simplified_real_data.py")
        return
    
    # 提取数据
    years = real_data['yearly_revenues']['years']
    scenario1_revenue = real_data['yearly_revenues']['scenario1']
    scenario2_revenue = real_data['yearly_revenues']['scenario2']
    waterfall = real_data['waterfall_breakdown']
    cost_revenue = real_data['cost_revenue_structure']
    crop_contributions = real_data['crop_type_contributions']
    metrics = real_data['key_metrics']
    
    # ============ 子图5.6a：7年收益变化趋势对比 ============
    fig_a, ax_a = plt.subplots(1, 1, figsize=(10, 6))
    
    # 绘制年度收益趋势
    ax_a.plot(years, scenario1_revenue, 'o-', color=scenario_colors[0], linewidth=3, 
             markersize=8, label='情景一：超产滞销', markerfacecolor='white', 
             markeredgecolor=scenario_colors[0], markeredgewidth=2)
    ax_a.plot(years, scenario2_revenue, 's-', color=scenario_colors[1], linewidth=3, 
             markersize=8, label='情景二：50%折价销售', markerfacecolor='white',
             markeredgecolor=scenario_colors[1], markeredgewidth=2)
    
    # 填充收益差异区域
    ax_a.fill_between(years, scenario1_revenue, scenario2_revenue, 
                     alpha=0.3, color=improvement_colors[1], label='收益提升区域')
    
    # 添加年度收益提升百分比标注
    for i, (year, rev1, rev2) in enumerate(zip(years, scenario1_revenue, scenario2_revenue)):
        improvement_pct = (rev2 - rev1) / rev1 * 100
        if i % 2 == 0:  # 只标注部分年份避免拥挤
            ax_a.annotate(f'+{improvement_pct:.1f}%', 
                        xy=(year, rev2), xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom', fontweight='bold', 
                        color=improvement_colors[0], fontsize=9)
    
    ax_a.set_xlabel('年份', fontsize=14)
    ax_a.set_ylabel('年度净收益 (万元)', fontsize=14)
    ax_a.set_title('图5.6a 7年收益变化趋势对比', fontsize=15, fontweight='bold', pad=20)
    ax_a.legend(loc='upper left', fontsize=12)
    ax_a.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('图5.6a_年度收益变化趋势对比.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # ============ 子图5.6b：收益提升分解瀑布图 ============
    fig_b, ax_b = plt.subplots(1, 1, figsize=(10, 6))
    
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
            ax_b.bar(i, values[i], color=scenario_colors[0], alpha=0.8, 
                   edgecolor='white', linewidth=1.5)
        else:
            # 增量收益柱
            ax_b.bar(i, values[i], bottom=cumulative[i], color=improvement_colors[i-1], 
                   alpha=0.8, edgecolor='white', linewidth=1.5)
        
        # 添加数值标签
        if i == 0:
            y_pos = values[i] / 2
        else:
            y_pos = cumulative[i] + values[i] / 2
        
        ax_b.text(i, y_pos, f'{values[i]:.0f}万元', ha='center', va='center', 
                fontweight='bold', color='white', fontsize=11)
    
    # 总收益柱
    ax_b.bar(len(improvement_sources)-1, values[-1], color=scenario_colors[1], 
           alpha=0.8, edgecolor='white', linewidth=1.5)
    ax_b.text(len(improvement_sources)-1, values[-1]/2, f'{values[-1]:.0f}万元', 
            ha='center', va='center', fontweight='bold', color='white', fontsize=11)
    
    # 连接线显示累积效果
    for i in range(len(improvement_sources)-2):
        ax_b.plot([i+0.4, i+0.6], [cumulative[i+1], cumulative[i+1]], 
                'k--', alpha=0.5, linewidth=1)
    
    ax_b.set_ylabel('收益 (万元)', fontsize=14)
    ax_b.set_title('图5.6b 收益提升来源分解分析', fontsize=15, fontweight='bold', pad=20)
    ax_b.set_xticks(range(len(improvement_sources)))
    ax_b.set_xticklabels(improvement_sources, fontsize=12)
    ax_b.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('图5.6b_收益提升来源分解分析.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # ============ 子图5.6c：成本收入结构对比 ============
    fig_c, ax_c = plt.subplots(1, 1, figsize=(10, 6))
    
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
    # 顶部留白，避免标注与图框/网格线重叠
    ymax_c = max(scenario1_values + scenario2_values)
    ax_c.set_ylim(0, ymax_c * 1.28)
    
    bars1 = ax_c.bar(x_pos - width/2, scenario1_values, width, label='情景一：超产滞销',
                   color=scenario_colors[0], alpha=0.8, edgecolor='white', linewidth=1.5)
    bars2 = ax_c.bar(x_pos + width/2, scenario2_values, width, label='情景二：50%折价销售',
                   color=scenario_colors[1], alpha=0.8, edgecolor='white', linewidth=1.5)
    
    # 添加数值标签和提升标注
    label_offset_c = max(ymax_c * 0.025, 60)
    for i, (bar1, bar2, val1, val2) in enumerate(zip(bars1, bars2, scenario1_values, scenario2_values)):
        # 情景一数值
        txt_c1 = ax_c.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + label_offset_c,
                f'{val1:.0f}万元', ha='center', va='bottom', fontweight='bold', fontsize=10, zorder=5)
        try:
            txt_c1.set_path_effects([patheffects.withStroke(linewidth=2, foreground='white')])
        except Exception:
            pass
        
        # 情景二数值
        txt_c2 = ax_c.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + label_offset_c,
                f'{val2:.0f}万元', ha='center', va='bottom', fontweight='bold', fontsize=10, zorder=5)
        try:
            txt_c2.set_path_effects([patheffects.withStroke(linewidth=2, foreground='white')])
        except Exception:
            pass
        
        # 提升标注（除了成本）
        if i != 1:  # 成本相同，不标注提升
            improvement = val2 - val1
            improvement_pct = improvement / val1 * 100
            y_top = max(val1, val2) + ymax_c * 0.06
            ann = ax_c.annotate(f'+{improvement:.0f}万元\n(+{improvement_pct:.1f}%)', 
                        xy=(i, y_top), xytext=(0, 0), textcoords='offset points',
                        ha='center', va='bottom', clip_on=False,
                        fontweight='bold', color=improvement_colors[0], fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9), zorder=6)
            try:
                ann.set_path_effects([patheffects.withStroke(linewidth=2, foreground='white')])
            except Exception:
                pass
    
    ax_c.set_ylabel('金额 (万元)', fontsize=14)
    ax_c.set_title('图5.6c 7年总成本收入结构对比', fontsize=15, fontweight='bold', pad=20)
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(categories, fontsize=12)
    # 将图例移到图外，避免与标注冲突
    ax_c.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize=12)
    ax_c.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('图5.6c_成本收入结构对比.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # ============ 子图5.6d：不同作物类型收益贡献分析 ============
    fig_d, ax_d = plt.subplots(1, 1, figsize=(12, 6))
    
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
    imp_max = max(improvements) if improvements else 0
    
    # 创建双轴图
    ax_d_twin = ax_d.twinx()
    
    x_pos = np.arange(len(crop_types))
    width = 0.25
    
    # 收益贡献柱状图
    bars1 = ax_d.bar(x_pos - width, revenue_contribution_s1, width, label='情景一收益',
                    color=scenario_colors[0], alpha=0.8, edgecolor='white', linewidth=1)
    bars2 = ax_d.bar(x_pos, revenue_contribution_s2, width, label='情景二收益',
                    color=scenario_colors[1], alpha=0.8, edgecolor='white', linewidth=1)
    
    # 收益提升折线图
    line = ax_d_twin.plot(x_pos, improvements, 'o-', color=improvement_colors[0], 
                        linewidth=3, markersize=8, label='收益提升', 
                        markerfacecolor='white', markeredgecolor=improvement_colors[0], 
                        markeredgewidth=2)
    # 顶部留白，避免标注与图框/网格重叠
    try:
        ax_d_twin.set_ylim(0, imp_max * 1.35 if imp_max > 0 else 1)
    except Exception:
        pass
    
    # 添加数值标签
    label_offset_d = max((max(revenue_contribution_s1 + revenue_contribution_s2) if (revenue_contribution_s1 or revenue_contribution_s2) else 0) * 0.02, 50)
    for i, (imp, s1, s2) in enumerate(zip(improvements, revenue_contribution_s1, revenue_contribution_s2)):
        if imp > 0:
            txt_imp = ax_d_twin.annotate(f'+{imp:.0f}万元', xy=(i, imp), xytext=(0, 14),
                                textcoords='offset points', ha='center', va='bottom',
                                fontweight='bold', color=improvement_colors[0], fontsize=10, zorder=6, clip_on=False)
            try:
                txt_imp.set_path_effects([patheffects.withStroke(linewidth=2, foreground='white')])
            except Exception:
                pass
        
        # 柱状图数值标签
        txt_d1 = ax_d.text(i - width, s1 + label_offset_d, f'{s1:.0f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=9, zorder=5)
        txt_d2 = ax_d.text(i, s2 + label_offset_d, f'{s2:.0f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=9, zorder=5)
        try:
            txt_d1.set_path_effects([patheffects.withStroke(linewidth=2, foreground='white')])
            txt_d2.set_path_effects([patheffects.withStroke(linewidth=2, foreground='white')])
        except Exception:
            pass
    
    ax_d.set_xlabel('作物类型', fontsize=14)
    ax_d.set_ylabel('收益贡献 (万元)', fontsize=14)
    ax_d_twin.set_ylabel('收益提升 (万元)', fontsize=14, color=improvement_colors[0])
    ax_d.set_title('图5.6d 不同作物类型收益贡献分析', fontsize=15, fontweight='bold', pad=20)
    ax_d.set_xticks(x_pos)
    ax_d.set_xticklabels(crop_types, fontsize=12)
    
    # 合并图例
    lines1, labels1 = ax_d.get_legend_handles_labels()
    lines2, labels2 = ax_d_twin.get_legend_handles_labels()
    # 将图例移到图外，避免与标注冲突
    ax_d.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize=11)
    
    ax_d.grid(True, alpha=0.3)
    
    # 添加关键指标文本框
    total_improvement = sum(improvements)
    improvement_rate = metrics['improvement_rate']
    key_metrics = ("关键指标:\n"
                  f"• 总收益提升: {total_improvement:.0f}万元\n"
                  f"• 提升率: {improvement_rate:.1f}%\n"
                  f"• 主要贡献: 蔬菜类+食用菌")
    
    ax_d.text(0.98, 0.95, key_metrics, transform=ax_d.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=improvement_colors[2], alpha=0.8),
            fontsize=10, ha='right', va='top', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('图5.6d_作物类型收益贡献分析.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("="*60)
    print("✅ 图5.6分割版本生成完成 (基于真实数据)")
    print("="*60)
    print("📁 生成的文件:")
    print("   - 图5.6a_年度收益变化趋势对比.png")
    print("   - 图5.6b_收益提升来源分解分析.png") 
    print("   - 图5.6c_成本收入结构对比.png")
    print("   - 图5.6d_作物类型收益贡献分析.png")
    print()
    print("🎯 关键修正内容:")
    print(f"   - 收益提升率: 从模拟的8-13%修正为真实的{improvement_rate:.1f}%")
    print(f"   - 绝对提升: 从约100万元修正为{metrics['absolute_improvement']:.0f}万元")
    print(f"   - 年均增益: 从约154万元修正为{metrics['absolute_improvement']/7:.0f}万元")
    print("   - 所有数据均基于实际程序运行结果")
    print("="*60)

# 运行函数生成分割图像
if __name__ == "__main__":
    create_separate_revenue_charts()
