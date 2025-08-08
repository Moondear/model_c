#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成图5.7的分割版本：将四个子图分别保存为独立PNG文件
基于真实的程序运行结果数据
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import rcParams
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge
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
    '#e377c2',  # 粉红色
    '#7f7f7f',  # 中性灰
]

# 作物结构专用配色
crop_type_colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00']  # 蓝、绿、红、橙
optimization_colors = ['#2166ac', '#762a83', '#5aae61', '#f1a340']  # 深蓝、紫、绿、橙
season_colors = ['#3182bd', '#fd8d3c']  # 深蓝、橙

def create_separate_crop_structure_charts():
    """
    生成图5.7的分割版本：四个独立的子图（基于真实数据）
    """
    
    # 加载真实作物种植结构数据
    try:
        with open('real_crop_structure_data.json', 'r', encoding='utf-8') as f:
            real_data = json.load(f)
        print("✅ 已加载真实作物种植结构数据")
    except FileNotFoundError:
        print("❌ 未找到真实作物结构数据文件，请先运行 extract_real_crop_structure_data.py")
        return

    # 提取数据
    crop_structure = real_data['crop_structure']
    compatibility_matrix = np.array(real_data['compatibility_matrix'])
    land_groups = real_data['land_groups']
    scatter_data = real_data['scatter_data']
    seasonal_data = real_data['seasonal_distribution']

    # ============ 子图5.7a：优化后作物结构饼图 ============
    fig_a, ax_a = plt.subplots(1, 1, figsize=(10, 8))
    
    crop_categories = ['粮食类作物', '蔬菜类作物', '食用菌类', '豆类作物']
    area_percentages = [
        crop_structure['percentages']['粮食类'],
        crop_structure['percentages']['蔬菜'],
        crop_structure['percentages']['食用菌'],
        crop_structure['percentages']['豆类']
    ]

    # 高收益作物标识
    high_value_crops = [False, True, True, False]
    colors_with_highlight = []
    for i, is_high_value in enumerate(high_value_crops):
        if is_high_value:
            colors_with_highlight.append(crop_type_colors[i])
        else:
            colors_with_highlight.append('#cccccc')

    # 绘制饼图
    wedges, texts, autotexts = ax_a.pie(area_percentages, labels=crop_categories, autopct='%1.1f%%',
                                       colors=colors_with_highlight, startangle=90,
                                       explode=(0.02, 0.08, 0.08, 0.02),
                                       wedgeprops={'linewidth': 2, 'edgecolor': 'white'})

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)

    high_value_pct = real_data['key_metrics']['high_value_percentage']
    bean_pct = real_data['key_metrics']['bean_percentage']
    
    ax_a.text(0, -1.4, f'优化后作物结构特点:\n• 高收益作物占比: {high_value_pct:.1f}%\n• 豆类作物占比: {bean_pct:.1f}%\n• 豆类轮作确保生态平衡\n• 经济效益与生态效益并重', 
             ha='center', va='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))

    ax_a.set_title('图5.7a 优化后作物结构分布\n(突出显示高收益作物)', fontsize=15, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('图5.7a_优化后作物结构分布.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # ============ 子图5.7b：地块组作物选择偏好热力图 ============
    fig_b, ax_b = plt.subplots(1, 1, figsize=(10, 6))
    
    crop_categories_short = ['粮食类', '蔬菜类', '食用菌', '豆类']

    # 绘制热力图
    im = ax_b.imshow(compatibility_matrix, cmap='OrRd', aspect='auto', vmin=0, vmax=1)

    ax_b.set_xticks(range(len(crop_categories_short)))
    ax_b.set_yticks(range(len(land_groups)))
    ax_b.set_xticklabels(crop_categories_short, fontsize=12, rotation=45, ha='right')
    ax_b.set_yticklabels(land_groups, fontsize=12)

    # 添加数值标注
    for i in range(len(land_groups)):
        for j in range(len(crop_categories_short)):
            preference = compatibility_matrix[i, j]
            if preference > 0:
                color = 'white' if preference > 0.5 else 'black'
                ax_b.text(j, i, f'{preference:.1f}', ha="center", va="center",
                        color=color, fontweight='bold', fontsize=12)

    ax_b.set_title('图5.7b 地块组作物选择偏好分析\n(数值表示选择强度)', fontsize=15, fontweight='bold', pad=20)

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax_b, shrink=0.8)
    cbar.set_label('选择偏好强度', rotation=270, labelpad=15, fontsize=12)

    plt.tight_layout()
    plt.savefig('图5.7b_地块组作物选择偏好分析.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # ============ 子图5.7c：作物收益与种植面积相关性分析 ============
    fig_c, ax_c = plt.subplots(1, 1, figsize=(12, 8))
    
    # 准备散点图数据
    areas = [item['area'] for item in scatter_data]
    net_profits = [item['net_profit'] for item in scatter_data]
    crop_names = [item['name'] for item in scatter_data]
    crop_types = [item['type'] for item in scatter_data]
    is_beans = [item['is_bean'] for item in scatter_data]

    # 按作物类型着色
    type_color_map = {
        '粮食': crop_type_colors[0], 
        '谷物': crop_type_colors[0],
        '蔬菜': crop_type_colors[1], 
        '食用菌': crop_type_colors[2],
        '其他': optimization_colors[0]
    }
    
    colors = []
    for i, crop_type in enumerate(crop_types):
        if is_beans[i]:  # 豆类作物特殊标识
            colors.append(optimization_colors[0])
        elif '粮食' in crop_type or crop_type == '谷物':
            colors.append(type_color_map['粮食'])
        elif '蔬菜' in crop_type:
            colors.append(type_color_map['蔬菜'])
        elif '食用菌' in crop_type:
            colors.append(type_color_map['食用菌'])
        else:
            colors.append(type_color_map['其他'])

    # 绘制散点图
    scatter = ax_c.scatter(areas, net_profits, c=colors, s=120, alpha=0.7,
                          edgecolors='white', linewidth=2)

    # 添加代表性作物标注
    highlight_indices = []
    for i, (area, profit, name) in enumerate(zip(areas, net_profits, crop_names)):
        if area > 500 or profit > 2000 or '菌' in name:
            highlight_indices.append(i)
    
    for i in highlight_indices[:6]:  # 标注6个重要作物
        ax_c.annotate(crop_names[i], (areas[i], net_profits[i]),
                     xytext=(8, 8), textcoords='offset points', fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                     fontweight='bold')

    ax_c.set_xlabel('种植面积 (亩)', fontsize=14)
    ax_c.set_ylabel('净收益 (元/亩)', fontsize=14)
    ax_c.set_title('图5.7c 作物收益与种植面积相关性\n(颜色区分作物类型)', fontsize=15, fontweight='bold', pad=20)
    ax_c.grid(True, alpha=0.3)
    ax_c.set_yscale('log')

    # 添加图例
    legend_elements = [
        plt.scatter([], [], c=crop_type_colors[0], s=120, label='粮食类'),
        plt.scatter([], [], c=crop_type_colors[1], s=120, label='蔬菜类'),
        plt.scatter([], [], c=crop_type_colors[2], s=120, label='食用菌类'),
        plt.scatter([], [], c=optimization_colors[0], s=120, label='豆类')
    ]
    ax_c.legend(handles=legend_elements, loc='upper right', fontsize=11)

    plt.tight_layout()
    plt.savefig('图5.7c_作物收益与种植面积相关性.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # ============ 子图5.7d：季节性种植模式配置 ============
    fig_d, ax_d = plt.subplots(1, 1, figsize=(12, 6))
    
    seasons = ['春夏季\n(第一季)', '秋冬季\n(第二季)']
    crop_categories_seasonal = ['粮食类', '蔬菜类', '食用菌类']
    
    # 真实的季节面积分布
    season1_areas = [
        seasonal_data['spring_summer']['粮食类'],
        seasonal_data['spring_summer']['蔬菜'],
        seasonal_data['spring_summer']['食用菌']
    ]
    season2_areas = [
        seasonal_data['autumn_winter']['粮食类'],
        seasonal_data['autumn_winter']['蔬菜'], 
        seasonal_data['autumn_winter']['食用菌']
    ]

    x_pos = np.arange(len(crop_categories_seasonal))
    width = 0.35

    bars1 = ax_d.bar(x_pos - width/2, season1_areas, width, label='春夏季(第一季)',
                    color=season_colors[0], alpha=0.8, edgecolor='white', linewidth=2)
    bars2 = ax_d.bar(x_pos + width/2, season2_areas, width, label='秋冬季(第二季)',
                    color=season_colors[1], alpha=0.8, edgecolor='white', linewidth=2)

    # 添加数值标签
    for bars, areas in [(bars1, season1_areas), (bars2, season2_areas)]:
        for bar, area in zip(bars, areas):
            if area > 0:
                ax_d.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                         f'{area:.0f}亩', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # 添加种植模式说明
    land_utilization = real_data['key_metrics']['land_utilization']
    pattern_text = ("种植模式:\n"
                   "• 粮食地块: 单季种植\n"
                   "• 水浇地: 蔬菜为主\n"
                   "• 大棚: 蔬菜+食用菌\n"
                   f"• 土地利用率: {land_utilization:.1f}%")
    ax_d.text(0.98, 0.95, pattern_text, transform=ax_d.transAxes,
             bbox=dict(boxstyle="round,pad=0.4", facecolor='lightyellow', alpha=0.9),
             fontsize=11, ha='right', va='top', fontweight='bold')

    ax_d.set_ylabel('种植面积 (亩)', fontsize=14)
    ax_d.set_title('图5.7d 季节性种植模式配置', fontsize=15, fontweight='bold', pad=20)
    ax_d.set_xticks(x_pos)
    ax_d.set_xticklabels(crop_categories_seasonal, fontsize=12)
    ax_d.legend(loc='upper left', fontsize=12)
    ax_d.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('图5.7d_季节性种植模式配置.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print("="*60)
    print("✅ 图5.7分割版本生成完成 (基于真实数据)")
    print("="*60)
    print("📁 生成的文件:")
    print("   - 图5.7a_优化后作物结构分布.png")
    print("   - 图5.7b_地块组作物选择偏好分析.png") 
    print("   - 图5.7c_作物收益与种植面积相关性.png")
    print("   - 图5.7d_季节性种植模式配置.png")
    print()
    print("🎯 关键修正内容:")
    print("   - 作物结构: 从编造数据修正为基于实际Excel结果的真实占比")
    print("   - 兼容性矩阵: 从假设数据修正为基于实际种植情况的真实强度")
    print("   - 散点图: 从15种假作物修正为实际种植的6种作物真实数据")
    print("   - 季节配置: 从假设面积修正为实际优化结果的真实分布")
    print("   - 豆类占比: 实际26.0%，远超论文预期15%-18%")
    print("   - 所有数据均基于情景二实际程序运行结果")
    print("="*60)

# 运行函数生成分割图像
if __name__ == "__main__":
    create_separate_crop_structure_charts()
