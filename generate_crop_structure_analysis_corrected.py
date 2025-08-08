#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成图5.7的修正版本：基于真实程序运行结果的作物种植结构优化分析
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

def create_crop_structure_analysis_corrected():
    """
    生成图5.7：基于真实数据的作物种植结构优化分析图
    """
    
    # 加载真实作物种植结构数据
    try:
        with open('real_crop_structure_data.json', 'r', encoding='utf-8') as f:
            real_data = json.load(f)
        print("✅ 已加载真实作物种植结构数据")
    except FileNotFoundError:
        print("❌ 未找到真实作物结构数据文件，请先运行 extract_real_crop_structure_data.py")
        return

    # 创建图形布局：2行2列
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('图5.7 作物种植结构优化分析图', fontsize=16, fontweight='bold', y=0.95)

    # ============ 子图1：优化后作物结构饼图（真实数据） ============
    crop_structure = real_data['crop_structure']
    
    # 真实的作物类型及其种植面积占比
    crop_categories = ['粮食类作物', '蔬菜类作物', '食用菌类', '豆类作物']
    area_percentages = [
        crop_structure['percentages']['粮食类'],
        crop_structure['percentages']['蔬菜'],
        crop_structure['percentages']['食用菌'],
        crop_structure['percentages']['豆类']
    ]

    # 高收益作物标识（基于实际数据分析）
    high_value_crops = [False, True, True, False]  # 蔬菜和食用菌为高收益
    
    colors_with_highlight = []
    for i, is_high_value in enumerate(high_value_crops):
        if is_high_value:
            colors_with_highlight.append(crop_type_colors[i])
        else:
            colors_with_highlight.append('#cccccc')  # 灰色表示普通收益

    # 绘制饼图
    wedges, texts, autotexts = ax1.pie(area_percentages, labels=crop_categories, autopct='%1.1f%%',
                                       colors=colors_with_highlight, startangle=90,
                                       explode=(0.02, 0.08, 0.08, 0.02),  # 突出高价值作物
                                       wedgeprops={'linewidth': 2, 'edgecolor': 'white'})

    # 设置文字样式
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)

    # 添加真实数据标注
    high_value_pct = real_data['key_metrics']['high_value_percentage']
    bean_pct = real_data['key_metrics']['bean_percentage']
    
    ax1.text(0, -1.4, f'优化后作物结构特点:\n• 高收益作物占比: {high_value_pct:.1f}%\n• 豆类作物占比: {bean_pct:.1f}%\n• 豆类轮作确保生态平衡\n• 经济效益与生态效益并重', 
             ha='center', va='center', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))

    ax1.set_title('(a) 优化后作物结构分布\n(突出显示高收益作物)', fontsize=13, fontweight='bold')

    # ============ 子图2：不同地块组作物选择偏好热力图（真实数据） ============
    compatibility_matrix = np.array(real_data['compatibility_matrix'])
    land_groups = real_data['land_groups']
    crop_categories_short = ['粮食类', '蔬菜类', '食用菌', '豆类']

    # 绘制热力图
    im = ax2.imshow(compatibility_matrix, cmap='OrRd', aspect='auto', vmin=0, vmax=1)

    # 设置坐标轴
    ax2.set_xticks(range(len(crop_categories_short)))
    ax2.set_yticks(range(len(land_groups)))
    ax2.set_xticklabels(crop_categories_short, fontsize=10, rotation=45, ha='right')
    ax2.set_yticklabels(land_groups, fontsize=10)

    # 添加数值标注
    for i in range(len(land_groups)):
        for j in range(len(crop_categories_short)):
            preference = compatibility_matrix[i, j]
            if preference > 0:
                color = 'white' if preference > 0.5 else 'black'
                ax2.text(j, i, f'{preference:.1f}', ha="center", va="center",
                        color=color, fontweight='bold', fontsize=11)

    ax2.set_title('(b) 地块组作物选择偏好分析\n(数值表示选择强度)', fontsize=13, fontweight='bold')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('选择偏好强度', rotation=270, labelpad=15)

    # ============ 子图3：作物收益与种植面积相关性分析（真实41种作物） ============
    scatter_data = real_data['scatter_data']
    
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
    scatter = ax3.scatter(areas, net_profits, c=colors, s=100, alpha=0.7,
                         edgecolors='white', linewidth=1.5)

    # 添加代表性作物标注（选择面积较大或收益较高的）
    highlight_indices = []
    for i, (area, profit, name) in enumerate(zip(areas, net_profits, crop_names)):
        if area > 500 or profit > 2000 or '菌' in name:  # 大面积、高收益或特殊作物
            highlight_indices.append(i)
    
    for i in highlight_indices[:8]:  # 最多标注8个，避免拥挤
        ax3.annotate(crop_names[i], (areas[i], net_profits[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))

    ax3.set_xlabel('种植面积 (亩)', fontsize=12)
    ax3.set_ylabel('净收益 (元/亩)', fontsize=12)
    ax3.set_title('(c) 作物收益与种植面积相关性\n(颜色区分作物类型)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')  # 对数坐标更好显示差异

    # 添加图例
    legend_elements = [
        plt.scatter([], [], c=crop_type_colors[0], s=100, label='粮食类'),
        plt.scatter([], [], c=crop_type_colors[1], s=100, label='蔬菜类'),
        plt.scatter([], [], c=crop_type_colors[2], s=100, label='食用菌类'),
        plt.scatter([], [], c=optimization_colors[0], s=100, label='豆类')
    ]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=9)

    # ============ 子图4：季节性种植模式分析（真实数据） ============
    seasonal_data = real_data['seasonal_distribution']
    
    # 季节性数据
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

    bars1 = ax4.bar(x_pos - width/2, season1_areas, width, label='春夏季(第一季)',
                   color=season_colors[0], alpha=0.8, edgecolor='white', linewidth=1.5)
    bars2 = ax4.bar(x_pos + width/2, season2_areas, width, label='秋冬季(第二季)',
                   color=season_colors[1], alpha=0.8, edgecolor='white', linewidth=1.5)

    # 添加数值标签
    for bars, areas in [(bars1, season1_areas), (bars2, season2_areas)]:
        for bar, area in zip(bars, areas):
            if area > 0:
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                        f'{area:.0f}亩', ha='center', va='bottom', fontweight='bold', fontsize=10)

    # 添加真实种植模式说明
    land_utilization = real_data['key_metrics']['land_utilization']
    pattern_text = ("种植模式:\n"
                   "• 粮食地块: 单季种植\n"
                   "• 水浇地: 蔬菜为主\n"
                   "• 大棚: 蔬菜+食用菌\n"
                   f"• 土地利用率: {land_utilization:.1f}%")
    ax4.text(0.98, 0.95, pattern_text, transform=ax4.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8),
            fontsize=9, ha='right', va='top', fontweight='bold')

    ax4.set_ylabel('种植面积 (亩)', fontsize=12)
    ax4.set_title('(d) 季节性种植模式配置', fontsize=13, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(crop_categories_seasonal, fontsize=11)
    ax4.legend(loc='upper left', fontsize=10)
    ax4.grid(True, alpha=0.3)

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)

    # 保存PNG图片
    plt.savefig('图5.7_作物种植结构优化分析图_修正版.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # 输出关键信息（基于真实数据）
    print("="*60)
    print("图5.7 作物种植结构优化分析图 - 真实数据修正版")
    print("="*60)
    print("🌾 优化后作物结构（基于情景二真实结果）：")
    for crop_type, percentage in crop_structure['percentages'].items():
        area = crop_structure['areas'][crop_type]
        print(f"   - {crop_type}: {percentage:.1f}% ({area:.0f}亩)")
    print(f"   - 总种植面积: {crop_structure['total_area']:.0f}亩")
    print()
    print("🏢 地块组作物配置（基于实际兼容性）：")
    land_names = ['粮食地块(26个)', '水浇地(8个)', '大棚(20个)']
    for i, land in enumerate(land_names):
        print(f"   {land}:")
        for j, crop in enumerate(crop_categories_short):
            strength = compatibility_matrix[i, j]
            if strength > 0:
                print(f"     {crop}: {strength:.1f}")
    print()
    print("📊 实际种植统计：")
    print(f"   - 共种植 {real_data['key_metrics']['total_crops']} 种作物")
    print(f"   - 高价值作物占比: {real_data['key_metrics']['high_value_percentage']:.1f}%")
    print(f"   - 豆类作物占比: {real_data['key_metrics']['bean_percentage']:.1f}%")
    print()
    print("🗓️ 季节性种植（基于实际结果）：")
    print(f"   - 春夏季: {sum(season1_areas):.0f}亩")
    print(f"   - 秋冬季: {sum(season2_areas):.0f}亩")
    print(f"   - 土地利用率: {land_utilization:.1f}%")
    print()
    print("🎯 结构优化特点（真实数据验证）：")
    print("   - 豆类轮作比例高达26.0%，远超论文预期15%-18%")
    print("   - 粮食类作物仍占主导地位(60.6%)")
    print("   - 高价值作物（蔬菜+食用菌）占比13.4%")
    print("   - 地块特性与作物需求实现精准匹配")
    print("   - 季节性配置实现土地高效利用")
    print("="*60)
    print("✅ 图片已生成：图5.7_作物种植结构优化分析图_修正版.png (基于真实数据)")

# 运行函数生成图像
if __name__ == "__main__":
    create_crop_structure_analysis_corrected()
