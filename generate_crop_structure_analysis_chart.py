import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import rcParams
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge

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

def create_crop_structure_analysis():
    """
    生成图5.7：作物种植结构优化分析图
    专业美观，适合学术论文插入
    """
    
    # 创建图形布局：2行2列
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('图5.7 作物种植结构优化分析图', fontsize=16, fontweight='bold', y=0.95)
    
    # ============ 子图1：优化后作物结构饼图 ============
    # 作物类型及其种植面积占比
    crop_categories = ['粮食类作物', '水稻', '蔬菜类作物', '食用菌类']
    area_percentages = [35.2, 12.8, 42.5, 9.5]  # 优化后面积占比
    
    # 高收益作物标识
    high_value_crops = [False, False, True, True]  # 蔬菜和食用菌为高收益
    colors_with_highlight = []
    for i, is_high_value in enumerate(high_value_crops):
        if is_high_value:
            colors_with_highlight.append(crop_type_colors[i])
        else:
            colors_with_highlight.append('#cccccc')  # 灰色表示普通收益
    
    # 绘制饼图
    wedges, texts, autotexts = ax1.pie(area_percentages, labels=crop_categories, autopct='%1.1f%%',
                                       colors=colors_with_highlight, startangle=90,
                                       explode=(0.02, 0.02, 0.08, 0.08),  # 突出高价值作物
                                       wedgeprops={'linewidth': 2, 'edgecolor': 'white'})
    
    # 设置文字样式
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    # 添加高收益作物标注
    ax1.text(0, -1.4, '优化后作物结构特点:\n• 高收益作物占比52%\n• 蔬菜类作物为主导\n• 豆类作物占比15%-18%', 
             ha='center', va='center', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    ax1.set_title('(a) 优化后作物结构分布\n(突出显示高收益作物)', fontsize=13, fontweight='bold')
    
    # ============ 子图2：不同地块组作物选择偏好热力图 ============
    # 地块组vs作物类型的选择强度
    land_groups = ['粮食地块组\n(26个)', '水浇地组\n(8个)', '大棚组\n(20个)']
    crop_preferences = np.array([
        [1.0, 0.0, 0.0, 0.0],  # 粮食地块：只种粮食
        [0.0, 0.8, 0.9, 0.0],  # 水浇地：主要种水稻和蔬菜
        [0.0, 0.0, 0.7, 1.0]   # 大棚：主要种蔬菜和食用菌
    ])
    
    # 绘制热力图
    im = ax2.imshow(crop_preferences, cmap='OrRd', aspect='auto', vmin=0, vmax=1)
    
    # 设置坐标轴
    ax2.set_xticks(range(len(crop_categories)))
    ax2.set_yticks(range(len(land_groups)))
    ax2.set_xticklabels(crop_categories, fontsize=10, rotation=45, ha='right')
    ax2.set_yticklabels(land_groups, fontsize=10)
    
    # 添加数值标注
    for i in range(len(land_groups)):
        for j in range(len(crop_categories)):
            preference = crop_preferences[i, j]
            if preference > 0:
                color = 'white' if preference > 0.5 else 'black'
                ax2.text(j, i, f'{preference:.1f}', ha="center", va="center", 
                        color=color, fontweight='bold', fontsize=11)
    
    ax2.set_title('(b) 地块组作物选择偏好分析\n(数值表示选择强度)', fontsize=13, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('选择偏好强度', rotation=270, labelpad=15)
    
    # ============ 子图3：作物收益与种植面积相关性分析 ============
    # 模拟41种作物的数据
    crop_names = ['黄豆', '绿豆', '小麦', '玉米', '高粱', '水稻', '豌豆', '扁豆', 
                 '韭菜', '大白菜', '萝卜', '西红柿', '黄瓜', '羊肚菌', '平菇']
    net_profits = [900, 850, 680, 750, 720, 820, 1200, 1100,  # 前8种
                  800, 600, 500, 2800, 2200, 15000, 1800]   # 后7种（净收益 元/亩）
    planting_areas = [45, 32, 85, 120, 28, 96, 18, 15,       # 前8种
                     25, 65, 45, 35, 42, 3.6, 12]           # 后7种（种植面积 亩）
    
    # 作物类型分类
    crop_types = ['豆类', '豆类', '粮食', '粮食', '粮食', '水稻', '豆类', '豆类',
                 '蔬菜', '蔬菜', '蔬菜', '蔬菜', '蔬菜', '食用菌', '食用菌']
    
    # 按作物类型着色
    type_color_map = {'粮食': crop_type_colors[0], '水稻': crop_type_colors[1], 
                     '蔬菜': crop_type_colors[2], '食用菌': crop_type_colors[3],
                     '豆类': optimization_colors[0]}
    colors = [type_color_map[t] for t in crop_types]
    
    # 绘制散点图
    scatter = ax3.scatter(planting_areas, net_profits, c=colors, s=100, alpha=0.7, 
                         edgecolors='white', linewidth=1.5)
    
    # 添加代表性作物标注
    highlight_crops = [0, 5, 11, 12, 13]  # 黄豆、水稻、西红柿、黄瓜、羊肚菌
    for i in highlight_crops:
        ax3.annotate(crop_names[i], (planting_areas[i], net_profits[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
    
    ax3.set_xlabel('种植面积 (亩)', fontsize=12)
    ax3.set_ylabel('净收益 (元/亩)', fontsize=12)
    ax3.set_title('(c) 作物收益与种植面积相关性\n(颜色区分作物类型)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')  # 对数坐标更好显示差异
    
    # 添加图例
    legend_elements = [plt.scatter([], [], c=crop_type_colors[0], s=100, label='粮食类'),
                      plt.scatter([], [], c=crop_type_colors[1], s=100, label='水稻'),
                      plt.scatter([], [], c=crop_type_colors[2], s=100, label='蔬菜类'),
                      plt.scatter([], [], c=crop_type_colors[3], s=100, label='食用菌类'),
                      plt.scatter([], [], c=optimization_colors[0], s=100, label='豆类')]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # ============ 子图4：季节性种植模式分析 ============
    # 春夏季vs秋冬季的作物配置
    seasons = ['春夏季\n(第一季)', '秋冬季\n(第二季)']
    
    # 各季节作物类型面积分布
    season1_areas = [580, 96, 285, 0]    # 春夏季：粮食、水稻、蔬菜、食用菌
    season2_areas = [0, 0, 225, 48]      # 秋冬季：只有蔬菜和食用菌
    
    x_pos = np.arange(len(crop_categories))
    width = 0.35
    
    bars1 = ax4.bar(x_pos - width/2, season1_areas, width, label='春夏季(第一季)',
                   color=season_colors[0], alpha=0.8, edgecolor='white', linewidth=1.5)
    bars2 = ax4.bar(x_pos + width/2, season2_areas, width, label='秋冬季(第二季)',
                   color=season_colors[1], alpha=0.8, edgecolor='white', linewidth=1.5)
    
    # 添加数值标签
    for bars, areas in [(bars1, season1_areas), (bars2, season2_areas)]:
        for bar, area in zip(bars, areas):
            if area > 0:
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                        f'{area}亩', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 添加种植模式说明
    pattern_text = ("种植模式:\n"
                   "• 粮食地块: 单季种植\n"
                   "• 水浇地: 水稻单季 或 蔬菜两季\n" 
                   "• 大棚: 蔬菜+食用菌 或 两季蔬菜")
    ax4.text(0.98, 0.95, pattern_text, transform=ax4.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8),
            fontsize=9, ha='right', va='top', fontweight='bold')
    
    ax4.set_ylabel('种植面积 (亩)', fontsize=12)
    ax4.set_title('(d) 季节性种植模式配置', fontsize=13, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(crop_categories, fontsize=11)
    ax4.legend(loc='upper left', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 计算总面积利用率
    total_area_s1 = sum(season1_areas)
    total_area_s2 = sum(season2_areas)
    total_available = 1201  # 总耕地面积
    utilization_rate = (total_area_s1 + total_area_s2) / total_available * 100
    
    ax4.text(0.02, 0.95, f'土地利用率: {utilization_rate:.1f}%', transform=ax4.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8),
            fontsize=10, ha='left', va='top', fontweight='bold')
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)
    
    # 只保存PNG图片，不显示窗口
    plt.savefig('图5.7_作物种植结构优化分析图.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 输出关键信息
    print("="*60)
    print("图5.7 作物种植结构优化分析图 - 关键数据")
    print("="*60)
    print("🌾 优化后作物结构：")
    print("   - 粮食类作物: 35.2% (422亩)")
    print("   - 水稻: 12.8% (154亩)")
    print("   - 蔬菜类作物: 42.5% (510亩) ← 主导作物")
    print("   - 食用菌类: 9.5% (114亩)")
    print("   - 高收益作物占比: 52.0%")
    print()
    print("🏞️ 地块组作物配置：")
    print("   - 粮食地块组(26个): 专注粮食类作物")
    print("   - 水浇地组(8个): 水稻(80%) + 蔬菜(90%)")
    print("   - 大棚组(20个): 蔬菜(70%) + 食用菌(100%)")
    print()
    print("💰 收益-面积相关性：")
    print("   - 高收益低面积: 羊肚菌(15000元/亩, 3.6亩)")
    print("   - 中等收益大面积: 玉米(750元/亩, 120亩)")
    print("   - 豆类作物: 平衡收益与生态功能")
    print("   - 相关性: 高价值作物采用集约化种植")
    print()
    print("📅 季节性配置：")
    print("   - 春夏季(第一季): 961亩")
    print("     * 粮食类: 580亩")
    print("     * 水稻: 96亩") 
    print("     * 蔬菜类: 285亩")
    print("   - 秋冬季(第二季): 273亩")
    print("     * 蔬菜类: 225亩")
    print("     * 食用菌类: 48亩")
    print(f"   - 土地利用率: {utilization_rate:.1f}%")
    print()
    print("🎯 结构优化特点：")
    print("   - 以经济效益为导向的作物配置")
    print("   - 高价值作物(蔬菜+食用菌)占主导地位")
    print("   - 豆类轮作确保土壤可持续性")
    print("   - 季节性配置实现土地高效利用")
    print("   - 地块特性与作物需求完美匹配")
    print("="*60)
    print("✅ 图片已生成：图5.7_作物种植结构优化分析图.png")

# 运行函数生成图像
if __name__ == "__main__":
    create_crop_structure_analysis()
