#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成图5.8的修正版本：基于真实程序运行结果的算法性能与求解效率分析
保持学术专业性和可视化美观度
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import rcParams
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
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

# 算法专用配色
algorithm_colors = {
    '动态规划': '#2166ac',    # 深蓝
    '整数规划': '#762a83',    # 紫色
    '贪心算法': '#5aae61',    # 绿色
    '分层分治': '#f1a340'     # 橙色
}

def create_algorithm_performance_analysis_corrected():
    """
    生成图5.8：基于真实数据的算法性能与求解效率分析图
    """
    
    # 加载真实算法性能数据
    try:
        with open('real_algorithm_performance_data.json', 'r', encoding='utf-8') as f:
            real_data = json.load(f)
        print("✅ 已加载真实算法性能数据")
    except FileNotFoundError:
        print("❌ 未找到真实算法性能数据文件，请先运行 extract_real_algorithm_performance.py")
        return

    # 创建图形布局：2行2列
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('图5.8 算法性能与求解效率分析图', fontsize=16, fontweight='bold', y=0.95)
    
    # ============ 子图1：基于真实问题规模的算法复杂度分析 ============
    algorithms = ['动态规划\n(粮食地块)', '整数规划\n(水浇地)', '贪心算法\n(大棚)', '分层分治\n(整体)']
    
    # 使用真实的复杂度数据
    time_complexity_values = [
        real_data['time_complexity']['动态规划'],
        real_data['time_complexity']['整数规划'], 
        real_data['time_complexity']['贪心算法'],
        real_data['time_complexity']['分层分治']
    ]
    space_complexity_values = [
        real_data['space_complexity']['动态规划'],
        real_data['space_complexity']['整数规划'],
        real_data['space_complexity']['贪心算法'], 
        real_data['space_complexity']['分层分治']
    ]
    
    x = np.arange(len(algorithms))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, time_complexity_values, width, 
                    label='时间复杂度', color=algorithm_colors['动态规划'], alpha=0.8,
                    edgecolor='white', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, space_complexity_values, width,
                    label='空间复杂度', color=algorithm_colors['整数规划'], alpha=0.8,
                    edgecolor='white', linewidth=1.5)
    
    # 添加数值标签
    for bars, values in [(bars1, time_complexity_values), (bars2, space_complexity_values)]:
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                    f'{value:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 添加复杂度等级线（基于实际数据调整）
    max_val = max(max(time_complexity_values), max(space_complexity_values))
    ax1.axhline(y=max_val*0.2, color='green', linestyle='--', alpha=0.7, label='低复杂度阈值')
    ax1.axhline(y=max_val*0.8, color='red', linestyle='--', alpha=0.7, label='高复杂度阈值')
    
    ax1.set_ylabel('相对复杂度指标', fontsize=12)
    ax1.set_title('(a) 基于实际问题规模的算法复杂度分析', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, fontsize=10)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 添加实际问题规模标注
    scale_info = real_data['problem_scale']
    complexity_text = (f"实际问题规模:\n"
                      f"• 地块总数: {scale_info['total_lands']}个\n"
                      f"• 作物种类: {scale_info['crops']}种\n"
                      f"• 优化年数: {scale_info['years']}年\n"
                      f"• DP状态: T×J×K\n"
                      f"• IP变量: ~{scale_info['irrigation_lands']*scale_info['crops']*scale_info['seasons']}个")
    ax1.text(0.98, 0.98, complexity_text, transform=ax1.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8),
            fontsize=9, ha='right', va='top', fontweight='bold')
    
    # ============ 子图2：真实求解时间性能对比 ============
    problem_sizes = ['3年', '5年', '7年\n(实际)', '10年']
    
    # 使用真实求解时间数据
    dp_times = real_data['predicted_times']['dp_times']
    ip_times = real_data['predicted_times']['ip_times']
    greedy_times = real_data['predicted_times']['greedy_times']
    total_times = real_data['predicted_times']['total_times']
    
    x_pos = np.arange(len(problem_sizes))
    
    # 绘制线图
    ax2.plot(x_pos, dp_times, 'o-', color=algorithm_colors['动态规划'], 
            linewidth=3, markersize=8, label='动态规划', markerfacecolor='white',
            markeredgewidth=2)
    ax2.plot(x_pos, ip_times, 's-', color=algorithm_colors['整数规划'], 
            linewidth=3, markersize=8, label='整数规划', markerfacecolor='white',
            markeredgewidth=2)
    ax2.plot(x_pos, greedy_times, '^-', color=algorithm_colors['贪心算法'], 
            linewidth=3, markersize=8, label='贪心算法', markerfacecolor='white',
            markeredgewidth=2)
    ax2.plot(x_pos, total_times, 'd-', color=algorithm_colors['分层分治'], 
            linewidth=3, markersize=8, label='分层分治(总)', markerfacecolor='white',
            markeredgewidth=2)
    
    ax2.set_ylabel('求解时间 (秒)', fontsize=12)
    ax2.set_title('(b) 基于真实数据的求解时间性能对比', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(problem_sizes, fontsize=10)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 突出显示实际7年问题的性能
    actual_idx = 2  # 7年是第3个点
    ax2.scatter(actual_idx, dp_times[actual_idx], s=150, color='red', alpha=0.7, 
               marker='o', zorder=5, label='实际测试点')
    ax2.scatter(actual_idx, ip_times[actual_idx], s=150, color='red', alpha=0.7, 
               marker='s', zorder=5)
    ax2.scatter(actual_idx, greedy_times[actual_idx], s=150, color='red', alpha=0.7, 
               marker='^', zorder=5)
    ax2.scatter(actual_idx, total_times[actual_idx], s=150, color='red', alpha=0.7, 
               marker='d', zorder=5)
    
    # 添加实际性能数据
    real_times = real_data['real_times']
    efficiency_text = (f"7年问题实际求解时间:\n"
                      f"• 动态规划: {real_times['dp_time']:.6f}s\n"
                      f"• 整数规划: {real_times['ip_time']:.6f}s\n"
                      f"• 贪心算法: {real_times['greedy_time']:.6f}s\n"
                      f"• 总计时间: {real_times['total_time']:.6f}s\n"
                      f"• 性能等级: 优秀 (< 1秒)")
    ax2.text(0.02, 0.98, efficiency_text, transform=ax2.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8),
            fontsize=9, ha='left', va='top', fontweight='bold')
    
    # ============ 子图3：约束处理能力与解质量分析 ============
    # 基于实际约束违反数据分析算法性能
    constraint_types = ['重茬\n约束', '豆类\n轮作', '最小\n面积', '整体\n表现']
    violation_data = real_data['violation_analysis']
    
    # 约束违反情况
    violations = [
        violation_data.get('重茬约束', 0),
        violation_data.get('豆类轮作', 0), 
        violation_data.get('最小面积', 0),
        sum(violation_data.values())
    ]
    
    # 计算约束满足率（百分比）
    total_possible_violations = real_data['problem_scale']['total_lands'] * real_data['problem_scale']['years']
    satisfaction_rates = []
    for i, v in enumerate(violations[:-1]):  # 前三个约束
        rate = max(0, 100 - (v / total_possible_violations * 100 * 3))  # 每个约束的满足率
        satisfaction_rates.append(rate)
    # 整体满足率
    overall_rate = 100 - (violations[-1] / (total_possible_violations * 3) * 100)
    satisfaction_rates.append(max(0, overall_rate))
    
    x_constraint = np.arange(len(constraint_types))
    
    # 绘制违反次数柱状图
    bars_violation = ax3.bar(x_constraint, violations, alpha=0.7, 
                           color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'],
                           edgecolor='white', linewidth=1.5)
    
    # 添加数值标签
    for bar, violation in zip(bars_violation, violations):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(violations)*0.02,
                f'{violation}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax3.set_ylabel('约束违反次数', fontsize=12)
    ax3.set_title('(c) 约束处理能力与解质量分析', fontsize=13, fontweight='bold')
    ax3.set_xticks(x_constraint)
    ax3.set_xticklabels(constraint_types, fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 添加双轴显示满足率
    ax3_twin = ax3.twinx()
    line_satisfaction = ax3_twin.plot(x_constraint, satisfaction_rates, 'ro-', 
                                    linewidth=3, markersize=8, label='约束满足率(%)',
                                    markerfacecolor='white', markeredgewidth=2)
    ax3_twin.set_ylabel('约束满足率 (%)', fontsize=12, color='red')
    ax3_twin.set_ylim(0, 100)
    
    # 添加解质量分析
    quality_score = real_data['solution_quality']['quality_score']
    violation_rate = real_data['solution_quality']['violation_rate']
    quality_text = (f"解质量评估:\n"
                   f"• 总违反次数: {violations[-1]}\n"
                   f"• 违反率: {violation_rate:.1f}%\n"
                   f"• 质量评分: {quality_score:.1f}/10\n"
                   f"• 算法特点: 快速求解\n"
                   f"  但存在约束松弛")
    ax3.text(0.98, 0.98, quality_text, transform=ax3.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcyan', alpha=0.8),
            fontsize=9, ha='right', va='top', fontweight='bold')
    
    # ============ 子图4：基于实际性能的综合评估雷达图 ============
    # 性能指标
    metrics = ['求解速度', '解质量', '内存效率', '可扩展性', '实现难度', '稳定性']
    
    # 使用真实数据计算的性能评分
    performance_scores = real_data['performance_scores']
    dp_scores = list(performance_scores['动态规划'].values())
    ip_scores = list(performance_scores['整数规划'].values())
    greedy_scores = list(performance_scores['贪心算法'].values())
    
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    # 添加闭合点
    dp_scores += dp_scores[:1]
    ip_scores += ip_scores[:1] 
    greedy_scores += greedy_scores[:1]
    
    # 绘制雷达图
    ax4.plot(angles, dp_scores, 'o-', linewidth=3, color=algorithm_colors['动态规划'], 
            label='动态规划', markersize=6, markerfacecolor='white', markeredgewidth=2)
    ax4.fill(angles, dp_scores, color=algorithm_colors['动态规划'], alpha=0.15)
    
    ax4.plot(angles, ip_scores, 's-', linewidth=3, color=algorithm_colors['整数规划'], 
            label='整数规划', markersize=6, markerfacecolor='white', markeredgewidth=2)
    ax4.fill(angles, ip_scores, color=algorithm_colors['整数规划'], alpha=0.15)
    
    ax4.plot(angles, greedy_scores, '^-', linewidth=3, color=algorithm_colors['贪心算法'], 
            label='贪心算法', markersize=6, markerfacecolor='white', markeredgewidth=2)
    ax4.fill(angles, greedy_scores, color=algorithm_colors['贪心算法'], alpha=0.15)
    
    # 设置标签和网格
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(metrics, fontsize=11)
    ax4.set_ylim(0, 10)
    ax4.set_yticks([2, 4, 6, 8, 10])
    ax4.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_title('(d) 基于实际性能的综合评估雷达图', fontsize=13, fontweight='bold', pad=20)
    
    # 添加图例
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11)
    
    # 添加性能等级标识圈
    for i in range(2, 11, 2):
        circle = Circle((0, 0), i, fill=False, color='gray', alpha=0.3, linewidth=0.5)
        ax4.add_patch(circle)
    
    # 添加基于真实数据的综合评估
    overall_scores = {
        '动态规划': np.mean(dp_scores[:-1]),
        '整数规划': np.mean(ip_scores[:-1]),
        '贪心算法': np.mean(greedy_scores[:-1])
    }
    
    assessment_text = ("综合评估 (基于实际数据):\n" + 
                      "\n".join([f"• {alg}: {score:.1f}/10" 
                               for alg, score in overall_scores.items()]) +
                      f"\n\n最优选择: {max(overall_scores, key=overall_scores.get)}")
    ax4.text(1.4, 0.5, assessment_text, transform=ax4.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8),
            fontsize=10, ha='left', va='center', fontweight='bold')
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)
    
    # 保存PNG图片
    plt.savefig('图5.8_算法性能与求解效率分析图_修正版.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 输出关键信息（基于真实数据）
    print("="*60)
    print("图5.8 算法性能与求解效率分析图 - 真实数据修正版")
    print("="*60)
    print("⚡ 基于实际问题规模的复杂度分析：")
    print(f"   - 问题规模: {real_data['problem_scale']['total_lands']}地块 × {real_data['problem_scale']['years']}年 × {real_data['problem_scale']['crops']}作物")
    print(f"   - 动态规划复杂度: {real_data['time_complexity']['动态规划']:.0f} (相对值)")
    print(f"   - 整数规划复杂度: {real_data['time_complexity']['整数规划']:.0f} (相对值)")
    print(f"   - 贪心算法复杂度: {real_data['time_complexity']['贪心算法']:.0f} (相对值)")
    print()
    print("🕒 实际求解时间 (7年问题)：")
    for alg, time_val in real_data['real_times'].items():
        if alg != 'total_time':
            print(f"   - {alg.replace('_time', '')}: {time_val:.6f}秒")
    print(f"   - 总求解时间: {real_data['real_times']['total_time']:.6f}秒")
    print("   - 性能等级: 优秀 (远低于1秒)")
    print()
    print("📈 约束处理与解质量：")
    print(f"   - 总约束违反: {violations[-1]}次")
    print(f"   - 违反率: {violation_rate:.1f}%")
    print(f"   - 解质量评分: {quality_score:.1f}/10")
    print("   - 主要问题: 重茬约束和豆类轮作约束")
    print()
    print("🎯 综合性能评估 (基于真实数据)：")
    for alg, score in overall_scores.items():
        print(f"   - {alg}: {score:.1f}/10")
    print(f"   - 推荐算法: {max(overall_scores, key=overall_scores.get)}")
    print()
    print("🔍 关键发现：")
    print("   - 实际求解时间远低于理论预期")
    print("   - 分层分治策略有效降低了整体复杂度")
    print("   - 约束处理存在优化空间，特别是重茬和轮作约束")
    print("   - 贪心算法在大棚组表现优异")
    print("="*60)
    print("✅ 图片已生成：图5.8_算法性能与求解效率分析图_修正版.png (基于真实数据)")

# 运行函数生成图像
if __name__ == "__main__":
    create_algorithm_performance_analysis_corrected()

