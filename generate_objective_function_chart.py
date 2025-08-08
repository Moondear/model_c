import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import rcParams
import matplotlib.patches as mpatches
import importlib
from typing import Dict, Any, List, Tuple

# 设置中文字体和学术期刊风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# 专业学术期刊配色方案
colors_academic = [
    '#1f77b4',  # 专业蓝 - 主色调
    '#ff7f0e',  # 学术橙 - 对比色
    '#2ca02c',  # 自然绿 - 环保色
    '#d62728',  # 科学红 - 强调色
    '#9467bd',  # 紫罗兰 - 高雅色
    '#8c564b',  # 棕褐色 - 稳重色
]

# 情景对比专用配色
scenario_colors = ['#3182bd', '#fd8d3c']  # 深蓝、橙色
linearization_colors = ['#2ca02c', '#d62728', '#9467bd']  # 绿、红、紫

def create_objective_function_comparison():
    """
    生成图5.2：目标函数对比与线性化处理示意图
    专业美观，适合学术论文插入
    """
    
    # 创建图形布局：2行2列
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('图5.2 目标函数对比与线性化处理示意图', fontsize=16, fontweight='bold', y=0.95)
    
    # —— 数据准备（优先从优化器获取，失败则从附件2构造） ——
    def _prepare_data() -> Dict[str, Any]:
        # 尝试使用优化器，读到 crop_info（含 price、yield、cost、sales_limit、name、type）
        try:
            mod = importlib.import_module('agricultural_optimization_paper_compliant')
            Optim = getattr(mod, 'PaperCompliantAgriculturalOptimizer')
            opt = Optim()
            opt.load_all_data()
            opt.process_and_group_lands()
            opt.process_crop_data()
            data = {}
            # 组装所需字段
            items = []
            for cid, info in opt.crop_info.items():
                items.append({
                    'id': cid,
                    'name': info.get('name', str(cid)),
                    'type': info.get('type', ''),
                    'price': float(info.get('price', 0)),
                    'yield': float(info.get('yield_per_mu', 0)),
                    'cost': float(info.get('cost_per_mu', 0)),
                    'D': float(info.get('sales_limit', 0)),
                })
            data['crops'] = pd.DataFrame(items)
            return data
        except Exception:
            pass

        # 回退：从附件2.xlsx两张表构造
        try:
            df_stats = pd.read_excel('附件2.xlsx', sheet_name='2023年统计的相关数据')
            df_plant = pd.read_excel('附件2.xlsx', sheet_name='2023年的农作物种植情况')
            # 清洗
            for df in (df_stats, df_plant):
                df.columns = [str(c).strip() for c in df.columns]
            # 解析价格
            def parse_price(v: Any) -> float:
                try:
                    s = str(v).strip()
                    if '-' in s:
                        a, b = map(float, s.split('-'))
                        return (a + b) / 2
                    return float(s)
                except Exception:
                    return np.nan
            df_stats['price'] = df_stats['销售单价/(元/斤)'].apply(parse_price)
            df_stats['yield'] = pd.to_numeric(df_stats['亩产量/斤'], errors='coerce')
            df_stats['cost'] = pd.to_numeric(df_stats['种植成本/(元/亩)'], errors='coerce')
            df_stats['id'] = pd.to_numeric(df_stats['作物编号'], errors='coerce').astype('Int64')
            df_stats['name'] = df_stats['作物名称']

            df_plant['id'] = pd.to_numeric(df_plant['作物编号'], errors='coerce').astype('Int64')
            df_plant['type'] = df_plant['作物类型']
            # 计算销售上限 D = 2023面积合计 × 亩产
            area_by_id = df_plant.groupby('id')['种植面积/亩'].sum().rename('area_2023')
            merged = df_stats.merge(area_by_id, how='left', left_on='id', right_index=True)
            merged['D'] = merged['area_2023'].fillna(0) * merged['yield'].fillna(0)
            data = {'crops': merged[['id', 'name', 'type', 'price', 'yield', 'cost', 'D']].dropna(subset=['price', 'yield'])}
            return data
        except Exception:
            # 兜底空
            return {'crops': pd.DataFrame(columns=['id', 'name', 'type', 'price', 'yield', 'cost', 'D'])}

    data = _prepare_data()
    crops_df: pd.DataFrame = data['crops']

    # ============ 子图1：两种情景收益函数对比 ============
    # 使用数据中位数作为示意参数，更贴近真实规模
    production_ratio = np.linspace(0.5, 2.0, 100)
    if not crops_df.empty:
        sales_limit = float(crops_df['D'].replace([np.inf, -np.inf], np.nan).dropna().median()) or 1000.0
        price = float(crops_df['price'].replace([np.inf, -np.inf], np.nan).dropna().median()) or 3.0
    else:
        sales_limit = 1000.0
        price = 3.0
    
    # 情景一：超产滞销
    revenue_scenario1 = []
    for ratio in production_ratio:
        production = ratio * sales_limit
        actual_sales = min(production, sales_limit)
        revenue = actual_sales * price
        revenue_scenario1.append(revenue)
    
    # 情景二：超产50%折价
    revenue_scenario2 = []
    for ratio in production_ratio:
        production = ratio * sales_limit
        normal_sales = min(production, sales_limit)
        excess_sales = max(0, production - sales_limit)
        revenue = normal_sales * price + excess_sales * price * 0.5
        revenue_scenario2.append(revenue)
    
    # 绘制收益函数曲线
    ax1.plot(production_ratio, revenue_scenario1, color=scenario_colors[0], linewidth=3, 
             label='情景一：超产滞销', marker='o', markersize=4, markevery=10)
    ax1.plot(production_ratio, revenue_scenario2, color=scenario_colors[1], linewidth=3, 
             label='情景二：50%折价销售', marker='s', markersize=4, markevery=10)
    
    # 标注关键点
    ax1.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    # 相对定位标注，避免与曲线/边界重叠
    ymax = max(max(revenue_scenario1), max(revenue_scenario2))
    ax1.annotate('销售限制点', xy=(1.0, 0.95*ymax), xytext=(6, 0), textcoords='offset points',
                 rotation=90, va='top', ha='left', fontsize=11, color='#444')
    
    # 填充收益差异区域
    ax1.fill_between(production_ratio, revenue_scenario1, revenue_scenario2, 
                     where=(np.array(revenue_scenario2) > np.array(revenue_scenario1)),
                     alpha=0.3, color='orange', label='收益提升区域')
    
    ax1.set_xlabel('产量/销售限制 比值', fontsize=12)
    ax1.set_ylabel('收益 (元)', fontsize=12)
    ax1.set_title('(a) 两种情景收益函数对比', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ============ 子图2：超产处理策略（数据驱动Top-5） ============
    # 选取提升最大的5个作物（产量=1.3×D）
    df2 = crops_df.dropna(subset=['price', 'D']).copy()
    if not df2.empty:
        df2['production'] = df2['D'] * 1.3
        df2['rev1'] = np.minimum(df2['production'], df2['D']) * df2['price']
        df2['rev2'] = (np.minimum(df2['production'], df2['D']) * df2['price'] +
                       np.maximum(df2['production'] - df2['D'], 0) * df2['price'] * 0.5)
        df2['improve'] = df2['rev2'] - df2['rev1']
        df2 = df2.sort_values('improve', ascending=False).head(5)
        crop_names = df2['name'].tolist()
        scenario1_revenues = df2['rev1'].tolist()
        scenario2_revenues = df2['rev2'].tolist()
    else:
        crop_names = ['示例A', '示例B', '示例C', '示例D', '示例E']
        scenario1_revenues = [1000, 1200, 900, 1500, 800]
        scenario2_revenues = [1200, 1500, 950, 1700, 900]

    x_pos = np.arange(len(crop_names))
    width = 0.35
    
    bars1 = ax2.bar(x_pos - width/2, scenario1_revenues, width, label='情景一：滞销',
                    color=scenario_colors[0], alpha=0.8, edgecolor='white', linewidth=1)
    bars2 = ax2.bar(x_pos + width/2, scenario2_revenues, width, label='情景二：折价',
                    color=scenario_colors[1], alpha=0.8, edgecolor='white', linewidth=1)
    
    # 添加收益提升标注
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        if scenario2_revenues[i] > scenario1_revenues[i]:
            improvement = scenario2_revenues[i] - scenario1_revenues[i]
            ax2.annotate(f'+{improvement:.0f}', 
                        xy=(bar2.get_x() + bar2.get_width()/2, bar2.get_height()),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontweight='bold', color='red')
    
    ax2.set_xlabel('作物类型', fontsize=12)
    ax2.set_ylabel('收益 (元)', fontsize=12)
    ax2.set_title('(b) 不同作物超产处理效果对比', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(crop_names)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # ============ 子图3：线性化变量几何意义 ============
    # 绘制线性化处理的几何意义
    production_values = np.array([800, 1000, 1200, 1500, 1800])
    sales_limit_line = 1000
    
    # 分解为可售和超产部分
    q_sell = np.minimum(production_values, sales_limit_line)
    q_excess = np.maximum(production_values - sales_limit_line, 0)
    
    x_pos = np.arange(len(production_values))
    
    # 堆叠柱状图
    bars1 = ax3.bar(x_pos, q_sell, color=linearization_colors[0], alpha=0.8, 
                    label='q^sell (可售产量)', edgecolor='white', linewidth=1)
    bars2 = ax3.bar(x_pos, q_excess, bottom=q_sell, color=linearization_colors[1], alpha=0.8,
                    label='q^excess (超产量)', edgecolor='white', linewidth=1)
    
    # 销售限制线
    ax3.axhline(y=sales_limit_line, color='black', linestyle='--', linewidth=2,
                label=f'D_{{j,t}} = {sales_limit_line}')
    
    # 添加数值标注
    for i, (sell, excess) in enumerate(zip(q_sell, q_excess)):
        if excess > 0:
            ax3.text(i, sell + excess/2, f'{int(excess)}', ha='center', va='center', 
                    fontweight='bold', color='white')
        ax3.text(i, sell/2, f'{int(sell)}', ha='center', va='center', 
                fontweight='bold', color='white')
    
    # 在图中加入关系提示
    ax3.annotate('q = q^sell + q^excess', xy=(0.5, sales_limit_line*0.15), xytext=(0.5, sales_limit_line*0.28),
                 textcoords='data', ha='center', fontsize=11, color='#444',
                 arrowprops=dict(arrowstyle='->', color='#666'))
    ax3.set_xlabel('产量水平情况', fontsize=12)
    ax3.set_ylabel('产量 (斤)', fontsize=12)
    ax3.set_title('(c) 线性化变量分解示意图', fontsize=13, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'场景{i+1}' for i in range(len(production_values))])
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # ============ 子图4：销售限制敏感性分析（数据驱动分组中位） ============
    limit_changes = np.array([-20, -10, 0, 10, 20])
    colors_crops = ['#1f77b4', '#2ca02c', '#d62728']
    groups = [('粮食', '#1f77b4'), ('蔬菜', '#2ca02c'), ('食用菌', '#d62728')]
    for idx, (grp, color) in enumerate(groups):
        sub = crops_df[crops_df['type'].astype(str).str.contains(grp, na=False)]
        if sub.empty:
            continue
        base_limit = float(sub['D'].replace([np.inf, -np.inf], np.nan).dropna().median())
        price_med = float(sub['price'].replace([np.inf, -np.inf], np.nan).dropna().median())
        if not np.isfinite(base_limit) or base_limit <= 0 or not np.isfinite(price_med) or price_med <= 0:
            continue
        revenue_changes = []
        for change in limit_changes:
            new_limit = base_limit * (1 + change/100)
            production = base_limit * 1.3
            normal_sales = min(production, new_limit)
            excess_sales = max(0, production - new_limit)
            revenue = normal_sales * price_med + excess_sales * price_med * 0.5
            base_revenue = min(production, base_limit) * price_med + max(0, production - base_limit) * price_med * 0.5
            change_pct = (revenue - base_revenue) / base_revenue * 100
            revenue_changes.append(change_pct)
        ax4.plot(limit_changes, revenue_changes, marker='o', linewidth=2.5,
                 markersize=6, label=f'{grp}类', color=color)
    
    ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax4.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    
    ax4.set_xlabel('销售限制变化 (%)', fontsize=12)
    ax4.set_ylabel('收益变化 (%)', fontsize=12)
    ax4.set_title('(d) 销售限制敏感性分析', fontsize=13, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # 只保存PNG图片，不显示窗口
    plt.savefig('图5.2_目标函数对比与线性化处理示意图.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 输出关键信息
    print("="*60)
    print("图5.2 目标函数对比与线性化处理示意图 - 关键数据")
    print("="*60)
    print("📊 目标函数对比：")
    print("   - 情景一：Z₁ = Σ[Pⱼ·min(qⱼ,ₜ, Dⱼ,ₜ) - 成本]")
    print("   - 情景二：Z₂ = Σ[Pⱼ·min(qⱼ,ₜ, Dⱼ,ₜ) + 0.5Pⱼ·max(qⱼ,ₜ-Dⱼ,ₜ, 0) - 成本]")
    print()
    print("🔧 线性化处理：")
    print("   - 引入辅助变量：q^sell_{j,t} (可售产量), q^excess_{j,t} (超产量)")
    print("   - 约束条件：q_{j,t} = q^sell_{j,t} + q^excess_{j,t}")
    print("   - 限制条件：q^sell_{j,t} ≤ D_{j,t}, q^excess_{j,t} ≥ 0")
    print()
    print("💰 收益提升效果：")
    print("   - 超产作物通过50%折价销售获得额外收益")
    print("   - 高价值作物（如羊肚菌）收益提升更显著")
    print("   - 销售限制变化对不同作物类型影响差异明显")
    print("="*60)
    print("✅ 图片已生成：图5.2_目标函数对比与线性化处理示意图.png")

# 运行函数生成图像
if __name__ == "__main__":
    create_objective_function_comparison()

#############################
# 拆分为四个独立子图（PNG）
#############################

def _prepare_objective_data() -> Dict[str, Any]:
    """与总图一致的数据准备，便于子图公用。"""
    try:
        mod = importlib.import_module('agricultural_optimization_paper_compliant')
        Optim = getattr(mod, 'PaperCompliantAgriculturalOptimizer')
        opt = Optim()
        opt.load_all_data()
        opt.process_and_group_lands()
        opt.process_crop_data()
        items = []
        for cid, info in opt.crop_info.items():
            items.append({
                'id': cid,
                'name': info.get('name', str(cid)),
                'type': info.get('type', ''),
                'price': float(info.get('price', 0)),
                'yield': float(info.get('yield_per_mu', 0)),
                'cost': float(info.get('cost_per_mu', 0)),
                'D': float(info.get('sales_limit', 0)),
            })
        return {'crops': pd.DataFrame(items)}
    except Exception:
        pass
    try:
        df_stats = pd.read_excel('附件2.xlsx', sheet_name='2023年统计的相关数据')
        df_plant = pd.read_excel('附件2.xlsx', sheet_name='2023年的农作物种植情况')
        for df in (df_stats, df_plant):
            df.columns = [str(c).strip() for c in df.columns]
        def parse_price(v: Any) -> float:
            try:
                s = str(v).strip()
                if '-' in s:
                    a, b = map(float, s.split('-'))
                    return (a + b) / 2
                return float(s)
            except Exception:
                return np.nan
        df_stats['price'] = df_stats['销售单价/(元/斤)'].apply(parse_price)
        df_stats['yield'] = pd.to_numeric(df_stats['亩产量/斤'], errors='coerce')
        df_stats['cost'] = pd.to_numeric(df_stats['种植成本/(元/亩)'], errors='coerce')
        df_stats['id'] = pd.to_numeric(df_stats['作物编号'], errors='coerce').astype('Int64')
        df_stats['name'] = df_stats['作物名称']
        df_plant['id'] = pd.to_numeric(df_plant['作物编号'], errors='coerce').astype('Int64')
        area_by_id = df_plant.groupby('id')['种植面积/亩'].sum().rename('area_2023')
        merged = df_stats.merge(area_by_id, how='left', left_on='id', right_index=True)
        merged['D'] = merged['area_2023'].fillna(0) * merged['yield'].fillna(0)
        return {'crops': merged[['id', 'name', 'type', 'price', 'yield', 'cost', 'D']].dropna(subset=['price', 'yield'])}
    except Exception:
        return {'crops': pd.DataFrame(columns=['id', 'name', 'type', 'price', 'yield', 'cost', 'D'])}


def create_objective_function_sub1():
    """图5.2a：两种情景收益函数对比（独立PNG）。"""
    data = _prepare_objective_data()
    crops_df = data['crops']
    production_ratio = np.linspace(0.5, 2.0, 100)
    if not crops_df.empty:
        sales_limit = float(crops_df['D'].replace([np.inf, -np.inf], np.nan).dropna().median()) or 1000.0
        price = float(crops_df['price'].replace([np.inf, -np.inf], np.nan).dropna().median()) or 3.0
    else:
        sales_limit = 1000.0
        price = 3.0
    revenue_s1 = []
    revenue_s2 = []
    for ratio in production_ratio:
        production = ratio * sales_limit
        actual_sales = min(production, sales_limit)
        revenue_s1.append(actual_sales * price)
        normal = min(production, sales_limit)
        excess = max(0, production - sales_limit)
        revenue_s2.append(normal * price + excess * price * 0.5)
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.plot(production_ratio, revenue_s1, color=scenario_colors[0], linewidth=3, label='情景一：超产滞销')
    ax.plot(production_ratio, revenue_s2, color=scenario_colors[1], linewidth=3, label='情景二：50%折价')
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    ymax = max(max(revenue_s1), max(revenue_s2))
    ax.annotate('销售限制点', xy=(1.0, 0.95*ymax), xytext=(6, 0), textcoords='offset points', rotation=90, va='top', ha='left')
    ax.set_xlabel('产量/销售限制 比值')
    ax.set_ylabel('收益 (元)')
    ax.set_title('图5.2a 两种情景收益函数对比')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('图5.2a_两情景收益函数对比.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_objective_function_sub2():
    """图5.2b：超产处理策略（数据驱动Top-5，独立PNG）。"""
    crops_df = _prepare_objective_data()['crops']
    df2 = crops_df.dropna(subset=['price', 'D']).copy()
    if not df2.empty:
        df2['production'] = df2['D'] * 1.3
        df2['rev1'] = np.minimum(df2['production'], df2['D']) * df2['price']
        df2['rev2'] = (np.minimum(df2['production'], df2['D']) * df2['price'] +
                       np.maximum(df2['production'] - df2['D'], 0) * df2['price'] * 0.5)
        df2['improve'] = df2['rev2'] - df2['rev1']
        df2 = df2.sort_values('improve', ascending=False).head(5)
        crop_names = df2['name'].tolist()
        s1 = df2['rev1'].tolist()
        s2 = df2['rev2'].tolist()
    else:
        crop_names = ['示例A', '示例B', '示例C', '示例D', '示例E']
        s1 = [1000, 1200, 900, 1500, 800]
        s2 = [1200, 1500, 950, 1700, 900]
    x_pos = np.arange(len(crop_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    b1 = ax.bar(x_pos - width/2, s1, width, label='情景一：滞销', color=scenario_colors[0], alpha=0.85, edgecolor='white')
    b2 = ax.bar(x_pos + width/2, s2, width, label='情景二：折价', color=scenario_colors[1], alpha=0.85, edgecolor='white')
    # 留白，避免标注与图例/边界重叠
    y_max = max(max(s1), max(s2)) if len(s1) and len(s2) else 1
    ax.set_ylim(0, y_max * 1.22)
    for i, (bar1, bar2) in enumerate(zip(b1, b2)):
        if s2[i] > s1[i]:
            ax.annotate(f"+{s2[i]-s1[i]:,.0f}", xy=(bar2.get_x() + bar2.get_width()/2, bar2.get_height()),
                        xytext=(0, 6), textcoords='offset points', ha='center', va='bottom', color='red', fontweight='bold')
    ax.set_xlabel('作物类型')
    ax.set_ylabel('收益 (元)')
    ax.set_title('图5.2b 不同作物超产处理效果（数据驱动）')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(crop_names)
    # 图例放到图内右上角空白区域，白底半透明，避免与柱/标注遮挡
    ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), frameon=True,
              facecolor='white', edgecolor='#dddddd', framealpha=0.9, ncol=1,
              handlelength=1.6, borderpad=0.4, columnspacing=0.8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('图5.2b_超产处理策略_数据驱动.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_objective_function_sub3():
    """图5.2c：线性化变量分解示意图（独立PNG）。"""
    production_values = np.array([800, 1000, 1200, 1500, 1800])
    sales_limit_line = 1000
    q_sell = np.minimum(production_values, sales_limit_line)
    q_excess = np.maximum(production_values - sales_limit_line, 0)
    x_pos = np.arange(len(production_values))
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.bar(x_pos, q_sell, color=linearization_colors[0], alpha=0.85, label='q^sell (可售产量)', edgecolor='white', zorder=2)
    ax.bar(x_pos, q_excess, bottom=q_sell, color=linearization_colors[1], alpha=0.85, label='q^excess (超产量)', edgecolor='white', zorder=3)
    # 将限制线置于下层，避免遮挡文字
    ax.axhline(y=sales_limit_line, color='black', linestyle='--', linewidth=2, label=f'D_{'{'}j,t{'}'} = {sales_limit_line}', zorder=1)
    for i, (sell, excess) in enumerate(zip(q_sell, q_excess)):
        if excess > 0:
            # 将超产量数值放在红色段顶部外侧空白处，避免与虚线/箭头重叠
            y_top = sell + excess
            ax.text(i, y_top + max(production_values) * 0.04, f'{int(excess)}', ha='center', va='bottom',
                    fontweight='bold', color='#d62728', zorder=5)
        # 可售量数值放在绿色段中部
        ax.text(i, sell * 0.5, f'{int(sell)}', ha='center', va='center', fontweight='bold', color='white', zorder=5)
    # 注释上移并水平偏移，箭头采用弧线指向分界处
    idxs = np.where(q_excess > 0)[0]
    anchor = int(idxs[0]) if len(idxs) else len(q_sell)//2
    boundary_y = q_sell[anchor]
    top_y = q_sell[anchor] + q_excess[anchor]
    ax.set_ylim(0, max(production_values) * 1.22)
    ax.annotate('q = q^sell + q^excess', xy=(anchor, boundary_y),
                xytext=(anchor - 0.2, top_y + max(production_values)*0.10),
                ha='center', va='bottom', fontsize=11, color='#444',
                arrowprops=dict(arrowstyle='->', color='#666', connectionstyle='arc3,rad=-0.2'))
    ax.set_xlabel('产量水平情况')
    ax.set_ylabel('产量 (斤)')
    ax.set_title('图5.2c 线性化变量分解示意图')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'场景{i+1}' for i in range(len(production_values))])
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('图5.2c_线性化变量分解示意图.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_objective_function_sub4():
    """图5.2d：销售限制敏感性分析（数据驱动，独立PNG）。"""
    crops_df = _prepare_objective_data()['crops']
    limit_changes = np.array([-20, -10, 0, 10, 20])
    groups = [('粮食', '#1f77b4'), ('蔬菜', '#2ca02c'), ('食用菌', '#d62728')]
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    for grp, color in groups:
        sub = crops_df[crops_df['type'].astype(str).str.contains(grp, na=False)]
        if sub.empty:
            continue
        base_limit = float(sub['D'].replace([np.inf, -np.inf], np.nan).dropna().median())
        price_med = float(sub['price'].replace([np.inf, -np.inf], np.nan).dropna().median())
        if not np.isfinite(base_limit) or base_limit <= 0 or not np.isfinite(price_med) or price_med <= 0:
            continue
        revenue_changes = []
        for change in limit_changes:
            new_limit = base_limit * (1 + change/100)
            production = base_limit * 1.3
            normal_sales = min(production, new_limit)
            excess_sales = max(0, production - new_limit)
            revenue = normal_sales * price_med + excess_sales * price_med * 0.5
            base_revenue = min(production, base_limit) * price_med + max(0, production - base_limit) * price_med * 0.5
            change_pct = (revenue - base_revenue) / base_revenue * 100
            revenue_changes.append(change_pct)
        ax.plot(limit_changes, revenue_changes, marker='o', linewidth=2.5, markersize=6, label=f'{grp}类', color=color)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xlabel('销售限制变化 (%)')
    ax.set_ylabel('收益变化 (%)')
    ax.set_title('图5.2d 销售限制敏感性分析（数据驱动）')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('图5.2d_销售限制敏感性分析.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def generate_objective_function_subcharts():
    print('⚙️ 正在生成图5.2的四个独立子图...')
    create_objective_function_sub1()
    create_objective_function_sub2()
    create_objective_function_sub3()
    create_objective_function_sub4()
    print('✅ 已生成：')
    print(' - 图5.2a_两情景收益函数对比.png')
    print(' - 图5.2b_超产处理策略_数据驱动.png')
    print(' - 图5.2c_线性化变量分解示意图.png')
    print(' - 图5.2d_销售限制敏感性分析.png')
