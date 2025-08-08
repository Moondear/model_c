import importlib.util
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter


def load_problem2_impl():
    spec = importlib.util.spec_from_file_location(
        "p2_impl", "第二题_最终严格版本.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FinalStrictPaperImplementation


def main():
    # 避免Windows控制台编码导致的Emoji打印报错
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
        sys.stderr.reconfigure(encoding='utf-8', errors='ignore')
    except Exception:
        pass
    FinalStrictPaperImplementation = load_problem2_impl()

    # 构造优化器并准备数据（统一场景，公平对比）
    opt = FinalStrictPaperImplementation()
    opt.N_saa = 20  # 为多λ批量求解提速，代表性情景数进一步降低

    opt.load_and_process_data()
    opt.generate_stochastic_scenarios()
    opt.select_representative_scenarios()

    # 待评估的风险厌恶系数（λ）
    lambda_list = [0.60, 0.75, 0.90]

    records = []

    for lam in lambda_list:
        print(f"\n=== 评估 λ = {lam} 的最优解 ===")
        opt.lambda_risk = lam

        # 重建并求解模型（共享同一批情景，确保可比）
        opt.build_stochastic_programming_model()
        ok = opt.solve_model()
        if not ok:
            print(f"⚠️ λ={lam} 求解失败，跳过")
            continue

        # 使用已验证的全情景收益序列计算指标（solve_model 内已调用验证）
        profits = np.array(opt.scenario_profits, dtype=float)
        exp_profit = float(np.mean(profits))
        std_profit = float(np.std(profits))
        var_5 = float(np.percentile(profits, 5))
        cvar_5 = float(np.mean(profits[profits <= var_5]))
        tail_risk = max(0.0, exp_profit - cvar_5)  # 作为“尾部风险”度量

        records.append({
            'lambda': lam,
            'expected_profit': exp_profit,
            'profit_std': std_profit,
            'var_5': var_5,
            'cvar_5': cvar_5,
            'tail_risk': tail_risk,
        })

    if not records:
        print("❌ 无有效记录，无法绘图")
        return

    df = pd.DataFrame(records).sort_values('lambda')
    df.to_csv('图6.1_风险收益数据.csv', index=False, encoding='utf-8-sig')

    # 绘制“风险-收益权衡前沿图”：x=尾部风险(E-CVaR5)，y=期望收益
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    # 单位换算：x轴用百万元，y轴用亿元，避免科学计数法
    x_vals = df['tail_risk'].values / 1e6   # 百万元
    y_vals = df['expected_profit'].values / 1e8  # 亿元

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.plot(x_vals, y_vals, '-o', color='#1f77b4', linewidth=2,
            markerfacecolor='white', markeredgecolor='#1f77b4', markersize=7)

    # 关闭科学计数与偏移
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    # 标注 λ 值，带背景框，边距避免裁切
    for i, row in df.iterrows():
        x = row['tail_risk'] / 1e6
        y = row['expected_profit'] / 1e8
        offset = (8, 8) if i % 2 == 0 else (8, -10)
        ax.annotate(
            f"λ={row['lambda']:.2f}",
            (x, y),
            textcoords="offset points", xytext=offset, ha='left', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none')
        )

    ax.set_xlabel('尾部风险（百万元）= 期望收益 - 5% CVaR')  # 使用减号，避免字体方框
    ax.set_ylabel('期望收益（亿元）')
    ax.set_title('图6.1 风险-收益权衡前沿（SAA=20，情景=1000，α=5%）')
    ax.grid(alpha=0.3)
    ax.margins(x=0.12, y=0.12)

    plt.tight_layout()
    plt.savefig('图6.1_风险-收益权衡前沿.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ 已生成：图6.1_风险-收益权衡前沿.png 与 图6.1_风险收益数据.csv")


if __name__ == '__main__':
    main()


