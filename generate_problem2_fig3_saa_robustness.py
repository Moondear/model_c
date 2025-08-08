import importlib.util
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_problem2_impl():
    spec = importlib.util.spec_from_file_location(
        "p2_impl", "第二题_最终严格版本.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FinalStrictPaperImplementation


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
        sys.stderr.reconfigure(encoding='utf-8', errors='ignore')
    except Exception:
        pass

    FinalStrictPaperImplementation = load_problem2_impl()

    opt = FinalStrictPaperImplementation()
    opt.lambda_risk = 0.75

    # 统一情景，公平对比
    opt.load_and_process_data()
    opt.generate_stochastic_scenarios()

    saa_list = [10, 20, 30, 40, 50]
    records = []

    for saa in saa_list:
        print(f"\n=== 评估 SAA = {saa} 的解稳健性 ===")
        opt.N_saa = saa
        opt.select_representative_scenarios()
        opt.build_stochastic_programming_model()
        ok = opt.solve_model()
        if not ok:
            print(f"⚠️ SAA={saa} 求解失败，跳过")
            continue

        profits = np.array(opt.scenario_profits, dtype=float)
        expected_profit = float(np.mean(profits))
        profit_std = float(np.std(profits))
        var_5 = float(np.percentile(profits, 5))
        cvar_5 = float(np.mean(profits[profits <= var_5]))
        tail_risk = expected_profit - cvar_5

        records.append({
            'saa': saa,
            'expected_profit': expected_profit,
            'profit_std': profit_std,
            'var_5': var_5,
            'cvar_5': cvar_5,
            'tail_risk': tail_risk,
        })

    if not records:
        print("❌ 无有效记录，无法绘图")
        return

    df = pd.DataFrame(records).sort_values('saa')
    df.to_csv('图6.3_SAA_稳健性数据.csv', index=False, encoding='utf-8-sig')

    # 可视化：左轴=标准差（百万元），右轴=尾部风险（百万元）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    x = df['saa'].astype(str).values
    std_million = df['profit_std'].values / 1e6
    tail_million = df['tail_risk'].values / 1e6
    expected_yi = df['expected_profit'].values / 1e8

    fig, ax1 = plt.subplots(figsize=(10, 6))
    bars = ax1.bar(x, std_million, color='skyblue', edgecolor='black', alpha=0.8, label='收益标准差（百万元）')
    ax1.set_ylabel('收益标准差（百万元）')

    ax2 = ax1.twinx()
    ax2.plot(x, tail_million, '-o', color='firebrick', linewidth=2, label='尾部风险 E-CVaR5%（百万元）')
    ax2.set_ylabel('尾部风险（百万元）')

    # 在柱顶标注期望收益（亿元）
    for i, b in enumerate(bars):
        ax1.annotate(f"E={expected_yi[i]:.3f} 亿元", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                     xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9, color='dimgray')

    ax1.set_xlabel('SAA 代表性情景数')
    ax1.set_title('图6.3 SAA情景数对解稳健性的影响（K=1000，α=5%，λ=0.75）')
    ax1.grid(axis='y', alpha=0.3)

    # 合并图例
    lines, labels = [], []
    for a in (ax1, ax2):
        L = a.get_legend_handles_labels()
        lines += L[0]
        labels += L[1]
    ax1.legend(lines, labels, loc='upper right')

    plt.tight_layout()
    plt.savefig('图6.3_SAA对稳健性影响.png', dpi=300, bbox_inches='tight')
    plt.close()

    print('✅ 已生成：图6.3_SAA对稳健性影响.png 与 图6.3_SAA_稳健性数据.csv')


if __name__ == '__main__':
    main()


