import importlib.util
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


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
    opt.N_saa = 30  # 代表性情景数，求解稳定且较快

    # 统一流程，得到全情景收益序列
    opt.load_and_process_data()
    opt.generate_stochastic_scenarios()
    opt.select_representative_scenarios()
    opt.build_stochastic_programming_model()
    ok = opt.solve_model()
    if not ok:
        print("❌ 求解失败，无法生成图6.2")
        return

    profits = np.array(opt.scenario_profits, dtype=float)

    # 关键指标
    mean_val = float(np.mean(profits))
    std_val = float(np.std(profits))
    var_5 = float(np.percentile(profits, 5))
    cvar_5 = float(np.mean(profits[profits <= var_5]))

    # 数据导出
    pd.DataFrame({
        'profit': profits
    }).to_csv('图6.2_收益分布_raw.csv', index=False, encoding='utf-8-sig')

    with open('图6.2_收益分布指标.txt', 'w', encoding='utf-8') as f:
        f.write(f"期望收益: {mean_val:.2f}\n")
        f.write(f"标准差: {std_val:.2f}\n")
        f.write(f"5% VaR: {var_5:.2f}\n")
        f.write(f"5% CVaR: {cvar_5:.2f}\n")

    # 绘图：x轴亿元，专业标注与阴影
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    x_vals = profits / 1e8
    mean_g = mean_val / 1e8
    var_g = var_5 / 1e8
    cvar_g = cvar_5 / 1e8

    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.hist(x_vals, bins=40, color='lightsteelblue', edgecolor='black', alpha=0.85)

    # 左尾阴影（VaR阈值左侧）
    ax.axvspan(min(x_vals), var_g, color='tomato', alpha=0.22, label='5% 左尾区间')
    ax.axvline(var_g, color='tomato', linestyle='--', linewidth=2, label='5% VaR')
    ax.axvline(cvar_g, color='firebrick', linestyle='-', linewidth=2, label='5% CVaR')
    ax.axvline(mean_g, color='navy', linestyle='--', linewidth=2, label='期望')

    # 文本注释
    ax.annotate(f'期望: {mean_g:.3f} 亿元', xy=(mean_g, ax.get_ylim()[1]*0.85),
                xytext=(10, 0), textcoords='offset points', color='navy')
    ax.annotate(f'VaR5%: {var_g:.3f} 亿元', xy=(var_g, ax.get_ylim()[1]*0.65),
                xytext=(10, 0), textcoords='offset points', color='tomato')
    ax.annotate(f'CVaR5%: {cvar_g:.3f} 亿元', xy=(cvar_g, ax.get_ylim()[1]*0.45),
                xytext=(10, 0), textcoords='offset points', color='firebrick')

    ax.set_xlabel('总收益（亿元）')
    ax.set_ylabel('频次')
    ax.set_title('图6.2 收益分布与VaR/CVaR（K=1000，α=5%）')
    ax.grid(alpha=0.3)
    ax.legend()
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.margins(x=0.03)

    plt.tight_layout()
    plt.savefig('图6.2_收益分布_VaR_CVaR.png', dpi=300, bbox_inches='tight')
    plt.close()

    print('✅ 已生成：图6.2_收益分布_VaR_CVaR.png、图6.2_收益分布_raw.csv、图6.2_收益分布指标.txt')


if __name__ == '__main__':
    main()


