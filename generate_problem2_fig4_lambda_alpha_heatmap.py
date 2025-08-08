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

    # 固定情景池，公平对比
    opt = FinalStrictPaperImplementation()
    opt.load_and_process_data()
    opt.generate_stochastic_scenarios()

    # 参数网格（5×5）：λ（风险厌恶）× α（CVaR置信水平）
    lambda_list = [0.0, 0.25, 0.5, 0.75, 1.0]
    alpha_list = [0.20, 0.10, 0.05, 0.02, 0.01]
    N_saa = 30

    results = []

    for a in alpha_list:
        row = []
        for lam in lambda_list:
            print(f"\n=== λ={lam:.2f}, α={a:.2f}, SAA={N_saa} ===")
            opt.alpha = a
            opt.lambda_risk = lam
            opt.N_saa = N_saa
            opt.select_representative_scenarios()
            opt.build_stochastic_programming_model()
            ok = opt.solve_model()
            if not ok:
                print("⚠️ 求解失败，填充NaN")
                row.append({
                    'lambda': lam,
                    'alpha': a,
                    'expected_profit': np.nan,
                    'cvar_alpha': np.nan,
                    'tail_risk': np.nan,
                })
                continue

            profits = np.array(opt.scenario_profits, dtype=float)
            expected_profit = float(np.mean(profits))
            var_a = float(np.percentile(profits, a * 100))
            cvar_a = float(np.mean(profits[profits <= var_a]))
            tail_risk = expected_profit - cvar_a

            row.append({
                'lambda': lam,
                'alpha': a,
                'expected_profit': expected_profit,
                'cvar_alpha': cvar_a,
                'tail_risk': tail_risk,
            })
        results.append(row)

    # 整理为DataFrame并保存CSV（长表）
    flat = []
    for r in results:
        for cell in r:
            flat.append(cell)
    df = pd.DataFrame(flat)
    df.to_csv('图6.4_λ_α敏感性数据.csv', index=False, encoding='utf-8-sig')

    # 构造矩阵（α行 × λ列）
    profit_mat = np.array([[cell['expected_profit'] for cell in row] for row in results], dtype=float)
    cvar_mat = np.array([[cell['cvar_alpha'] for cell in row] for row in results], dtype=float)
    tail_mat = np.array([[cell['tail_risk'] for cell in row] for row in results], dtype=float)

    # 单位换算
    profit_yi = profit_mat / 1e8
    cvar_million = cvar_mat / 1e6
    tail_million = tail_mat / 1e6

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))

    im0 = axes[0].imshow(profit_yi, cmap='YlGn', aspect='auto', origin='upper')
    axes[0].set_title('期望收益（亿元）')
    axes[0].set_xticks(range(len(lambda_list)))
    axes[0].set_xticklabels([f'{lam:.2f}' for lam in lambda_list])
    axes[0].set_yticks(range(len(alpha_list)))
    axes[0].set_yticklabels([f'{a:.2f}' for a in alpha_list])
    axes[0].set_xlabel('λ（风险厌恶系数）')
    axes[0].set_ylabel('α（CVaR置信水平）')
    for i in range(len(alpha_list)):
        for j in range(len(lambda_list)):
            val = profit_yi[i, j]
            if np.isfinite(val):
                axes[0].text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8, color='black')

    cbar0 = fig.colorbar(im0, ax=axes[0])
    cbar0.ax.set_ylabel('亿元')

    im1 = axes[1].imshow(tail_million, cmap='OrRd', aspect='auto', origin='upper')
    axes[1].set_title('尾部风险 E−CVaRα（百万元）')
    axes[1].set_xticks(range(len(lambda_list)))
    axes[1].set_xticklabels([f'{lam:.2f}' for lam in lambda_list])
    axes[1].set_yticks(range(len(alpha_list)))
    axes[1].set_yticklabels([f'{a:.2f}' for a in alpha_list])
    axes[1].set_xlabel('λ（风险厌恶系数）')
    axes[1].set_ylabel('α（CVaR置信水平）')
    for i in range(len(alpha_list)):
        for j in range(len(lambda_list)):
            val = tail_million[i, j]
            if np.isfinite(val):
                axes[1].text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=8, color='black')

    cbar1 = fig.colorbar(im1, ax=axes[1])
    cbar1.ax.set_ylabel('百万元')

    fig.suptitle('图6.4 风险参数敏感性热力图（λ-α，K=1000，SAA=30）', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('图6.4_风险参数敏感性热力图.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 简要输出概览
    print('✅ 已生成：图6.4_风险参数敏感性热力图.png 与 图6.4_λ_α敏感性数据.csv')


if __name__ == '__main__':
    main()


