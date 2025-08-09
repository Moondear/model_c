# -*- coding: utf-8 -*-
# 文件名：run_optimize_7x_figs.py
import sys


def main():
    try:
        from 第三题_相关性与替代性_求解器 import Problem3CorrelatedSubstitutionSolver
    except ModuleNotFoundError:
        print("缺少依赖，请先安装：pip install pulp pandas numpy matplotlib openpyxl")
        raise

    s = Problem3CorrelatedSubstitutionSolver()
    # 统一实验设置（可按需调整）
    s.base.N_saa = 20
    s.base.random_seed = 42
    s.base.N_scenarios = 1000

    # 流程：数据→关系矩阵→情景→相关注入→SAA→建模→求解
    s.base.load_and_process_data()
    s.build_relationships()
    s.base.generate_stochastic_scenarios()
    s.generate_correlated_scenarios()
    s.base.select_representative_scenarios()
    s.base.build_stochastic_programming_model()
    ok = s.base.solve_model()
    if not ok:
        print("求解失败，已终止。")
        return

    # 生成全部 7.x 图（求解器内部已统一美化风格）
    s.generate_relation_heatmap('图7.1_作物关系热力图.png')
    s.generate_price_supply_scatter('图7.2_价格-供给散点.png')
    s.generate_inverse_demand_shift_plot('图7.X_倒需求_δ对比_主粮主菜.png')
    s.export_operation_schedule_and_gantt('问题三_作业清单.csv', '图7.X_作物更替甘特图.png')
    s.generate_greenhouse_monthly_utilization('问题三_大棚月度利用率.csv', '图7.X_大棚月度利用率.png')
    s.run_multi_seed_error_band(
        seeds=(42, 123, 2024),
        out_png='图7.X_多随机种子误差带.png',
        out_csv='问题三_多随机种子误差带.csv',
        saa=20
    )
    print("已完成全部 7.x 图像重绘。")


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
        sys.stderr.reconfigure(encoding='utf-8', errors='ignore')
    except Exception:
        pass
    main()


