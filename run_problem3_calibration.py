from 第三题_相关性与替代性_求解器 import Problem3CorrelatedSubstitutionSolver


def main():
    solver = Problem3CorrelatedSubstitutionSolver()
    # 基础准备与Copula标定
    solver.base.load_and_process_data()
    solver.build_relationships()
    solver.base.generate_stochastic_scenarios()
    solver.generate_correlated_scenarios()
    solver.write_param_sources_note()
    # 运行δ/ρ ±20%灵敏度（用较小SAA以控制时间与日志）
    solver.run_delta_rho_sensitivity(saa=10)
    print('DONE')


if __name__ == '__main__':
    main()



