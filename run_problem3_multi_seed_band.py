from 第三题_相关性与替代性_求解器 import Problem3CorrelatedSubstitutionSolver


def main():
    s = Problem3CorrelatedSubstitutionSolver()
    # 统一SAA以加速；如需更稳可改为50
    s.run_multi_seed_error_band(seeds=(42, 123, 2024), out_png='图7.X_多随机种子误差带.png', out_csv='问题三_多随机种子误差带.csv', saa=20)
    print('DONE')


if __name__ == '__main__':
    main()


