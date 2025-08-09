from 第三题_相关性与替代性_求解器 import Problem3CorrelatedSubstitutionSolver


def main():
    s = Problem3CorrelatedSubstitutionSolver()
    # 为对比可复现，将SAA设为20，运行四配置消融并输出CSV
    df = s.run_ablation_experiments(out_csv='问题三_消融实验.csv', saa=20)
    print(df)
    print('DONE')


if __name__ == '__main__':
    main()


