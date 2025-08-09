from 第三题_相关性与替代性_求解器 import Problem3CorrelatedSubstitutionSolver


def main():
    s = Problem3CorrelatedSubstitutionSolver()
    # 为加速稳健性实验，适当降低SAA代表性情景数
    s.base.N_saa = 20
    # 启动主流程，并对尾部M进行稳健性检验
    # 取 M 为 N_saa 的 10%、20% 与全体
    m20 = max(1, int(0.2 * s.base.N_saa))
    s.run(iterations=10, tail_M=m20, test_tail_robustness=True)
    print('DONE')


if __name__ == '__main__':
    main()


