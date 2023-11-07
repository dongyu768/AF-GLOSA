import argparse
def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--EPISODES', default=20000, type=int, help='')
    #sumo
    parse.add_argument('--OPTION_NOGUI', default=True, type=bool, help='')
    parse.add_argument('--configFile', default="E:/Program Files/PyWorkstation/AF_GLOSA/sumo/inter.sumocfg", type=str, help='')
    parse.add_argument('--tripFile', default="E:/Program Files/PyWorkstation/AF_GLOSA/sumo/tripinfo.tr", type=str, help='')
    parse.add_argument('--l_g', default=240, type=int, help='speed guidance length')

    #hppo
    parse.add_argument('--state_dim', type=int, default=6, help='')
    parse.add_argument('--discrete_dim', type=int, default=2, help='output dim')
    parse.add_argument('--continuous_dim', type=int, default=1, help='output dim')
    parse.add_argument('--batch_size', type=int, default=256, help='')
    parse.add_argument('--mini_batch_size', type=int, default=8, help='')
    parse.add_argument('--buffer_size', type=int, default=10000, help='')
    parse.add_argument('--gamma', type=float, default=0.99, help='')
    parse.add_argument('--epochs', type=int, default=12, help='')
    parse.add_argument('--lamda', type=float, default=0.2, help='')
    parse.add_argument('--entropy_coef', type=float, default=0.01, help='')
    parse.add_argument('--epsilon', type=float, default=0.1, help='')
    parse.add_argument('--discrete_LR', type=float, default=3e-6, help='')
    parse.add_argument('--continuous_LR', type=float, default=3e-5, help='')
    parse.add_argument('--critic_LR', type=float, default=0.001, help='')
    parse.add_argument('--amax', type=int, default=2.5, help='')
    parse.add_argument('--amin', type=int, default=2.5, help='')
    parse.add_argument('--vmax', type=int, default=11, help='')
    parse.add_argument('--vmin', type=int, default=2, help='')
    parse.add_argument('--control_step', type=int, default=2, help='')
    parse.add_argument('--modelPath', type=str, default='E:/Program Files/PyWorkstation/AF_GLOSA/models/model081401', help='')

    args = parse.parse_args()
    return args