from env import SumoEnv
from agent.hppo import HPPO
from utils import *
from sumoutils import *
from args import parse_args
from agent.stateNormalization import *
from agent.rewardScaling import *
import random
# random.seed(100)
def main2():
    env = SumoEnv(args.configFile, args.tripFile, args.OPTION_NOGUI)
    agent = HPPO(args, writer)
    agent.model_load()
    stateNormal = Normalization(7)
    rewardScaling = RewardScaling(1, 0.9)
    for i in range(args.EPISODES):
        loadSumo(args.configFile, args.tripFile, args.OPTION_NOGUI)
        vehicle_id = getControledVeh()
        while vehicle_id not in traci.vehicle.getIDList():
            traci.simulationStep()
        # departSpeed = random.randint(4, 11)  # randomly set the depart time of vehicle
        departSpeed = 8
        traci.vehicle.setSpeed(vehicle_id, departSpeed)
        # print("============", departSpeed, "============")

        epochs = getDecisionEpoch(vehicle_id)
        ep_reward = 0
        for epoch in range(epochs):
            if vehicle_id not in traci.vehicle.getIDList():
                break
            done = 0
            light = getAheadTLS(vehicle_id)
            l_t = getL_t(vehicle_id)
            while l_t > args.l_g:
                traci.simulationStep()
                l_t = getL_t(vehicle_id)
            else:
                # print('enter the speed guidance area...')
                # beforeTime = traci.simulation.getTime()
                state = env.reset()
                state[0:7] = stateNormal(state[0:7])
                state = np.array(state)
                state = torch.from_numpy(state).float()
                while True:
                    _, _, _, _, _, p_t = state
                    a_dis, a_prob_dis, a_con, a_prob_con = agent.choose_action(state)

                    cur_light = getAheadTLS(vehicle_id)
                    print(a_dis, a_con)
                    # if cur_light != light:
                    #     afterTime = traci.simulation.getTime()
                    # reward, state_ = env.step(a_dis, a_con, beforeTime)
                    reward, state_ = env.step(a_dis, a_con)
                    reward = rewardScaling(reward)
                    # ep_reward += reward

                    state_[0:7] = stateNormal(state_[0:7])
                    state_ = np.array(state_)
                    state_ = torch.from_numpy(state_).float()
                    agent.buffer.append(state, a_dis, a_prob_dis, a_con, a_prob_con, reward, state_, done)
                    if agent.buffer.is_full():
                        agent.update()
                    state = state_
                    ep_reward += reward

                    if cur_light != light:
                        done = 1
                        light = cur_light
                    if done == 1:
                        traci.vehicle.setSpeed(vehicle_id, args.vmax)
                        break
        print("ep_reward:", ep_reward)
        rewardScaling.reset()
        writer.add_scalar('reward/episodes', ep_reward, i)
        while vehicle_id in traci.vehicle.getIDList() and len(traci.vehicle.getRoadID(vehicle_id)) != 0:
            traci.simulationStep()

        traci.close()
        if (i + 1) % 10000 == 0:
            agent.model_save(i + 1)

if __name__ == '__main__':
    logPath = 'runs/log081401'
    writer = Writer(logPath)
    args = parse_args()
    main2()
