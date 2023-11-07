from env import SumoEnv
from agent.hppo import HPPO
from utils import *
from sumoutils import *
from args import parse_args
import torch
from agent.stateNormalization import *
from agent.rewardScaling import *
import csv
import bs4
def testHPPO():
    env = SumoEnv(args.configFile, args.tripFile, args.OPTION_NOGUI)
    agent = HPPO(args, writer)
    stateNormal = Normalization(5)
    # rewardScaling = RewardScaling(1, 0.9)
    csvfile = open('E:/Program Files/PyWorkstation/AF_GLOSA/data/MultiHPPO1401_2_random100_2700.csv', mode='w', newline='')
    fieldnames = ['duration', 'waitingtime', 'waitingcount', 'timeloss', 'co2', 'fuel']
    write = csv.DictWriter(csvfile, fieldnames=fieldnames)
    write.writeheader()
    for i in range(100):
        ep_co2 = 0
        ep_fuel = 0
        loadSumo(args.configFile, args.tripFile, args.OPTION_NOGUI)
        vehicle_id = getControledVeh()
        while vehicle_id not in traci.vehicle.getIDList():
            traci.simulationStep()
        departSpeed = 8
        traci.vehicle.setSpeed(vehicle_id, departSpeed)

        epochs = getDecisionEpoch(vehicle_id)
        for epoch in range(epochs):
            if vehicle_id not in traci.vehicle.getIDList():
                break
            done = 0
            light = getAheadTLS(vehicle_id)
            l_t = getL_t(vehicle_id)
            while l_t > args.l_g:
                co2 = traci.vehicle.getCO2Emission(vehicle_id)
                fuel = traci.vehicle.getFuelConsumption(vehicle_id)
                traci.simulationStep()
                ep_co2 += co2
                ep_fuel += fuel
                l_t = getL_t(vehicle_id)
            else:
                # print('enter the speed guidance area...')
                # beforeTime = traci.simulation.getTime()
                state = env.reset()
                agent.model_load()
                state[0:5] = stateNormal(state[0:5])
                state = np.array(state)
                state = torch.from_numpy(state).float()
                while True:
                    _, _, _, _, _, p_t = state
                    a_dis, a_prob_dis, a_con, a_prob_con = agent.choose_action(state)

                    cur_light = getAheadTLS(vehicle_id)
                    reward, state_ = env.step(a_dis, a_con)

                    state_[0:5] = stateNormal(state_[0:5])
                    state_ = np.array(state_)
                    state_ = torch.from_numpy(state_).float()

                    co2 = traci.vehicle.getCO2Emission(vehicle_id)
                    fuel = traci.vehicle.getFuelConsumption(vehicle_id)
                    ep_co2 += co2
                    ep_fuel += fuel
                    state = state_

                    if cur_light != light:
                        done = 1
                        light = cur_light
                    if done == 1:
                        traci.vehicle.setSpeed(vehicle_id, args.vmax)
                        break
        # rewardScaling.reset()
        while vehicle_id in traci.vehicle.getIDList() and len(traci.vehicle.getRoadID(vehicle_id)) != 0:
            co2 = traci.vehicle.getCO2Emission(vehicle_id)
            fuel = traci.vehicle.getFuelConsumption(vehicle_id)
            ep_co2 += co2
            ep_fuel += fuel
            traci.simulationStep()

        soup = bs4.BeautifulSoup(open(
            'E:/Program Files/PyWorkstation/AF_GLOSA/sumo/tripinfo.tr'))
        soup = soup.tripinfos
        info = {}
        for child in soup.children:
            if child.name == 'tripinfo':
                if child.attrs['id'] == '0':
                    info = child.attrs
        write.writerow(
            {'duration': info['duration'], 'waitingtime': info['waitingtime'],
             'waitingcount': info['waitingcount'],
             'timeloss': info['timeloss'],
             'co2': ep_co2, 'fuel': ep_fuel})

        traci.close()

if __name__ == '__main__':
    logPath = 'runs/hppo/log031501'
    writer = Writer(logPath)
    args = parse_args()
    testHPPO()
