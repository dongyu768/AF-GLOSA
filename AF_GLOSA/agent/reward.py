from utils import *

def reward_func(valid): # valid, beforeTime, afterTime
    vehicle_id = getControledVeh()
    r1 = traci.vehicle.getFuelConsumption(vehicle_id)
    v_t = traci.vehicle.getSpeed(vehicle_id)
    if v_t <= 0.1:
        r2 = -500
        print('stop------------------------')
    else:
        r2 = 0
    reward_sc = - 0.1 * r1 + 0.6 * r2 + valid * 10
    print(reward_sc)
    # reward_sc = - 0.1 * r1 + 0.6 * r2 + valid * 10 + (beforeTime - afterTime)
    # reward_sc = - 0.1 * r1 + 0.4 * r2 + valid * 10 + 0.5 * (beforeTime - afterTime)
    # reward_sc = - 0.1 * r1 + 0.8 * r2 + valid * 10
    # reward_sc = reward_sc * (afterTime - beforeTime)
    return reward_sc

