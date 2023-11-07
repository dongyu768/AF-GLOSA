from utils import *
from sumoutils import *
from agent.observation import getState
from agent.reward import reward_func
from args import parse_args
import sys
args = parse_args()
class SumoEnv(object):
    def __init__(self, configFile, tripFile, option_nogui):
        self.configFile = configFile
        self.tripFile = tripFile
        self.option_nogui = option_nogui
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

    def reset(self):
        state = getState()
        return state

    def step(self, a_dis, a_con): #a_dis, a_con, beforeTime
        # control_step = 3
        valid = doneAction(a_dis, a_con)
        # afterTime = traci.simulation.getTime()
        for i in range(args.control_step):
            traci.simulationStep()
        afterTime = traci.simulation.getTime()
        # reward = reward_func(valid, beforeTime, afterTime)
        reward = reward_func(valid)
        state_ = getState()
        return reward, state_

    def close(self):
        traci.close()

