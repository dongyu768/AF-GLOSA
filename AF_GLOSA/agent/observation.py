from utils import *

def getState():
    vehicle_id = getControledVeh()
    v_t = getSpeed(vehicle_id)
    l_t = getL_t(vehicle_id)
    intersction_info = getIntersectionInfo(vehicle_id)
    green, red =getDuration(vehicle_id)
    if len(intersction_info) != 0:
        p = intersction_info[0][3]
    else:
        p = 'G'
    if p == 'G' or p == 'g':
        p_t = 0 # green
    else:
        p_t = 1  # red
    m_t = getRemainTime(vehicle_id)
    if p_t == 0:
        w_t = m_t + red
    else:
        w_t = m_t
    # lane = traci.vehicle.getLaneIndex(vehicle_id)
    # tar_t = l_t / (v_t + 0.1)
    a_t = traci.vehicle.getAcceleration(vehicle_id)
    # leader = traci.vehicle.getLeader(vehicle_id, 30)
    # if leader == None:
    #     lea_v = -1
    #     lea_d = -1
    # else:
    #     lea_v = traci.vehicle.getSpeed(leader[0])
    #     lea_d = leader[1]
    # # state = [l_t, v_t, a_t, m_t, w_t, lea_v, lea_d, p_t]
    state = [l_t, v_t, a_t, m_t, w_t, p_t]
    return state