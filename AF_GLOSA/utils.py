import traci
from torch.utils.tensorboard import SummaryWriter
from args import parse_args

args = parse_args()

def Writer(logpath):
    writer = SummaryWriter(logpath)
    return writer

def calCycleTime(tls_id):
    tls_logic = traci.trafficlight.getAllProgramLogics(tls_id)
    cycle_time = 0
    for i in range(len(tls_logic[0].phases)):
        cycle_time += tls_logic[0].phases[i].duration
    return cycle_time

def calSpeedGuideLength(v_max, v_min, a, d, tr, c):
    Lg_max = c * v_min
    Lg_min = max((v_max**2-v_min**2)/2*a + v_min*tr, (v_max**2-v_min**2)/2*d + v_max*tr)
    l_g = (Lg_max + Lg_min)/2
    return l_g

def getIntersectionInfo(vehicle_id):
    '''
    :return: intersection_info
    '''
    intersection_info = traci.vehicle.getNextTLS(vehicle_id)
    return intersection_info

def getL_t(vehicle_id):
    '''
    :return: l_t
    '''
    intersection_info = getIntersectionInfo(vehicle_id)
    if len(intersection_info) == 0:
        l_t = 301
        return l_t
    l_t = intersection_info[0][2]
    return l_t
def getAheadTLS(vehicle_id):
    intersection_info = getIntersectionInfo(vehicle_id)
    if len(intersection_info) != 0:
        TLS_id = intersection_info[0][0]
    else:
        TLS_id = 'None'
    return TLS_id

def getControledVeh():
    '''
    :return: vehicle_id
    '''
    # vehicle_idList = traci.vehicle.getIDList()
    # return vehicle_idList[0]
    vehicle_id = '0'
    return vehicle_id

def getDecisionEpoch(vehicle_id):
    '''
    :param vehicle_id:
    :return: epochs
    '''
    # while vehicle_id == '0':
    #     vehicle_id = traci.vehicle.getIDList()[-1]
    intersection_info = getIntersectionInfo(vehicle_id)
    new_list = []
    for i in range(len(intersection_info)):
        if intersection_info[i][0] not in new_list:
            new_list.append(intersection_info[i][0])
    epochs = len(new_list)
    return epochs

def getSpeed(vehicle_id):
    v = traci.vehicle.getSpeed(vehicle_id)
    return v

def getRemainTime(vehicle_id):
    intersection_info = getIntersectionInfo(vehicle_id)
    if len(intersection_info) == 0:
        m_t = 3
        return m_t
    m_t = traci.trafficlight.getNextSwitch(intersection_info[0][0]) - traci.simulation.getTime()
    return m_t
def getDuration(vehicle_id):
    '''
    :param vehicle_id:
    :return:
    '''
    intersection_info = getIntersectionInfo(vehicle_id)
    if len(intersection_info) == 0:
        g_duration = 20
        r_duration = 20
        return g_duration, r_duration
    tls_logic = traci.trafficlight.getAllProgramLogics(intersection_info[0][0])
    g_duration = tls_logic[0].phases[0].duration
    r_duration = tls_logic[0].phases[1].duration
    return g_duration, r_duration

def getQuenceLength():
    det_id = traci.lanearea.getIDList()[0]
    q_t = traci.lanearea.getJamLengthMeters(det_id)
    return q_t

# def getVehNumOfQuence():
#     det_id = traci.lanearea.getIDList()[0]
#     n_t = traci.lanearea.getJamLengthVehicle(det_id)
#     return n_t

def getVehNumOfQuence():
    vehicle_id = getControledVeh()
    count = 0
    while traci.vehicle.getLeader(vehicle_id, 2.5) != None:
        count += 1
        vehicle_id = traci.vehicle.getLeader(vehicle_id, 2.5)[0]
    return count

def doneAction(a_dis, a_con):
    import torch
    vehicle_id = getControledVeh()
    valid = 0
    v0 = traci.vehicle.getSpeed(vehicle_id)

    intersction_info = getIntersectionInfo(vehicle_id)
    if len(intersction_info) != 0:
        p = intersction_info[0][3]
    else:
        p = 'G'
    if p == 'R' or p == 'r':
        p_t = 1  #
    else:
        p_t = 0
    # if v0 <= 0.1:
    #     return valid
    if v0 <= 0.1 and p_t and a_dis:
        valid = -4
        return valid
    if a_dis == 1:
        vt = v0 + a_con
        # if a_con < 0.1 or a_con > -0.1:
        #     valid = -1
        # if v0 <= 0.1 and p_t:
        #     valid = -4
        if vt >= args.vmin and vt <= args.vmax:
            # vt = torch.round(vt)
            # valid = 5
            traci.vehicle.setSpeed(vehicle_id, vt)
        else:
            valid = -2
    else:
        # valid = 2
        # valid = 4
        valid = 4
        if v0 < args.vmin:
            valid = -2
    return valid

def getTravelDistance(vehicle_id):
    distance = traci.vehicle.getDistance(vehicle_id)
    return distance
