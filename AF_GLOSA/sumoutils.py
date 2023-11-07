import traci
import os
from sumolib import checkBinary
import random
random.seed(1)
def startSumo(configFile, tripFile, option_nogui):
    if option_nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    sumoCmd = [sumoBinary, "-c", configFile, '--tripinfo-output', tripFile]
    traci.start(sumoCmd)
def stepSumo():
    traci.simulationStep()
def generateCVRoute():
    file_name = "E:/Program Files/PyWorkstation/AF_GLOSA/sumo/inter2.rou.xml"
    if os.path.exists(file_name):
        os.remove(file_name)
    depart_time = random.randint(0, 30)
    # depart_time = random.randint(30, 50)
    print('depart_time:', depart_time)

    with open(file_name, 'w') as f:
        f.write(
            '<routes>\n<vType id="type0" sigma="0" length="5" maxSpeed="11" minGap="2" />\n'
            '<route id="route0" edges="E15 E16 E19 E14 -E14 -E12 E20"/>\n' 
            '<vehicle id="0" type="type0" route="route0" depart="'
            ) #E15 E16 E19 E14 -E14 -E12 E20 #  E0 E1
        f.write(str(depart_time))
        f.write('" color="red" />\n'
                '</routes>')
    f.close()
    return depart_time

def loadSumo(configFile, tripFile, option_nogui):
    depart_time = generateCVRoute()
    startSumo(configFile, tripFile, option_nogui)
    # traci.setOrder(0)
    traci.load(['-c', configFile, '--start', '--tripinfo-output', tripFile])
    traci.simulationStep(depart_time)

def closeSumo():
    traci.close()

