from lib.sensorPosition import Sensor
from lib.sensorPosition import SensorNet
from lib.sensorPosition import SensorPosition

sensorNet = SensorNet()

for i in range(16):
    if i<4:
        plane = 1
    elif i<8:
        plane = 2
    elif i<12:
        plane = 3
    else:
        plane = 4

    sensor = Sensor(i+1, plane, SensorPosition[i+1])
    sensorNet.addSensor(sensor)

for s in sensorNet.Sensor:
    print(s.id)



