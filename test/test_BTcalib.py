from lib.sensorPosition import Sensor
from lib.sensorPosition import SensorNet
from lib.sensorPosition import SensorPosition
import numpy as np
from lib.magneticModel import *
from lib.magnetPosition import *
from scipy.optimize import least_squares
import scipy.io as sio
from lib.LS_optimization import *

data = sio.loadmat("../data/0205/Measureddata0205.mat")['Measured_data']
print(data)

'''magnet position configuration'''
X = -20e-3
Y = -20e-3
Z = 0
theta = 0
phi = 0

magnetPos = magnetPos_from_setup(X, Y, Z, theta, phi)

'''Sensor position configuration'''

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

Measured_data = sensorNet.sensorValue_world(data[4, :])


sensorPos = sensorNet.sensorPos()

M_field_theo = M_field_value_model(magnetPos, sensorPos)

M_data = Measured_data.reshape((16, 3)) * 1e-3

res = least_squares(BTerror, 1e-5, verbose=2, ftol=1e-10, method='lm', args=(sensorPos, magnetPos, M_data) )


print(res.x)





