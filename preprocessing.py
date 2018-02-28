import numpy as np
from lib.sensorPosition import Sensor
from lib.sensorPosition import SensorNet
from lib.sensorPosition import SensorPosition
import scipy.io as sio


'''Sensors configuration'''
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

sensorParam = sensorNet.sensorParam()


'''Geology magnetic field'''
data_dir = "./data/Data0206/no_magnet.txt"
data = np.loadtxt(data_dir)
num = int(data.shape[0] / 48)
data = data.reshape((num, 48))
Geo_Mag = np.average(data, axis = 0)
Geo_Mag = sensorNet.sensorValue_world(Geo_Mag)
sio.savemat("./data/dataMat0206/Geo_Mag.mat", {'data' : Geo_Mag})
#print(Geo_Mag.reshape((16,3)))


'''Processing data'''
datafile_dir = {
    0: '1_8_z_0_(0,0)',
    1: '1_8_z_0_(90,0)',
    2: '1_8_z_0_(90,90)',
    3: '1_8_z_-1_(0,0)',
    4: '1_8_z_-1_(90,0)',
    5: '1_8_z_-1_(90,90)',
    6: '2_24_z_0_(0,0)',
    7: '2_24_z_-1_(0,0)',
    8: '3_24_z_0_(0,0)',
    9: '3_24_z_-1_(0,0)',
    10: '4_24_z_0_(0,0)',
    11: '4_24_z_-1_(0,0)',
    12: 'cross_z_0_(90,0)',
    13: 'cross_z_0_(90,90)',
    14: 'cross_z_-1_(90,0)',
    15: 'cross_z_-1_(90,90)',
}


for i in range(16):
    dir = './data/Data0206/' + datafile_dir[i] + '.txt'
    data = np.loadtxt(dir)
    data_num = int(data.shape[0] / 48)
    data = data.reshape((data_num, 48))
    data = sensorNet.sensorValue_world(data)
    '''minus the geology magnetic field'''
    #in real application we should not consider this calibration
    #data = data - Geo_Mag
    matdir = './data/dataMat0206_0227/' + datafile_dir[i] + '.mat'
    sio.savemat(matdir, {'data' : data})


