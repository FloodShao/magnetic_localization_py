import scipy.io as sio
import numpy as np
from lib.sensorPosition import Sensor
from lib.sensorPosition import SensorNet
from lib.sensorPosition import SensorPosition
from lib.magnetPosition import magnetPos_from_setup
import math
from lib.LS_optimization import *
import lib.PreProcessingParam

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

'''Load measured data'''
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

datafile_dir = lib.PreProcessingParam.datafile0206_dir

for i in range(23):
    dir = './data/Data0206/' + datafile_dir[i] + '.txt'
    data = np.loadtxt(dir)
    data_num = int(data.shape[0] / 48)
    data = data.reshape((data_num, 48))
    data = sensorNet.sensorValue_world(data)
    matdir = './data/dataMat0206/noncal/' + datafile_dir[i] + '.mat'
    sio.savemat(matdir, {'data': data})
    '''minus the geology magnetic field'''
    data = data - Geo_Mag
    matdir = './data/dataMat0206/aftercal/' + datafile_dir[i] + '.mat'
    sio.savemat(matdir, {'data' : data})


'''Configure magnet position'''
l_1 = 20e-3
R_2 = 40e-3
R_3 = 60e-3
R_4 = 80e-3
R_5 = 100e-3
h = 15e-3

delta = 2 * math.pi / 24
circle_uxy = []
for i in range(24):
    circle_uxy.append([math.sin(i * delta), math.cos(i * delta)])
circle_uxy = np.array(circle_uxy)

p_xy = {
    '1' : np.array([
        [0., l_1],
        [l_1, l_1],
        [l_1, 0.],
        [l_1, -l_1],
        [0., -l_1],
        [-l_1, -l_1],
        [-l_1, 0.],
        [-l_1, l_1]
    ]),

    '2' : R_2 * circle_uxy,

    '3' : R_3 * circle_uxy,

    '4' : R_4 * circle_uxy,

    '5' : R_5 * circle_uxy,

    'c_phi0' : np.array([
        [0., R_5], [0., R_4], [0., R_3], [0., R_2], [0., l_1], [0., -l_1], [0., -R_2], [0., -R_3], [0., -R_4], [0., -R_5],
        [-R_5, 0.], [-R_4, 0.], [-R_3, 0.], [-R_2, 0.], [-l_1, 0.], [l_1, 0.], [R_2, 0.], [R_3, 0.], [R_4, 0.], [R_5, 0.]
    ]),

    'c_phi90' : np.array([
        [0., R_4], [0., R_3], [0., R_2], [0., l_1], [0., -l_1], [0., -R_2], [0., -R_3], [0., -R_4],
        [-R_5, 0.], [-R_4, 0.], [-R_3, 0.], [-R_2, 0.], [-l_1, 0.], [l_1, 0.], [R_2, 0.], [R_3, 0.], [R_4, 0.], [R_5, 0.]
    ])

}

'''1_8_z_-1_(90,90)的phi应该为-90'''
'''cross_z_0_(90,0)的phi应该为180'''
'''cross_z_0_(90,90)的phi应该为-90'''
'''cross_z_-1_(90,90)的phi应该为-90'''

p_m_original = {

    0: np.hstack((  p_xy['1'],      np.zeros(( p_xy['1'].shape[0], 1)),  np.zeros(( p_xy['1'].shape[0], 1)),
                    np.zeros(( p_xy['1'].shape[0], 1)) )),
    1: np.hstack((p_xy['1'], np.zeros((p_xy['1'].shape[0], 1)), 90 * np.ones((p_xy['1'].shape[0], 1)),
                  np.zeros((p_xy['1'].shape[0], 1)))),
    2: np.hstack((p_xy['1'], np.zeros((p_xy['1'].shape[0], 1)), 90 * np.ones((p_xy['1'].shape[0], 1)),
                  90 * np.ones((p_xy['1'].shape[0], 1)))),
    3: np.hstack((p_xy['1'], -h * np.ones((p_xy['1'].shape[0], 1)), np.zeros((p_xy['1'].shape[0], 1)),
                  np.zeros((p_xy['1'].shape[0], 1)))),
    4: np.hstack((p_xy['1'], -h * np.ones((p_xy['1'].shape[0], 1)), 90 * np.ones((p_xy['1'].shape[0], 1)),
                  np.zeros((p_xy['1'].shape[0], 1)))),
    5: np.hstack((p_xy['1'], -h * np.ones((p_xy['1'].shape[0], 1)), 90 * np.ones((p_xy['1'].shape[0], 1)),
                  -90 * np.ones((p_xy['1'].shape[0], 1)))),
    6: np.hstack((p_xy['2'], np.zeros((p_xy['2'].shape[0], 1)), np.zeros((p_xy['2'].shape[0], 1)),
                  np.zeros((p_xy['2'].shape[0], 1)))),
    7: np.hstack((p_xy['2'], -h * np.ones((p_xy['2'].shape[0], 1)), np.zeros((p_xy['2'].shape[0], 1)),
                  np.zeros((p_xy['2'].shape[0], 1)))),
    8: np.hstack((p_xy['3'], np.zeros((p_xy['3'].shape[0], 1)), np.zeros((p_xy['3'].shape[0], 1)),
                  np.zeros((p_xy['3'].shape[0], 1)))),
    9: np.hstack((p_xy['3'], -h * np.ones((p_xy['3'].shape[0], 1)), np.zeros((p_xy['3'].shape[0], 1)),
                  np.zeros((p_xy['3'].shape[0], 1)))),
    10: np.hstack((p_xy['4'], np.zeros((p_xy['4'].shape[0], 1)), np.zeros((p_xy['4'].shape[0], 1)),
                   np.zeros((p_xy['4'].shape[0], 1)))),
    11: np.hstack((p_xy['4'], -h * np.ones((p_xy['4'].shape[0], 1)), np.zeros((p_xy['4'].shape[0], 1)),
                   np.zeros((p_xy['4'].shape[0], 1)))),
    12: np.hstack((p_xy['c_phi0'], np.zeros((p_xy['c_phi0'].shape[0], 1)), 90 * np.ones((p_xy['c_phi0'].shape[0], 1)),
                   180*np.ones((p_xy['c_phi0'].shape[0], 1)))),
    13: np.hstack((p_xy['c_phi90'], np.zeros((p_xy['c_phi90'].shape[0], 1)), 90 * np.ones((p_xy['c_phi90'].shape[0], 1)),
                  -90 * np.ones((p_xy['c_phi90'].shape[0], 1)))),
    14: np.hstack((p_xy['c_phi0'], -h * np.zeros((p_xy['c_phi0'].shape[0], 1)),
                   90 * np.ones((p_xy['c_phi0'].shape[0], 1)), np.zeros((p_xy['c_phi0'].shape[0], 1)))),
    15: np.hstack((p_xy['c_phi90'], -h * np.zeros((p_xy['c_phi90'].shape[0], 1)),
                   90 * np.ones((p_xy['c_phi90'].shape[0], 1)), -90 * np.ones((p_xy['c_phi90'].shape[0], 1)))),

}

p_m_original = lib.PreProcessingParam.p_m_original

print(len(p_m_original))

p_m_actual = {}
for i in range(23):
    p_m_actual[i] = magnetPos_from_setup(p_m_original[i])

'''calibration Bt'''
Idx = [0,1,2,3,4,5,6,7,8,9]
#Idx = [0, 3, 6, 7, 8, 9, 10, 11]
#Idx = [12, 13, 14, 15]
#Idx = [7, 8, 9, 10, 11]
#Idx = [0, 3]
#Idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

magnetPos = np.vstack((
    p_m_actual[Idx[0]],
    p_m_actual[Idx[1]],
    p_m_actual[Idx[2]],
    p_m_actual[Idx[3]],
    p_m_actual[Idx[4]],
    p_m_actual[Idx[5]],
    p_m_actual[Idx[6]],
    p_m_actual[Idx[7]],
    p_m_actual[Idx[8]],
    p_m_actual[Idx[9]],
#    p_m_actual[Idx[10]],
#    p_m_actual[Idx[11]],
#    p_m_actual[Idx[12]],
#    p_m_actual[Idx[13]],
#    p_m_actual[Idx[14]],
#    p_m_actual[Idx[15]],
))


M_meas = []
for i in Idx:
    data = sio.loadmat("./data/dataMat0206/noncal/" + datafile_dir[i] + ".mat")['data']
    '''from n*48 to n*16*3'''
    data = data.reshape((data.shape[0], 16, 3))
    for d in data:
        M_meas += d.tolist()
M_meas = np.array(M_meas)

"""Calibrate Bt"""
#print(M_meas)
Bt0 = 1e-5
sensorPos = sensorParam[:, 0:3]
res = least_squares(BTerror, Bt0, verbose=2, ftol=1e-10, xtol=1e-12, method='lm', args=(sensorPos, magnetPos, M_meas) )

Bt = res.x[0]
print("Bt = ", Bt)
M_theo = []
for p in magnetPos:
    M_filed_theo = M_field_value_model(p, sensorPos)
    M_theo.extend(M_filed_theo.tolist())
M_theo = Bt * 1e7 * np.array(M_theo) #change to 'mG' unit
#print(M_theo)


"""Calibrate sensor rotations"""
#sensor_rotation = np.zeros((48,))
#res = least_squares(SensorOriError, sensor_rotation, verbose=2, ftol = 1e-2, xtol=1e-5, method='trf', args=(Bt, sensorPos, magnetPos, M_meas))
#sensor_rotation = res.x.reshape( (16,3) )

"""Calibrate sensor position"""
#res = least_squares(SensorPosError, sensorPos.ravel(), verbose=2, ftol=1e-2, xtol=1e-5, method='trf', args=(Bt, magnetPos, M_meas) )
#sensorPos = res.x.reshape( (16,3) )

"""Calibrate sensor position and rotations at the same time"""
res = least_squares(SensorError, sensorParam.ravel(), verbose=2, ftol=1e-2, xtol=1e-5, method='trf', args=(Bt, magnetPos, M_meas))
sensor_param = res.x.reshape( (16, 6))

'''updata the sensor position and orientation after calibration'''
for i in range(sensor_param.shape[0]):
    sp = sensor_param[i]
    sensorNet.Sensor[i].updateSensor( position=sp[0:3], orientation=sp[3:6])


"""save the experimental parameters in mat file"""
sio.savemat("LabParam0222.mat",
            {"Bt": Bt,
             "sensorParam": sensor_param}
            )

bt = sio.loadmat("LabParam0222.mat")["Bt"]
print(bt)
sp = sio.loadmat("LabParam0222.mat")["sensorParam"]
print(sp)

