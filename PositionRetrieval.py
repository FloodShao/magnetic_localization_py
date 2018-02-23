import scipy.io as sio
from lib.magnetPosition import magnetPos_from_setup
from lib.LS_optimization import *
import numpy as np
from lib.PreProcessingParam import *
from scipy.optimize import least_squares


Bt = sio.loadmat("LabParam0222.mat")['Bt']
sensorParam = sio.loadmat("LabParam0222.mat")['sensorParam']

Idx = [0]

matdir = "./data/datamat0206/Geo_Mag.mat"
data_cal = sio.loadmat(matdir)['data']
#print(data_cal)

for i in Idx:
    matdir = "./data/datamat0206/" + datafile0206_dir[i] + ".mat"
    data = sio.loadmat(matdir)['data']
#data = data-data_cal
#print(data)

#magnetPos = magnetPos_from_setup(p_m_original[ Idx[0] ])
#print(magnetPos)


magnetPos_init = np.array([0., 0., 0., 0., 0.])
for i in range(data.shape[0]):

    bounds = ([-190e-3, -120e-3, -140e-3, 0., 0.], [190e-3, 120e-3, 140e-3, math.pi, 2*math.pi])

    res = least_squares(MagnetPosError_5d, magnetPos_init, verbose=0, bounds=bounds, jac=MagnetPosError_5d_jac, ftol=1e-10, xtol=1e-8, method='trf',
                        args=(Bt, sensorParam, data[i]))

    print(res.x)


print(p_m_original[Idx[0]])