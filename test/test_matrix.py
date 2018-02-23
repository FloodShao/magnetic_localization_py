import numpy as np
import math
import scipy.io as sio
from lib.magnetPosition import magnetPos_from_setup
from lib.PreProcessingParam import *
from lib.util import *
from lib.magneticModel import *


sensor_param = sio.loadmat("../LabParam0222.mat")['sensorParam']
Bt = sio.loadmat("../LabParam0222.mat")['Bt']

magnet_pos = np.array([0., 0., 0., 0., 0.])

x = magnet_pos[0]
y = magnet_pos[1]
z = magnet_pos[2]
theta = magnet_pos[3]
phi = magnet_pos[4]

magnetPos = magnetPos_from_setup(magnet_pos)

sensorPos = sensor_param[:, 0:3]

a = sensorPos[:, 0] - magnetPos[0, 0]
b = sensorPos[:, 1] - magnetPos[0, 1]
c = sensorPos[:, 2] - magnetPos[0, 2]

m, n, p = magnetPos[0, 3], magnetPos[0, 4], magnetPos[0, 5]

Blx_a = (9*m*a + 3*n*b + 3*p*c)*(a**2 + b**2 + c**2)**(-5/2) - 5*a*(3*m*a**2 + 3*n*b*c + 3*p*c*a)*(a**2 + b**2 + c**2)**(-7/2)
Bly_a = (3*m*b + 3*n*a)*(a**2 + b**2 + c**2)**(-5/2) - 5*a*(3*m*a*b + 3*n*b**2 + 3*p*c*b)*(a**2 + b**2 + c**2)**(-7/2)
Blz_a = (3*m*c + 3*p*a)*(a**2 + b**2 + c**2)**(-5/2) - 5*a*(3*m*a*c + 3*n*b*c + 3*p*c**2)*(a**2 + b**2 + c**2)**(-7/2)

Blx_b = (3*m*a + 3*m*b)*(a**2 + b**2 + c**2)**(-5/2) - 5*b*(3*m*a**2 + 3*n*b*c + 3*p*a*c)*(a**2 + b**2 + c**2)**(-7/2)
Bly_b = (3*m*a + 9*n*b + 3*p*c)*(a**2 + b**2 + c**2)**(-5/2) - 5*b*(3*m*a*b + 3*n*b**2 + 3*p*c*b)*(a**2 + b**2 + c**2)**(-7/2)
Blz_b = (3*n*c + 3*p*b)*(a**2 + b**2 + c**2)**(-5/2) - 5*b*(3*m*a*c + 3*n*b*c + 3*p*c**2)*(a**2 + b**2 + c**2)**(-7/2)

Blx_c = (3*p*a + 3*m*c)*(a**2 + b**2 + c**2)**(-5/2) - 5*c*(3*m*a**2 + 3*n*b*a + 3*p*c*a)*(a**2 + b**2 + c**2)**(-7/2)
Bly_c = (3*p*b + 3*m*c)*(a**2 + b**2 + c**2)**(-5/2) - 5*c*(3*m*a*b + 3*n*b**2 + 3*p*c*b)*(a**2 + b**2 + c**2)**(-7/2)
Blz_c = (3*m*a + 3*n*b + 9*p*c)*(a**2 + b**2 + c**2)**(-5/2) - 5*c*(3*m*a*c + 3*n*b*c + 3*p*c**2)*(a**2 + b**2 + c**2)**(-7/2)

Blx_m = (2*a**2 - b**2 - c**2) * (a**2 + b**2 + c**2)**(-5/2)
Bly_m = (3*a*b) * (a**2 + b**2 + c**2)**(-5/2)
Blz_m = (3*a*c) * (a**2 + b**2 + c**2)**(-5/2)

Blx_n = (3*a*b) * (a**2 + b**2 + c**2)**(-5/2)
Bly_n = (2*b**2 - a**2 - c**2) * (a**2 + b**2 + c**2)**(-5/2)
Blz_n = (3*b*c) * (a**2 + b**2 + c**2)**(-5/2)

Blx_p = (3*a*c) * (a**2 + b**2 + c**2)**(-5/2)
Bly_p = (3*b*c) * (a**2 + b**2 + c**2)**(-5/2)
Blz_p = (2*c**2 - b**2 - a**2) * (a**2 + b**2 + c**2)**(-5/2)

a_x = -1
b_y = -1
c_z = -1

m_theta = math.cos(phi) * math.cos(theta)
n_theta = math.sin(phi) * math.cos(theta)
p_theta = -math.sin(theta)

m_phi = -math.sin(theta) * math.sin(phi)
n_phi = math.sin(theta) * math.cos(phi)
p_phi = 0


'''Theoretical magnetic value'''
matdir = "../data/dataMat0206/" + datafile0206_dir[0] + ".mat"
data = sio.loadmat(matdir)['data']


Measured_data = data[0].reshape((16, 3))

sensor_pos = sensor_param[:, 0:3]
sensor_rotation = sensor_param[:, 3:6]
sensor_rotation_matrix = []
for r_v in sensor_rotation:
    sensor_rotation_matrix.append(np.dot(yaw_matrix(r_v[2]), pitch_matrix(r_v[1]), roll_matrix(r_v[0])))
sensor_rotation_matrix = np.array(sensor_rotation_matrix)


theo_data_init = M_field_value_model(magnetPos[0], sensor_pos)
Theorectical_data = []
for i in range(theo_data_init.shape[0]):
    Theorectical_data.append(np.dot(theo_data_init[i], sensor_rotation_matrix[i]).tolist())
Theorectical_data = 1e7 * Bt * np.array(Theorectical_data)  # change to mG unit

Error_matrix = Theorectical_data - Measured_data

print(Error_matrix)

A = np.vstack((Blx_a*a_x, Bly_a*a_x, Blz_a*a_x)).transpose()
print(A)

print(sum(sum(A*Error_matrix)))

