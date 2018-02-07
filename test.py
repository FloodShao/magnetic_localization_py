import numpy as np
import scipy.io as sio
from lib import planeConf

Measured_data = []

for i in range(8):
    file_dir = "./data/0205/" + str(i+1) + ".txt"
    data = np.loadtxt(file_dir)
    data = data[0:480]
    data = data.reshape((10, 48))
    data = np.average(data, 0)

    Measured_data.append(data.tolist())

print(np.array(Measured_data))

sio.savemat("Measureddata0205.mat", {'Measured_data':np.array(Measured_data)})

data = sio.loadmat("Measureddata0205.mat")['Measured_data']


'''
fh = np.loadtxt("./data/2.txt")

a = fh.reshape((15, 48))

plane1 = planeConf.PlaneT['plane1']
plane2 = planeConf.PlaneT['plane2']
plane3 = planeConf.PlaneT['plane3']
plane4 = planeConf.PlaneT['plane4']

num = 1
idx = 'plane' + str(num)
plane = planeConf.PlaneT[idx]
print(plane)


b = np.array([])

for vec in a:
    vec = vec.reshape((16, 3))
    vp1 = vec[0:4].transpose()
    vp2 = vec[4:8].transpose()
    vp3 = vec[8:12].transpose()
    vp4 = vec[12:16].transpose()

    v1 = np.dot(plane1, vp1)
    v2 = np.dot(plane2, vp2)
    v3 = np.dot(plane3, vp3)
    v4 = np.dot(plane4, vp4)

    v_w = np.hstack((v1, v2, v3, v4))


c =0

'''


