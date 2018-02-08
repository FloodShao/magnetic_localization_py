import numpy as np
from lib.planeConf import PlaneT

SensorPosition = {
    1: 1e-3 * np.array([190, 50, 50]),
    2: 1e-3 * np.array([190, 50, -50]),
    3: 1e-3 * np.array([190, -50, -50]),
    4: 1e-3 * np.array([190, -50, 50]),

    5: 1e-3 * np.array([50, -140, 50]),
    6: 1e-3 * np.array([50, -140, -50]),
    7: 1e-3 * np.array([-50, -140, -50]),
    8: 1e-3 * np.array([-50, -140, 50]),

    9: 1e-3 * np.array([-50, 140, 50]),
    10: 1e-3 * np.array([-50, 140, -50]),
    11: 1e-3 * np.array([50, 140, -50]),
    12: 1e-3 * np.array([50, 140, 50]),

    13: 1e-3 * np.array([-190, -50, 50]),
    14: 1e-3 * np.array([-190, -50, -50]),
    15: 1e-3 * np.array([-190, 50, -50]),
    16: 1e-3 * np.array([-190, 50, 50]),
}


class Sensor(object):

    def __init__(self, id=None, plane=None, position=None, orientation=None):
        """

        :param id: int
        :param plane: int, {1, 2, 3, 4}
        :param position: numpy vector
        :param orientation: numpy array
        """

        if id is None:
            print("[Error] Please add sensor id!")
            return 1
        else:
            self.id = id

        if plane is not None:
            if plane > 5:
                print("[Error] Invalid plane index!")
            else:
                self.planeT = PlaneT['plane' + str(plane)]
        else:
            print("[Error] False plane indicator!")
            return 1

        if position is None:
            self.position = np.array([0., 0., 0.])
        else:
            self.position = position

        if orientation is None:
            self.orientation = np.eye(3, dtype=float)
        else:
            self.orientation = orientation

    def updateSensor(self, plane=None, position=None, orientation=None):
        if plane is not None:
            if plane > 5:
                print("[Error] Invalid plane index!")
            else:
                self.planeT = PlaneT['plane' + str(plane)]

        if position is not None:
            self.position = position

        if orientation is not None:
            self.orientation = orientation


class SensorNet(object):

    def __init__(self):
        self.Sensor = []
        return

    def addSensor(self, Sensor):
        """
        add a Sensor in the SensorNet
        :param Sensor: class object Sensor
        :return:
        """
        self.Sensor.append(Sensor)

    def sensorPos(self):
        sensor_pos = []
        for s in self.Sensor:
            sensor_pos.append(s.position.tolist())

        return np.array(sensor_pos)

    def sensorValue_world(self, original_data):
        """

        :param original_data: n*48 original data
        :return: (1)world coordinates
        """
        if len(original_data.shape) == 1:
            original_data = original_data.reshape(1, original_data.shape[0])

        output_data = np.zeros(original_data.shape)
        for i in range(16):
            si_data = original_data[:, i*3:(i+1)*3]
            T = self.Sensor[i].planeT
            output_data[:, i*3:(i+1)*3] = np.matmul(T, si_data.transpose()).transpose()

        return output_data





