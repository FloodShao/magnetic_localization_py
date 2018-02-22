from lib.magneticModel import M_field_value_model
import numpy as np
from  lib.util import *
from scipy.optimize import least_squares

def BTerror(Bt, sensor_pos, magnet_pos, Measured_data):
    """
    This function serve as a general error function for optimize BT
    :param BT:
    :param sensor_pos: (n, 3) array
    :param magnet_pos: (n_m, 6) array
    :param Measured_data: (n*N_m, 3) array
    :return: float error value
    """

    Theorectical_data = []
    if len(magnet_pos.shape) == 1:
        magnet_pos = np.array([magnet_pos])

    for mp in magnet_pos:
        theo_data = M_field_value_model(mp, sensor_pos)
        Theorectical_data += theo_data.tolist()

    Theorectical_data = 1e7 * Bt * np.array(Theorectical_data) #change to 'mG' unit

    E_Matrix = (Measured_data  - Theorectical_data).transpose()

    Error = sum(E_Matrix[0]**2) + sum(E_Matrix[1]**2) + sum(E_Matrix[2]**2)

    return Error


def SensorPosError(sensor_pos, Bt, magnet_pos, Measured_data):
    """

    :param sensor_pos: must have at most 1 dimension
    :param Bt:
    :param magnet_pos:
    :param Measured_data:
    :return:
    """

    sensor_pos = sensor_pos.reshape( (16, 3) )

    Theorectical_data = []
    if len(magnet_pos.shape) == 1:
        magnet_pos = np.array([magnet_pos])

    for mp in magnet_pos:
        theo_data = M_field_value_model(mp, sensor_pos)
        Theorectical_data += theo_data.tolist()

    Theorectical_data = 1e7 * Bt * np.array(Theorectical_data)  # change to 'mG' unit

    E_Matrix = (Measured_data - Theorectical_data).transpose()

    Error = sum(E_Matrix[0] ** 2) + sum(E_Matrix[1] ** 2) + sum(E_Matrix[2] ** 2)

    return Error


def SensorOriError(sensor_rotation, Bt, sensor_pos, magnet_pos, Measured_data):
    """

    :param sensor_rotation: (1*48) sensors rotation angle
    :param Bt:
    :param sensor_pos:
    :param magnet_pos:
    :param Measured_data:
    :return:
    """

    sensor_rotation = sensor_rotation.reshape((16, 3))
    sensor_rotation_matrix = []
    for r_v in sensor_rotation:
        sensor_rotation_matrix.append(np.dot( yaw_matrix(r_v[2]), pitch_matrix(r_v[1]), roll_matrix(r_v[0]) ))
    sensor_rotation_matrix = np.array(sensor_rotation_matrix)


    Theorectical_data = []
    if len(magnet_pos.shape) == 1:
        magnet_pos = np.array([magnet_pos])

    for mp in magnet_pos:
        theo_data = M_field_value_model(mp, sensor_pos)
        for i in range(theo_data.shape[0]):
            Theorectical_data.append( np.dot( theo_data[i], sensor_rotation_matrix[i] ).tolist() )
    Theorectical_data = 1e7 * Bt * np.array(Theorectical_data)  # change to 'mG' unit

    E_Matrix = (Measured_data - Theorectical_data).transpose()

    Error = sum(E_Matrix[0] ** 2) + sum(E_Matrix[1] ** 2) + sum(E_Matrix[2] ** 2)

    return Error


def SensorError(sensor_param, Bt, magnet_pos, Measured_data):
    """

    :param sensor_param: including sensor_pos[0:3] and sensor_orientation[3:6],
    :param Bt:
    :param magnet_pos:
    :param Measured_data:
    :return:
    """
    sensor_param = sensor_param.reshape( (16, 6) )
    sensor_pos = sensor_param[:, 0:3]
    sensor_rotation = sensor_param[:, 3:6]

    sensor_rotation_matrix = []
    for r_v in sensor_rotation:
        sensor_rotation_matrix.append( np.dot( yaw_matrix(r_v[2]), pitch_matrix(r_v[1]), roll_matrix(r_v[0]) ))
    sensor_rotation_matrix = np.array(sensor_rotation_matrix)


    Theorectical_data = []
    if len(magnet_pos.shape) == 1:
        magnet_pos = np.array([magnet_pos])

    for mp in magnet_pos:
        theo_data = M_field_value_model(mp, sensor_pos)
        for i in range(theo_data.shape[0]):
            Theorectical_data.append( np.dot( theo_data[i], sensor_rotation_matrix[i] ).tolist() )
    Theorectical_data = 1e7 * Bt * np.array(Theorectical_data)  # change to 'mG' unit

    E_Matrix = (Measured_data - Theorectical_data).transpose()

    Error = sum(E_Matrix[0] ** 2) + sum(E_Matrix[1] ** 2) + sum(E_Matrix[2] ** 2)

    return Error


def MagnetPosError(magnet_pos, Bt, sensor_param, Measured_data):

    """
    The function provide the error of one magnet_pos
    :param magnet_pos: 1*6 magnet position and orientation vector (x, y, z, m ,n, p)
    :param Bt:
    :param sensor_param: 16*6 sensor position and orientation matrix
    :param Measured_data:
    :return:
    """
    sensor_pos = sensor_param[:, 0:3]
    sensor_rotation = sensor_param[:, 3:6]
    sensor_rotation_matrix = []
    for r_v in sensor_rotation:
        sensor_rotation_matrix.append( np.dot( yaw_matrix(r_v[2]), pitch_matrix(r_v[1]), roll_matrix(r_v[0]) ))
    sensor_rotation_matrix = np.array(sensor_rotation_matrix)


    theo_data_init = M_field_value_model(magnet_pos, sensor_pos)
    Theorectical_data = []
    for i in range(sensor_pos.shape[0]):
        Theorectical_data.append( np.dot(theo_data_init[i], sensor_rotation_matrix[i]).tolist() )
    Theorectical_data = 1e7 * Bt * np.array(Theorectical_data)  #change to mG unit

    E_Matrix = (Measured_data - Theorectical_data).transpose()
    Error = sum(E_Matrix[0] ** 2) + sum(E_Matrix[1] ** 2) + sum(E_Matrix[2] ** 2)

    return Error







