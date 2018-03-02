from lib.magneticModel import M_field_value_model
import numpy as np
from lib.util import *
from lib.magnetPosition import magnetPos_from_setup
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


def MagnetPosError_5d(magnet_pos_5d, Bt, sensor_param, Measured_data):

    """
    The function provide the error of one magnet_pos
    :param magnet_pos: 1*5 magnet position and orientation vector (x, y, z, theta, phi)
    :param Bt:
    :param sensor_param: 16*6 sensor position and orientation matrix
    :param Measured_data: 1*48 data
    :return:
    """
    Measured_data = Measured_data.reshape((16,3))

    sensor_pos = sensor_param[:, 0:3]
    sensor_rotation = sensor_param[:, 3:6]
    sensor_rotation_matrix = []
    for r_v in sensor_rotation:
        sensor_rotation_matrix.append( np.dot( yaw_matrix(r_v[2]), pitch_matrix(r_v[1]), roll_matrix(r_v[0]) ))
    sensor_rotation_matrix = np.array(sensor_rotation_matrix)

    magnet_pos = magnetPos_from_setup(magnet_pos_5d)[0]

    theo_data_init = M_field_value_model(magnet_pos, sensor_pos)
    Theorectical_data = []
    for i in range(theo_data_init.shape[0]):
        Theorectical_data.append( np.dot(theo_data_init[i], sensor_rotation_matrix[i]).tolist() )
    Theorectical_data = 1e7 * Bt * np.array(Theorectical_data)  #change to mG unit

    E_Matrix = (Measured_data - Theorectical_data).transpose()
    Error = sum(E_Matrix[0] ** 2) + sum(E_Matrix[1] ** 2) + sum(E_Matrix[2] ** 2)

    return Error


def MagnetPosError_5d_jac(magnet_pos_5d, Bt, sensor_param, Measured_data):

    theta = magnet_pos_5d[3]
    phi = magnet_pos_5d[4]

    '''Theoretical magnetic value'''
    Measured_data = Measured_data.reshape((16, 3))

    sensor_pos = sensor_param[:, 0:3]
    sensor_rotation = sensor_param[:, 3:6]
    sensor_rotation_matrix = []
    for r_v in sensor_rotation:
        sensor_rotation_matrix.append(np.dot(yaw_matrix(r_v[2]), pitch_matrix(r_v[1]), roll_matrix(r_v[0])))
    sensor_rotation_matrix = np.array(sensor_rotation_matrix)

    magnet_pos = magnetPos_from_setup(magnet_pos_5d)[0]

    theo_data_init = M_field_value_model(magnet_pos, sensor_pos)
    Theorectical_data = []
    for i in range(theo_data_init.shape[0]):
        Theorectical_data.append(np.dot(theo_data_init[i], sensor_rotation_matrix[i]).tolist())
    Theorectical_data = 1e7 * Bt * np.array(Theorectical_data)  # change to mG unit

    Error_matrix = Theorectical_data - Measured_data

    '''derivative part'''
    a = sensor_pos[:, 0] - magnet_pos[0]
    b = sensor_pos[:, 1] - magnet_pos[1]
    c = sensor_pos[:, 2] - magnet_pos[2]

    m, n, p = magnet_pos[3], magnet_pos[4], magnet_pos[5]

    Blx_a = (9 * m * a + 3 * n * b + 3 * p * c) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2) - 5 * a * (
            3 * m * a ** 2 + 3 * n * b * c + 3 * p * c * a) * (a ** 2 + b ** 2 + c ** 2) ** (-7 / 2)
    Bly_a = (3 * m * b + 3 * n * a) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2) - 5 * a * (
            3 * m * a * b + 3 * n * b ** 2 + 3 * p * c * b) * (a ** 2 + b ** 2 + c ** 2) ** (-7 / 2)
    Blz_a = (3 * m * c + 3 * p * a) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2) - 5 * a * (
            3 * m * a * c + 3 * n * b * c + 3 * p * c ** 2) * (a ** 2 + b ** 2 + c ** 2) ** (-7 / 2)

    Blx_b = (3 * m * a + 3 * m * b) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2) - 5 * b * (
            3 * m * a ** 2 + 3 * n * b * c + 3 * p * a * c) * (a ** 2 + b ** 2 + c ** 2) ** (-7 / 2)
    Bly_b = (3 * m * a + 9 * n * b + 3 * p * c) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2) - 5 * b * (
            3 * m * a * b + 3 * n * b ** 2 + 3 * p * c * b) * (a ** 2 + b ** 2 + c ** 2) ** (-7 / 2)
    Blz_b = (3 * n * c + 3 * p * b) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2) - 5 * b * (
            3 * m * a * c + 3 * n * b * c + 3 * p * c ** 2) * (a ** 2 + b ** 2 + c ** 2) ** (-7 / 2)

    Blx_c = (3 * p * a + 3 * m * c) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2) - 5 * c * (
            3 * m * a ** 2 + 3 * n * b * a + 3 * p * c * a) * (a ** 2 + b ** 2 + c ** 2) ** (-7 / 2)
    Bly_c = (3 * p * b + 3 * m * c) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2) - 5 * c * (
            3 * m * a * b + 3 * n * b ** 2 + 3 * p * c * b) * (a ** 2 + b ** 2 + c ** 2) ** (-7 / 2)
    Blz_c = (3 * m * a + 3 * n * b + 9 * p * c) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2) - 5 * c * (
            3 * m * a * c + 3 * n * b * c + 3 * p * c ** 2) * (a ** 2 + b ** 2 + c ** 2) ** (-7 / 2)

    Blx_m = (2 * a ** 2 - b ** 2 - c ** 2) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2)
    Bly_m = (3 * a * b) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2)
    Blz_m = (3 * a * c) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2)

    Blx_n = (3 * a * b) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2)
    Bly_n = (2 * b ** 2 - a ** 2 - c ** 2) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2)
    Blz_n = (3 * b * c) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2)

    Blx_p = (3 * a * c) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2)
    Bly_p = (3 * b * c) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2)
    Blz_p = (2 * c ** 2 - b ** 2 - a ** 2) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2)

    a_x = -1
    b_y = -1
    c_z = -1

    m_theta = math.cos(phi) * math.cos(theta)
    n_theta = math.sin(phi) * math.cos(theta)
    p_theta = -math.sin(theta)

    m_phi = -math.sin(theta) * math.sin(phi)
    n_phi = math.sin(theta) * math.cos(phi)
    p_phi = 0

    B_x = Bt*np.vstack((Blx_a*a_x, Bly_a*a_x, Blz_a*a_x)).transpose()
    B_y = Bt*np.vstack((Blx_b*b_y, Bly_b*b_y, Blz_b*b_y)).transpose()
    B_z = Bt*np.vstack((Blx_c*c_z, Bly_c*c_z, Blz_c*c_z)).transpose()
    B_theta = Bt*np.vstack((
        Blx_m*m_theta + Blx_n*n_theta + Blx_p*p_theta,
        Bly_m*m_theta + Bly_n*n_theta + Bly_p*p_theta,
        Blz_m*m_theta + Blz_n*n_theta + Blz_p*p_theta
    )).transpose()
    B_phi = Bt*np.vstack((
        Blx_m*m_phi + Blx_n*n_phi + Blx_p*p_phi,
        Bly_m*m_phi + Bly_n*n_phi + Bly_p*p_phi,
        Blz_m*m_phi + Blz_n*n_phi + Blz_p*p_phi
    )).transpose()

    Error_x = sum(sum(2 * Error_matrix * B_x))
    Error_y = sum(sum(2 * Error_matrix * B_y))
    Error_z = sum(sum(2 * Error_matrix * B_z))
    Error_theta = sum(sum(2 * Error_matrix * B_theta))
    Error_phi = sum(sum(2 * Error_matrix * B_phi))

    return np.array([0., 0., 0., Error_theta, Error_phi])


def MagnetPosError_6d(magnet_pos_6d, Bt, sensor_param, Measured_data):

    """
    The function provide the error of one magnet_pos
    :param magnet_pos: 1*6 magnet position and orientation vector (x, y, z, m, n, p)
    :param Bt:
    :param sensor_param: 16*6 sensor position and orientation matrix
    :param Measured_data: 1*48 data
    :return:
    """
    Measured_data = Measured_data.reshape((16,3))

    sensor_pos = sensor_param[:, 0:3]
    sensor_rotation = sensor_param[:, 3:6]
    sensor_rotation_matrix = []
    for r_v in sensor_rotation:
        sensor_rotation_matrix.append( np.dot( yaw_matrix(r_v[2]), pitch_matrix(r_v[1]), roll_matrix(r_v[0]) ))
    sensor_rotation_matrix = np.array(sensor_rotation_matrix)

    magnet_pos = magnet_pos_6d

    theo_data_init = M_field_value_model(magnet_pos, sensor_pos)
    Theorectical_data = []
    for i in range(theo_data_init.shape[0]):
        Theorectical_data.append( np.dot(theo_data_init[i], sensor_rotation_matrix[i]).tolist() )
    Theorectical_data = 1e7 * Bt * np.array(Theorectical_data)  #change to mG unit

    E_Matrix = (Measured_data - Theorectical_data).transpose()
    Error = sum(E_Matrix[0] ** 2) + sum(E_Matrix[1] ** 2) + sum(E_Matrix[2] ** 2)

    return Error


def MagnetPosError_6d_jac(magnet_pos_6d, Bt, sensor_param, Measured_data):

    '''Theoretical magnetic value'''
    Measured_data = Measured_data.reshape((16, 3))

    sensor_pos = sensor_param[:, 0:3]
    sensor_rotation = sensor_param[:, 3:6]
    sensor_rotation_matrix = []
    for r_v in sensor_rotation:
        sensor_rotation_matrix.append(np.dot(yaw_matrix(r_v[2]), pitch_matrix(r_v[1]), roll_matrix(r_v[0])))
    sensor_rotation_matrix = np.array(sensor_rotation_matrix)

    magnet_pos = magnet_pos_6d

    theo_data_init = M_field_value_model(magnet_pos, sensor_pos)
    Theorectical_data = []
    for i in range(theo_data_init.shape[0]):
        Theorectical_data.append(np.dot(theo_data_init[i], sensor_rotation_matrix[i]).tolist())
    Theorectical_data = 1e7 * Bt * np.array(Theorectical_data)  # change to mG unit

    Error_matrix = Theorectical_data - Measured_data

    '''derivative part'''
    a = sensor_pos[:, 0] - magnet_pos[0]
    b = sensor_pos[:, 1] - magnet_pos[1]
    c = sensor_pos[:, 2] - magnet_pos[2]

    m, n, p = magnet_pos[3], magnet_pos[4], magnet_pos[5]

    Blx_a = (9 * m * a + 3 * n * b + 3 * p * c) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2) - 5 * a * (
            3 * m * a ** 2 + 3 * n * b * c + 3 * p * c * a) * (a ** 2 + b ** 2 + c ** 2) ** (-7 / 2)
    Bly_a = (3 * m * b + 3 * n * a) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2) - 5 * a * (
            3 * m * a * b + 3 * n * b ** 2 + 3 * p * c * b) * (a ** 2 + b ** 2 + c ** 2) ** (-7 / 2)
    Blz_a = (3 * m * c + 3 * p * a) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2) - 5 * a * (
            3 * m * a * c + 3 * n * b * c + 3 * p * c ** 2) * (a ** 2 + b ** 2 + c ** 2) ** (-7 / 2)

    Blx_b = (3 * m * a + 3 * m * b) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2) - 5 * b * (
            3 * m * a ** 2 + 3 * n * b * c + 3 * p * a * c) * (a ** 2 + b ** 2 + c ** 2) ** (-7 / 2)
    Bly_b = (3 * m * a + 9 * n * b + 3 * p * c) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2) - 5 * b * (
            3 * m * a * b + 3 * n * b ** 2 + 3 * p * c * b) * (a ** 2 + b ** 2 + c ** 2) ** (-7 / 2)
    Blz_b = (3 * n * c + 3 * p * b) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2) - 5 * b * (
            3 * m * a * c + 3 * n * b * c + 3 * p * c ** 2) * (a ** 2 + b ** 2 + c ** 2) ** (-7 / 2)

    Blx_c = (3 * p * a + 3 * m * c) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2) - 5 * c * (
            3 * m * a ** 2 + 3 * n * b * a + 3 * p * c * a) * (a ** 2 + b ** 2 + c ** 2) ** (-7 / 2)
    Bly_c = (3 * p * b + 3 * m * c) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2) - 5 * c * (
            3 * m * a * b + 3 * n * b ** 2 + 3 * p * c * b) * (a ** 2 + b ** 2 + c ** 2) ** (-7 / 2)
    Blz_c = (3 * m * a + 3 * n * b + 9 * p * c) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2) - 5 * c * (
            3 * m * a * c + 3 * n * b * c + 3 * p * c ** 2) * (a ** 2 + b ** 2 + c ** 2) ** (-7 / 2)

    Blx_m = (2 * a ** 2 - b ** 2 - c ** 2) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2)
    Bly_m = (3 * a * b) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2)
    Blz_m = (3 * a * c) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2)

    Blx_n = (3 * a * b) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2)
    Bly_n = (2 * b ** 2 - a ** 2 - c ** 2) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2)
    Blz_n = (3 * b * c) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2)

    Blx_p = (3 * a * c) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2)
    Bly_p = (3 * b * c) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2)
    Blz_p = (2 * c ** 2 - b ** 2 - a ** 2) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2)

    a_x = -1
    b_y = -1
    c_z = -1

    B_x = Bt*np.vstack((Blx_a*a_x, Bly_a*a_x, Blz_a*a_x)).transpose()
    B_y = Bt*np.vstack((Blx_b*b_y, Bly_b*b_y, Blz_b*b_y)).transpose()
    B_z = Bt*np.vstack((Blx_c*c_z, Bly_c*c_z, Blz_c*c_z)).transpose()
    B_m = Bt*np.vstack((Blx_m, Bly_m, Blz_m )).transpose()
    B_n = Bt*np.vstack((Blx_n, Bly_n, Blz_n)).transpose()
    B_p = Bt*np.vstack((Blx_p, Bly_p, Blz_p)).transpose()

    Error_x = sum(sum(2 * Error_matrix * B_x))
    Error_y = sum(sum(2 * Error_matrix * B_y))
    Error_z = sum(sum(2 * Error_matrix * B_z))
    Error_m = sum(sum(2 * Error_matrix * B_m))
    Error_n = sum(sum(2 * Error_matrix * B_n))
    Error_p = sum(sum(2 * Error_matrix * B_p))

    return np.array([Error_x, Error_y, Error_z, Error_m, Error_n, Error_p])


def MagnetPosError_6d_jac_orientation(magnet_pos_6d, Bt, sensor_param, Measured_data):

    '''Theoretical magnetic value'''
    Measured_data = Measured_data.reshape((16, 3))

    sensor_pos = sensor_param[:, 0:3]
    sensor_rotation = sensor_param[:, 3:6]
    sensor_rotation_matrix = []
    for r_v in sensor_rotation:
        sensor_rotation_matrix.append(np.dot(yaw_matrix(r_v[2]), pitch_matrix(r_v[1]), roll_matrix(r_v[0])))
    sensor_rotation_matrix = np.array(sensor_rotation_matrix)

    magnet_pos = magnet_pos_6d

    theo_data_init = M_field_value_model(magnet_pos, sensor_pos)
    Theorectical_data = []
    for i in range(theo_data_init.shape[0]):
        Theorectical_data.append(np.dot(theo_data_init[i], sensor_rotation_matrix[i]).tolist())
    Theorectical_data = 1e7 * Bt * np.array(Theorectical_data)  # change to mG unit

    Error_matrix = Theorectical_data - Measured_data

    '''derivative part'''
    a = sensor_pos[:, 0] - magnet_pos[0]
    b = sensor_pos[:, 1] - magnet_pos[1]
    c = sensor_pos[:, 2] - magnet_pos[2]
    m, n, p = magnet_pos[3], magnet_pos[4], magnet_pos[5]

    Blx_m = (2 * a ** 2 - b ** 2 - c ** 2) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2)
    Bly_m = (3 * a * b) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2)
    Blz_m = (3 * a * c) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2)

    Blx_n = (3 * a * b) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2)
    Bly_n = (2 * b ** 2 - a ** 2 - c ** 2) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2)
    Blz_n = (3 * b * c) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2)

    Blx_p = (3 * a * c) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2)
    Bly_p = (3 * b * c) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2)
    Blz_p = (2 * c ** 2 - b ** 2 - a ** 2) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2)

    B_m = Bt*np.vstack((Blx_m, Bly_m, Blz_m )).transpose()
    B_n = Bt*np.vstack((Blx_n, Bly_n, Blz_n)).transpose()
    B_p = Bt*np.vstack((Blx_p, Bly_p, Blz_p)).transpose()

    Error_m = sum(sum(2 * Error_matrix * B_m))
    Error_n = sum(sum(2 * Error_matrix * B_n))
    Error_p = sum(sum(2 * Error_matrix * B_p))

    return np.array([0., 0., 0., Error_m, Error_n, Error_p])


def MagnetPosError_6d_jac_position(magnet_pos_6d, Bt, sensor_param, Measured_data):

    '''Theoretical magnetic value'''
    Measured_data = Measured_data.reshape((16, 3))

    sensor_pos = sensor_param[:, 0:3]
    sensor_rotation = sensor_param[:, 3:6]
    sensor_rotation_matrix = []
    for r_v in sensor_rotation:
        sensor_rotation_matrix.append(np.dot(yaw_matrix(r_v[2]), pitch_matrix(r_v[1]), roll_matrix(r_v[0])))
    sensor_rotation_matrix = np.array(sensor_rotation_matrix)

    magnet_pos = magnet_pos_6d

    theo_data_init = M_field_value_model(magnet_pos, sensor_pos)
    Theorectical_data = []
    for i in range(theo_data_init.shape[0]):
        Theorectical_data.append(np.dot(theo_data_init[i], sensor_rotation_matrix[i]).tolist())
    Theorectical_data = 1e7 * Bt * np.array(Theorectical_data)  # change to mG unit

    Error_matrix = Theorectical_data - Measured_data

    '''derivative part'''
    a = sensor_pos[:, 0] - magnet_pos[0]
    b = sensor_pos[:, 1] - magnet_pos[1]
    c = sensor_pos[:, 2] - magnet_pos[2]
    m, n, p = magnet_pos[3], magnet_pos[4], magnet_pos[5]

    Blx_a = (9 * m * a + 3 * n * b + 3 * p * c) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2) - 5 * a * (
            3 * m * a ** 2 + 3 * n * b * c + 3 * p * c * a) * (a ** 2 + b ** 2 + c ** 2) ** (-7 / 2)
    Bly_a = (3 * m * b + 3 * n * a) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2) - 5 * a * (
            3 * m * a * b + 3 * n * b ** 2 + 3 * p * c * b) * (a ** 2 + b ** 2 + c ** 2) ** (-7 / 2)
    Blz_a = (3 * m * c + 3 * p * a) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2) - 5 * a * (
            3 * m * a * c + 3 * n * b * c + 3 * p * c ** 2) * (a ** 2 + b ** 2 + c ** 2) ** (-7 / 2)

    Blx_b = (3 * m * a + 3 * m * b) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2) - 5 * b * (
            3 * m * a ** 2 + 3 * n * b * c + 3 * p * a * c) * (a ** 2 + b ** 2 + c ** 2) ** (-7 / 2)
    Bly_b = (3 * m * a + 9 * n * b + 3 * p * c) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2) - 5 * b * (
            3 * m * a * b + 3 * n * b ** 2 + 3 * p * c * b) * (a ** 2 + b ** 2 + c ** 2) ** (-7 / 2)
    Blz_b = (3 * n * c + 3 * p * b) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2) - 5 * b * (
            3 * m * a * c + 3 * n * b * c + 3 * p * c ** 2) * (a ** 2 + b ** 2 + c ** 2) ** (-7 / 2)

    Blx_c = (3 * p * a + 3 * m * c) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2) - 5 * c * (
            3 * m * a ** 2 + 3 * n * b * a + 3 * p * c * a) * (a ** 2 + b ** 2 + c ** 2) ** (-7 / 2)
    Bly_c = (3 * p * b + 3 * m * c) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2) - 5 * c * (
            3 * m * a * b + 3 * n * b ** 2 + 3 * p * c * b) * (a ** 2 + b ** 2 + c ** 2) ** (-7 / 2)
    Blz_c = (3 * m * a + 3 * n * b + 9 * p * c) * (a ** 2 + b ** 2 + c ** 2) ** (-5 / 2) - 5 * c * (
            3 * m * a * c + 3 * n * b * c + 3 * p * c ** 2) * (a ** 2 + b ** 2 + c ** 2) ** (-7 / 2)

    a_x = -1
    b_y = -1
    c_z = -1

    B_x = Bt*np.vstack((Blx_a*a_x, Bly_a*a_x, Blz_a*a_x)).transpose()
    B_y = Bt*np.vstack((Blx_b*b_y, Bly_b*b_y, Blz_b*b_y)).transpose()
    B_z = Bt*np.vstack((Blx_c*c_z, Bly_c*c_z, Blz_c*c_z)).transpose()

    Error_x = sum(sum(2 * Error_matrix * B_x))
    Error_y = sum(sum(2 * Error_matrix * B_y))
    Error_z = sum(sum(2 * Error_matrix * B_z))

    return np.array([Error_x, Error_y, Error_z, 0., 0., 0.])


def MagnetPosError_cons(magnet_pos_6d):
    #satisfied the unit norm constraints
    return magnet_pos_6d[3]**2 + magnet_pos_6d[4]**2 + magnet_pos_6d[5]**2 - 1


def MagnetPosError_cons_jac(magnet_pos_6d):
    return np.array([0., 0., 0., 2*magnet_pos_6d[3], 2*magnet_pos_6d[4], 2*magnet_pos_6d[5]])