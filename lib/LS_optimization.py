from lib.magneticModel import M_field_value_model
import numpy as np
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






