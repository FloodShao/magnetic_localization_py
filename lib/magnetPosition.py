import numpy as np
import math

def magnetPos_from_setup(X, Y, Z, theta, phi):

    theta = math.pi * theta / 180
    phi = math.pi * phi / 180
    x, y, z = X, Y, Z + 1e-3*12.5
    m, n, p = math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), math.cos(theta)
    return np.array([x, y, z, m, n, p])

def magnetPos_from_setup(P_v_5):
    """

    :param P_v_5: n*5 position, X, Y, Z, theta, phi
    theta, phi in degree
    :return:
    """
    P_v_5 = P_v_5.transpose()
    x = P_v_5[0]
    y = P_v_5[1]
    z = P_v_5[2] + 1e-3*12.5 #container has 10mm height, and the position board has 5 mm height

    m = np.sin(np.radians(P_v_5[3]))*np.cos(np.radians(P_v_5[4]))
    n = np.sin(np.radians(P_v_5[3]))*np.sin(np.radians(P_v_5[4]))
    p = np.cos(np.radians(P_v_5[3]))

    return np.vstack((x, y, z, m, n, p)).transpose()