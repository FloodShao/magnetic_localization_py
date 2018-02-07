import numpy as np

def M_field_value_model(magnet_pos, sensor_position):
    """
    This function calculate the magnetic field value given the magnet position and sensor position
    Type is np array
    :param magnet: 1*6 dimension, 0:3 is the position of the magnet, 3:6 is the orientation vector of the magnet
    :param sensor_position: n*3 matrix, sensor position in world coordinates
    :return: n*3 matrix, magnetic field value of each position in world coordinate axis
    """

    position_m = magnet_pos[0:3]
    orientation_m = magnet_pos[3:6]

    if np.linalg.norm(orientation_m) != 1:
        orientation_m /= np.linalg.norm(orientation_m)

    output = []
    distance = position_m - sensor_position
    i = 0
    for p in sensor_position:
        R = np.linalg.norm(distance[i])
        H0_F1 = np.dot(orientation_m, p)
        B = 1/R**5 * (3 * H0_F1 * p - R**2 * orientation_m)
        output.append(B.tolist())
        i += 1

    return np.array(output)










