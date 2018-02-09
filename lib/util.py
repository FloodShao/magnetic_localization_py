import numpy as np
import math


def yaw_matrix(angle):
    return np.array([
        [math.cos(angle),   -math.sin(angle),   0],
        [math.sin(angle),   math.cos(angle),    0],
        [0,                 0,                  1]
    ])


def roll_matrix(angle):
    return np.array([
        [1,                 0,                  0],
        [0,                 math.cos(angle),    math.sin(angle)],
        [0,                 -math.sin(angle),   math.cos(angle)]
    ])


def pitch_matrix(angle):
    return np.array([
        [math.cos(angle),   0,                  math.sin(angle)],
        [0,                 1,                  0],
        [-math.sin(angle),  0,                  math.cos(angle)],
    ])

