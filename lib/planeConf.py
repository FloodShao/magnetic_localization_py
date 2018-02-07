import numpy as np

PlaneT = {
    'plane1' : np.array([[0, 0, -1],
                         [-1, 0, 0],
                         [0, 1, 0]]),

    'plane2' : np.array([[-1, 0, 0],
                         [0, 0, 1],
                         [0, 1, 0]]),

    'plane3' : np.array([[1, 0, 0],
                         [0, 0, -1],
                         [0, 1, 0]]),

    'plane4' : np.array([[0, 0, 1],
                         [1, 0, 0],
                         [0, 1, 0]])

}