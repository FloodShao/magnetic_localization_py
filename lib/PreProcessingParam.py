import numpy as np
import math

datafile0206_dir = {
    0: '1_8_z_0_(0,0)',
    1: '1_8_z_0_(90,0)',
    2: '1_8_z_0_(90,90)',
    3: '1_8_z_-1_(0,0)',
    4: '1_8_z_-1_(90,0)',
    5: '1_8_z_-1_(90,90)',
    6: '2_24_z_0_(0,0)',
    7: '2_24_z_-1_(0,0)',
    8: '3_24_z_0_(0,0)',
    9: '3_24_z_-1_(0,0)',
    10: '4_24_z_0_(0,0)',
    11: '4_24_z_-1_(0,0)',
    12: 'cross_z_0_(90,0)',
    13: 'cross_z_0_(90,90)',
    14: 'cross_z_-1_(90,0)',
    15: 'cross_z_-1_(90,90)',
}

l_1 = 20e-3
R_2 = 40e-3
R_3 = 60e-3
R_4 = 80e-3
R_5 = 100e-3
h = 15e-3

delta = 2 * math.pi / 24
circle_uxy = []
for i in range(24):
    circle_uxy.append([math.sin(i * delta), math.cos(i * delta)])
circle_uxy = np.array(circle_uxy)

p_xy = {
    '1' : np.array([
        [0., l_1],
        [l_1, l_1],
        [l_1, 0.],
        [l_1, -l_1],
        [0., -l_1],
        [-l_1, -l_1],
        [-l_1, 0.],
        [-l_1, l_1]
    ]),

    '2' : R_2 * circle_uxy,

    '3' : R_3 * circle_uxy,

    '4' : R_4 * circle_uxy,

    '5' : R_5 * circle_uxy,

    'c_phi0' : np.array([
        [0., R_5], [0., R_4], [0., R_3], [0., R_2], [0., l_1], [0., -l_1], [0., -R_2], [0., -R_3], [0., -R_4], [0., -R_5],
        [-R_5, 0.], [-R_4, 0.], [-R_3, 0.], [-R_2, 0.], [-l_1, 0.], [l_1, 0.], [R_2, 0.], [R_3, 0.], [R_4, 0.], [R_5, 0.]
    ]),

    'c_phi90' : np.array([
        [0., R_4], [0., R_3], [0., R_2], [0., l_1], [0., -l_1], [0., -R_2], [0., -R_3], [0., -R_4],
        [-R_5, 0.], [-R_4, 0.], [-R_3, 0.], [-R_2, 0.], [-l_1, 0.], [l_1, 0.], [R_2, 0.], [R_3, 0.], [R_4, 0.], [R_5, 0.]
    ])

}

'''1_8_z_-1_(90,90)的phi应该为-90'''
'''cross_z_0_(90,0)的phi应该为180'''
'''cross_z_0_(90,90)的phi应该为-90'''
'''cross_z_-1_(90,90)的phi应该为-90'''

'''p_m_original is (x, y, z, theta, phi) vector for each magnet position'''
p_m_original = {

    0: np.hstack((  p_xy['1'],      np.zeros(( p_xy['1'].shape[0], 1)),  np.zeros(( p_xy['1'].shape[0], 1)),
                    np.zeros(( p_xy['1'].shape[0], 1)) )),
    1: np.hstack((p_xy['1'], np.zeros((p_xy['1'].shape[0], 1)), 90 * np.ones((p_xy['1'].shape[0], 1)),
                  np.zeros((p_xy['1'].shape[0], 1)))),
    2: np.hstack((p_xy['1'], np.zeros((p_xy['1'].shape[0], 1)), 90 * np.ones((p_xy['1'].shape[0], 1)),
                  90 * np.ones((p_xy['1'].shape[0], 1)))),
    3: np.hstack((p_xy['1'], -h * np.ones((p_xy['1'].shape[0], 1)), np.zeros((p_xy['1'].shape[0], 1)),
                  np.zeros((p_xy['1'].shape[0], 1)))),
    4: np.hstack((p_xy['1'], -h * np.ones((p_xy['1'].shape[0], 1)), 90 * np.ones((p_xy['1'].shape[0], 1)),
                  np.zeros((p_xy['1'].shape[0], 1)))),
    5: np.hstack((p_xy['1'], -h * np.ones((p_xy['1'].shape[0], 1)), 90 * np.ones((p_xy['1'].shape[0], 1)),
                  -90 * np.ones((p_xy['1'].shape[0], 1)))),
    6: np.hstack((p_xy['2'], np.zeros((p_xy['2'].shape[0], 1)), np.zeros((p_xy['2'].shape[0], 1)),
                  np.zeros((p_xy['2'].shape[0], 1)))),
    7: np.hstack((p_xy['2'], -h * np.ones((p_xy['2'].shape[0], 1)), np.zeros((p_xy['2'].shape[0], 1)),
                  np.zeros((p_xy['2'].shape[0], 1)))),
    8: np.hstack((p_xy['3'], np.zeros((p_xy['3'].shape[0], 1)), np.zeros((p_xy['3'].shape[0], 1)),
                  np.zeros((p_xy['3'].shape[0], 1)))),
    9: np.hstack((p_xy['3'], -h * np.ones((p_xy['3'].shape[0], 1)), np.zeros((p_xy['3'].shape[0], 1)),
                  np.zeros((p_xy['3'].shape[0], 1)))),
    10: np.hstack((p_xy['4'], np.zeros((p_xy['4'].shape[0], 1)), np.zeros((p_xy['4'].shape[0], 1)),
                   np.zeros((p_xy['4'].shape[0], 1)))),
    11: np.hstack((p_xy['4'], -h * np.ones((p_xy['4'].shape[0], 1)), np.zeros((p_xy['4'].shape[0], 1)),
                   np.zeros((p_xy['4'].shape[0], 1)))),
    12: np.hstack((p_xy['c_phi0'], np.zeros((p_xy['c_phi0'].shape[0], 1)), 90 * np.ones((p_xy['c_phi0'].shape[0], 1)),
                   180*np.ones((p_xy['c_phi0'].shape[0], 1)))),
    13: np.hstack((p_xy['c_phi90'], np.zeros((p_xy['c_phi90'].shape[0], 1)), 90 * np.ones((p_xy['c_phi90'].shape[0], 1)),
                  -90 * np.ones((p_xy['c_phi90'].shape[0], 1)))),
    14: np.hstack((p_xy['c_phi0'], -h * np.zeros((p_xy['c_phi0'].shape[0], 1)),
                   90 * np.ones((p_xy['c_phi0'].shape[0], 1)), np.zeros((p_xy['c_phi0'].shape[0], 1)))),
    15: np.hstack((p_xy['c_phi90'], -h * np.zeros((p_xy['c_phi90'].shape[0], 1)),
                   90 * np.ones((p_xy['c_phi90'].shape[0], 1)), -90 * np.ones((p_xy['c_phi90'].shape[0], 1)))),

}