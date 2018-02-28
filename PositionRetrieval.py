import scipy.io as sio
from lib.magnetPosition import magnetPos_from_setup
from lib.LS_optimization import *
from lib.PreProcessingParam import *
from scipy.optimize import least_squares
from scipy.optimize import minimize


Bt = sio.loadmat("LabParam0222.mat")['Bt']
sensorParam = sio.loadmat("LabParam0222.mat")['sensorParam']

Idx = [5]


for i in Idx:
    matdir = "./data/datamat0206/aftercal/" + datafile0206_dir[i] + ".mat"
    data = sio.loadmat(matdir)['data']
#data = data-data_cal
#print(data)

#magnetPos = magnetPos_from_setup(p_m_original[ Idx[0] ])
#print(magnetPos)


magnetPos5d_init = np.array([0., 0., 0., 0., 0.])
magnetPos6d_init = np.array([0., 0., 0., math.sqrt(1/3), math.sqrt(1/3), math.sqrt(1/3)])
x_opt, y_opt, z_opt = 0., 0., 0.
for i in range(data.shape[0]):

    '''least_squares'''
    '''
    bounds = ([-190e-3, -120e-3, -140e-3, 0., 0.], [190e-3, 120e-3, 140e-3, 180., 360.])

    res_5d = least_squares(MagnetPosError_5d, magnetPos5d_init, verbose=2, bounds=bounds,
                           jac=MagnetPosError_5d_jac, ftol=1e-8, xtol=1e-6, method='trf',
                           args=(Bt, sensorParam, data[i]))
    print("lease_squares angle")
    print(res_5d.x)
    '''

    '''minimize_5d'''
    '''
    bounds_5d = ((-190e-3, 190e-3), (-120e-3, 120e-3), (-140e-3, 140e-3), (0., 180.), (0., 360.))
    opt_5d = {'disp': False}
    res = minimize(MagnetPosError_5d, magnetPos5d_init, args=(Bt, sensorParam, data[i]), method='L-BFGS-B',
                   jac=MagnetPosError_5d_jac, bounds=bounds_5d, tol=1e-6, options=opt_5d)
    #print('L-BFGS-B')
    #print(res.x)
    '''


    '''minimize 6d simultaneous optimize'''
    '''
    cons = {
        'type': 'eq',
        'fun': MagnetPosError_cons,
        'jac': MagnetPosError_cons_jac
    }
    bounds_6d = ((-190e-3, 190e-3), (-120e-3, 120e-3), (-140e-3, 140e-3), (-1., 1.), (-1., 1.), (-1., 1.))
    opt = {'maxiter': 100, 'disp': False}
    res_6d = minimize(MagnetPosError_6d, magnetPos6d_init, args=(Bt, sensorParam, data[i]), method='SLSQP',
                      jac=MagnetPosError_6d_jac, bounds=bounds_6d, constraints=cons, tol=1e-6,
                      options=opt)
    print(res_6d.x)
    '''


    '''minimize 6d, first orienttaion next position'''
    cons = {
        'type':'eq',
        'fun': MagnetPosError_cons,
        'jac': MagnetPosError_cons_jac
    }
    bounds_6d = ((-190e-3, 190e-3), (-120e-3, 120e-3), (-140e-3, 140e-3), (None, None), (None, None), (None, None))
    opt = {'maxiter': 100, 'disp': True}
    res_6d = minimize(MagnetPosError_6d, magnetPos6d_init, args=(Bt, sensorParam, data[i]), method='SLSQP',
                   jac=MagnetPosError_6d_jac_orientation, bounds=bounds_6d, constraints=cons, tol=1e-6,
                   options=opt)

    m_opt, n_opt, p_opt = res_6d.x[3], res_6d.x[4], res_6d.x[5]

    if math.isnan(m_opt) or math.isnan(n_opt) or math.isnan(p_opt):
        print("[Warning] Failed to estimate the orientation of the magnet")
        magnetPos6d_init = np.array([0., 0., 0., math.sqrt(1 / 3), math.sqrt(1 / 3), math.sqrt(1 / 3)])
    else:
        magnetPos6d_init = np.array([0., 0., 0., m_opt, n_opt, p_opt])

    print(magnetPos6d_init)

    bounds = ([-190e-3, -120e-3, -140e-3, -1., -1., -1.], [190e-3, 120e-3, 140e-3, 1., 1., 1.])
    res_final = least_squares(MagnetPosError_6d, magnetPos6d_init, verbose=0, bounds = bounds,
                           jac=MagnetPosError_6d_jac, ftol=1e-8, xtol=1e-10, method='trf',
                           args=(Bt, sensorParam, data[i]))

    x_opt, y_opt, z_opt = res_final.x[0], res_final.x[1], res_final.x[2]
    print("optimization results")
    print(res_final.x)

    magnetPos6d_init = np.array([0., 0., 0., math.sqrt(1/3), math.sqrt(1/3), math.sqrt(1/3)])

    print("original position")
    print(magnetPos_from_setup(p_m_original[Idx[0]])[i])

    print("\n")



