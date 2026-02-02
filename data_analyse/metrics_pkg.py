import numpy as np
from scipy import integrate

q = ['trunk_ie','trunk_aa','trunk_fe',
	'clav_dep_ev','clav_prot_ret',
	'shoulder_fe','shoulder_aa','shoulder_ie',
	'elbow_fe','elbow_ps']

def compute_impulse(time,torque):
    """ Compute impulse of each joint using trapezoidal integration """
    torque_abs = np.abs(torque)
    impulse = []
    for i in range(torque.shape[1]):
        impulse_q = integrate.simpson(torque_abs[:,i], x=time)
        impulse.append(impulse_q)
    return np.array(impulse)

def compute_peak_tau(torque):
    """ Compute peak torque of each joint """
    peak_tau = np.max(np.abs(torque),axis=0)
    # for i in range(torque.shape[1]):
    #     print(q[i],peak_tau[i])
    return peak_tau