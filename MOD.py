from rsome import ro
import rsome as rso
from rsome import cvx_solver as cvx
from rsome import ort_solver as ort
from rsome import grb_solver as grb
from rsome import msk_solver as msk
import numpy as np

def Dual(nS, nA, P_est, R_sa, gamma):
# Dual MDP
    D = np.ones(nS)*(1/nS)
    model = ro.Model()
    # variables
    u = model.dvar((nS, nA))
    # objective function
    model.max((R_sa * u).sum())
    # constraints
    model.st(u[s_prime, :].sum() - D[s_prime] - gamma*(P_est[:, :, s_prime]*u).sum() <= 0 for s_prime in range(nS))
    model.st(u >= 0)
    model.solve(cvx)
    TAU = model.get()
    u_sol = u.get()
    Policy = np.argmax(u_sol, axis=1)
    #print("optimal solution", u_sol)
    return TAU, Policy




#########################
######## Backup #########
#########################

