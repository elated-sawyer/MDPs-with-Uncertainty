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

def RSMDP(nS, nA, P_est, R_sa, TAU, Gamma):
# Robust Satisficing MDP
    # Parameters
    D = np.ones(nS)*(1/nS)
    W = np.ones(nS)*(1/nS)
    W = W.transpose()
    model = ro.Model()
    # decision variable
    mu = model.dvar((nS, nA))
    k = model.dvar(nS)
    zeta = model.dvar((nS, nS, nA, nS))
    # random variables
    p = model.rvar((nS, nA, nS))
    # uncertainty set
    Uset = (
            p >= 0,     # Support Set 1
            p.sum(axis=2) == 1,     # Support Set 2+
            # rso.norm((p - P_emp).reshape(S*A*S), 1) <= 0.5     # Support Set 3+
            )    # define the uncertainty set
    # define model objective and constraints
    model.minmax((W*k).sum(), Uset)
    model.st( mu[s_prime, :].sum() - D[s_prime] - Gamma*(p[:, :, s_prime]*mu).sum() <= (zeta[s_prime, :, :, :] * (p - P_est)).sum() for s_prime in range(nS))
    model.st(rso.norm((zeta[s_prime, :, :, :]).reshape(nS*nA*nS), 1) <= k[s_prime] for s_prime in range(nS))
    model.st((R_sa * mu).sum() >= TAU)
    model.st(k >= 0)
    model.st(mu >= 0)
    model.solve(msk)
    mu_sol = mu.get()
    # Calculate probs policy based on the solution mu
    Probs = np.zeros([nS, nA])
    for s in range(nS):
        for a in range(nA):
            if mu_sol.sum(axis=1)[s] != 0:
                Probs[s, a] = np.divide(mu_sol[s, a], mu_sol.sum(axis=1)[s])
            else:
                Probs[s, a] = 1/nA
            Probs = np.around(Probs, 4)
    for s in range(nS):
        if Probs.sum(axis=1)[s] != 1:
            diff = 1 - Probs.sum(axis=1)[s]
            index_NO = np.argsort(Probs[s])
            if diff >= 0:
                Probs[s, index_NO[-2]] += diff
            else:
                Probs[s, index_NO[-1]] += diff

    TAU_real = ((R_sa * mu_sol).sum())
    Policy = np.argmax(mu_sol, axis=1)
    # print("mu optimal solution", mu_sol)
    # print("optimal value", model.get())
    # print("TAU is", TAU)
    # print("TAU_real", TAU_real)
    # print("optimal solution-probs", Probs)
    # print("optimal policy is", Policy)
    return TAU_real, Probs, Policy



#########################
######## Backup #########
#########################

def DRMDP(nS, nA, p_est, r, V, gamma, theta=0.0, k=1):
# Distributionally robust MDPs
    N = p_est.shape[0]
    model = ro.Model()
    # variables
    p = model.dvar((N, nS))
    # objective function
    model.min(r + (p.sum(axis=0) * (1/N) * (gamma * V)).sum())
    # constraints
    model.st(rso.norm((p - p_est).reshape(nS * N), k) * (1 / N) <= theta**k)  # L-k norm
    model.st(p >= 0)
    model.st(p <= 1)
    model.st(p.sum(axis=1) == 1)
    model.solve(msk)

    V_val = model.get()
    p_sol = p.get()
    return V_val, p_sol
