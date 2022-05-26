import numpy as np

# bellman optimality equation
def BellmanOp(P, V, R_sa, gamma):
    BV = np.sum(P * (R_sa + gamma * V))
    return BV

# value iteration
def value_iteration(P, nS, nA, R_sa, gamma, max_iteration=10000, tol=1e-3):
    V = np.zeros((nS,), dtype=float)
    V.fill(1000)
    policy = np.zeros(nS, dtype=int)

    for iter_count in range(max_iteration):
        newV = np.zeros(nS)
        for state in range(nS):
            BV = np.zeros(nA)
            for action in range(nA):
                BV[action] = R_sa[state][action] + np.sum(P[state][action] * (gamma * V))  # BellmanOp
            newV[state] = BV.max()
        Vdiff = np.max(np.abs(newV - V))
        V = newV
        if Vdiff < tol:
            break
    for state in range(nS):
        BV = np.zeros(nA)
        for action in range(nA):
            BV[action] = R_sa[state][action] + np.sum(P[state][action] * (gamma * V))
        policy[state] = np.argmax(BV)
        # print(BV)
    return V, policy

# robust value iteration
def value_iteration_robust(P, nS, nA, R_sa, gamma, max_iteration, tol, radius):
    V = np.zeros((nS,), dtype=float)
    V.fill(1000)
    policy = np.zeros(nS, dtype=int)
    z = gamma * V

    for iter_count in range(max_iteration):
        newV = np.zeros(nS)
        index = z.argsort()
        for state in range(nS):
            BV = np.zeros(nA)
            for action in range(nA):

                # Reference: Algorithm 2 in
                # "RAAM: The Benefits of Robustness in Approximating Aggregated MDPs in Reinforcement Learning"
                #################
                PP = P[state][action]
                i = nS - 1
                O = PP.copy()
                eta = np.min([1 - PP[index[0]], radius / 2])
                O[index[0]] = eta + PP[index[0]]  # the sum of transition matrix is equal to 1
                while eta > 0 and i >= 0:
                    O[index[i]] = np.round(O[index[i]] - np.min([eta, O[index[i]]]), 3)
                    eta = np.round(eta - np.min([eta, PP[index[i]]]), 3)
                    i -= 1
                O = np.around(O, 5)
                ##################

                BV[action] = R_sa[state][action] + np.sum(O * z)  # BellmanOp
            newV[state] = BV.max()
        Vdiff = np.max(np.abs(newV - V))
        V = newV
        z = gamma * V
        if Vdiff < tol:
            break
    for state in range(nS):
        BV = np.zeros(nA)
        for action in range(nA):
            BV[action] = R_sa[state][action] + np.sum(P[state][action] * (gamma * V))
        policy[state] = np.argmax(BV)
    return V, policy



#########################
######## Backup #########
#########################

