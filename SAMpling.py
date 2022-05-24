import numpy as np
import pandas as pd
from AGENT import run_single_random

# provide a well-estimated transition kernel (with large sampling size)
def sampling_WE(ENV, nS, nA, gamma, seed, sam_size_WE, MaxStePE_WE):
    np.random.seed(seed)
    iter_tot = MaxStePE_WE  # MaxStePE: Max Step per episode
    P_count_sum = np.zeros([nS, nA, nS])  # the sum of transition counts for each episode
    P_count_ave = np.zeros([nS, nA, nS])

    for k in range(sam_size_WE):
        ob_init = np.random.choice(nS)
        episode_reward, P_count, ob_init = run_single_random(ENV, nS, nA, iter_tot=iter_tot, gamma=gamma, ob_g=ob_init)
        P_count_sum += P_count
        print(k)
    for s in range(nS):
        for a in range(nA):
            if P_count_sum.sum(axis=2)[s, a] != 0:
                for s_prime in range(nS):
                    P_count_ave[s, a, s_prime] = np.divide(P_count_sum[s, a, s_prime],
                                                           P_count_sum.sum(axis=2)[s, a],
                                                           out=np.zeros_like(P_count_sum[s, a, s_prime]),
                                                           where=P_count_sum.sum(axis=2)[s, a] != 0)
            else:
                P_count_ave[s, a] = np.ones(nS) * (1 / nS)
                # P_count_ave[s, a] = np.random.dirichlet(np.ones(nS))

    P_count_ave = np.around(P_count_ave, 3)
    for s in range(nS):
        for a in range(nA):
            if P_count_ave.sum(axis=2)[s, a] != 1:
                diff = 1 - P_count_ave.sum(axis=2)[s, a]
                index_NO = np.argsort(P_count_ave[s, a])
                if diff >= 0:
                    P_count_ave[s, a, index_NO[-2]] += diff
                else:
                    P_count_ave[s, a, index_NO[-1]] += diff
    print("sam_size_WE" + str(sam_size_WE) + "MaxStePE_WE" + str(MaxStePE_WE))
    print(P_count_ave)
    print("Sum for each state-action pair:", P_count_ave.sum(axis=2))
    print("Non-zero count:", np.count_nonzero(P_count_ave, 2))
    np.save("./Sampling/sam_size_WE" + str(sam_size_WE) + "MaxStePE_WE" + str(MaxStePE_WE), P_count_ave)
    return


# estimate transition kernels without prior knowledge
def sampling(ENV, nS, nA, gamma, seed, MaxStePE, MaxEPS):
    np.random.seed(seed)
    sam_size = np.arange(1, MaxEPS, 1, dtype=int)
    iter_tot = MaxStePE  # MaxStePE: Max Step per episode
    P_count_sum = np.zeros([nS, nA, nS])  # the sum of transition counts for each episode
    P_count_ave = np.zeros([nS, nA, nS])

    ob_init = 0
    for k in range(len(sam_size)):
        episode_reward, P_count, ob_init = run_single_random(ENV, nS, nA, iter_tot=iter_tot, gamma=gamma, ob_g=ob_init)
        P_count_sum += P_count
        for s in range(nS):
            for a in range(nA):
                # TODO if P_count_sum.sum(axis=2)[s, a]==0
                if P_count_sum.sum(axis=2)[s, a] != 0:
                    for s_prime in range(nS):
                        P_count_ave[s, a, s_prime] = np.divide(P_count_sum[s, a, s_prime],
                                                               P_count_sum.sum(axis=2)[s, a],
                                                               out=np.zeros_like(P_count_sum[s, a, s_prime]),
                                                               where=P_count_sum.sum(axis=2)[s, a] != 0)
                else:
                    P_count_ave[s, a] = np.ones(nS) * (1 / nS)
                    # P_count_ave[s, a] = np.random.dirichlet(np.ones(nS))

        P_count_ave = np.around(P_count_ave, 3)
        for s in range(nS):
            for a in range(nA):
                if P_count_ave.sum(axis=2)[s, a] != 1:
                    diff = 1 - P_count_ave.sum(axis=2)[s, a]
                    index_NO = np.argsort(P_count_ave[s, a])
                    if diff >= 0:
                        P_count_ave[s, a, index_NO[-2]] += diff
                    else:
                        P_count_ave[s, a, index_NO[-1]] += diff

        print("SamSiz" + str(sam_size[k]) + "MaxSte" + str(iter_tot))
        print(P_count_ave)
        print("Sum for each state-action pair:", P_count_ave.sum(axis=2))
        np.save("./Sampling/SamSiz" + str(sam_size[k]) + "MaxSte" + str(iter_tot), P_count_ave)
    return


# estimate transition kernels with prior knowledge
def sampling_PriorKnowledge(ENV, MatrixWE, nS, nA, gamma, seed, MaxStePE, MaxEPS):
    np.random.seed(seed)
    sam_size = np.arange(1, MaxEPS, 1, dtype=int)
    iter_tot = MaxStePE  # MaxStePE: Max Step per episode
    P_count_sum = np.zeros([nS, nA, nS])  # the sum of transition counts for each episode
    P_count_ave = np.zeros([nS, nA, nS])

    ob_init = 0
    for k in range(len(sam_size)):
        episode_reward, P_count, ob_init = run_single_random(ENV, nS, nA, iter_tot=iter_tot, gamma=gamma, ob_g=ob_init)
        P_count_sum += P_count
        for s in range(nS):
            for a in range(nA):
                # TODO if P_count_sum.sum(axis=2)[s, a]==0
                if P_count_sum.sum(axis=2)[s, a] != 0:
                    for s_prime in range(nS):
                        P_count_ave[s, a, s_prime] = np.divide(P_count_sum[s, a, s_prime],
                                                               P_count_sum.sum(axis=2)[s, a],
                                                               out=np.zeros_like(P_count_sum[s, a, s_prime]),
                                                               where=P_count_sum.sum(axis=2)[s, a] != 0)
                else:
                    for s_prime in np.nonzero(MatrixWE[s, a])[0]:
                        P_count_ave[s, a, s_prime] = 1 / np.count_nonzero(MatrixWE[s, a])

        P_count_ave = np.around(P_count_ave, 3)
        for s in range(nS):
            for a in range(nA):
                if P_count_ave.sum(axis=2)[s, a] != 1:
                    diff = 1 - P_count_ave.sum(axis=2)[s, a]
                    index_NO = np.argsort(P_count_ave[s, a])
                    if diff >= 0:
                        P_count_ave[s, a, index_NO[-2]] += diff
                    else:
                        P_count_ave[s, a, index_NO[-1]] += diff

        print("SamSiz" + str(sam_size[k]) + "MaxSte" + str(iter_tot))
        print(P_count_ave)
        print("Sum for each state-action pair:", P_count_ave.sum(axis=2))
        np.save("./Sampling/SamSiz" + str(sam_size[k]) + "MaxSte" + str(iter_tot), P_count_ave)
    return
