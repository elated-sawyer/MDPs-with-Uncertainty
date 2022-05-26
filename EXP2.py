from ENV import ENV_MR, ENV_GW, ENV_RiSw
from VI import value_iteration, value_iteration_robust
from MOD import Dual, RSMDP
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shutil


#####################################
#### Generate transition matrix #####
#####################################
# add pertibetion to the real transition kernel
def MatGen(file_WETR, ENV, TrainingSize, TestSize, mu, sigma):
    P_real = np.load("./"+file_WETR+"/sam_size_WE1500MaxStePE_WE10000.npy")########
    nS = ENV.nS
    nA = ENV.nA

    for i in [1,2]:
        if i == 1:
            GenSize = TrainingSize
            file = "Train"
        else:
            GenSize = TestSize
            file = "Test"
        np.random.seed(i)

        MatID = np.arange(1, GenSize+1, 1, dtype=int)
        for k in range(len(MatID)):
            pert_1 = np.absolute(np.random.normal(mu, sigma, size=(nS,nA,nS)))
            #pert_2 = np.random.choice([0, 0.5, 1], size=(nS,nA,nS), p=[6./ 8, 1./ 8, 1./ 8]) # 1
            P_wPert = P_real + pert_1

            P_rand = np.zeros([nS, nA, nS])
            for s in range(nS):
                for a in range(nA):
                    # TODO if P_wPert.sum(axis=2)[s, a]==0
                    if P_wPert.sum(axis=2)[s, a] != 0:
                        for s_prime in range(nS):
                            P_rand[s, a, s_prime] = np.divide(P_wPert[s, a, s_prime],
                                                                   P_wPert.sum(axis=2)[s, a],
                                                                   out=np.zeros_like(P_wPert[s, a, s_prime]),
                                                                   where=P_wPert.sum(axis=2)[s, a] != 0)
                    else:
                        P_rand[s, a] = np.ones(nS) * (1 / nS)
                        # P_rand[s, a] = np.random.dirichlet(np.ones(nS))
            P_rand = np.around(P_rand, 3)

            for s in range(nS):
                for a in range(nA):
                    if P_rand.sum(axis=2)[s, a] != 1:
                        diff = 1 - P_rand.sum(axis=2)[s, a]
                        index_NO = np.argsort(P_rand[s, a])
                        if diff >= 0:
                            P_rand[s, a, index_NO[-2]] += diff
                        else:
                            P_rand[s, a, index_NO[-1]] += diff
            #print("MatID" + str(MatID[k]))
            #print("Sum for each state-action pair:", P_rand.sum(axis=2))
            np.save("./TOF_GenMat/"+file+"/MatID"+str(MatID[k]), P_rand)
    return


#####################################
#### transition matrix process ######
#####################################
# make the transition kernel meet the constraint, P.sum(axis=2)[s, a]==0
def MatProcess(ENV, P_wPert):
    nS = ENV.nS
    nA = ENV.nA
    P_rand = np.zeros([nS, nA, nS])
    for s in range(nS):
        for a in range(nA):
            # TODO if P_wPert.sum(axis=2)[s, a]==0
            if P_wPert.sum(axis=2)[s, a] != 0:
                for s_prime in range(nS):
                    P_rand[s, a, s_prime] = np.divide(P_wPert[s, a, s_prime],
                                                           P_wPert.sum(axis=2)[s, a],
                                                           out=np.zeros_like(P_wPert[s, a, s_prime]),
                                                           where=P_wPert.sum(axis=2)[s, a] != 0)
            else:
                P_rand[s, a] = np.ones(nS) * (1 / nS)
                # P_rand[s, a] = np.random.dirichlet(np.ones(nS))
    P_rand = np.around(P_rand, 3)
    for s in range(nS):
        for a in range(nA):
            if P_rand.sum(axis=2)[s, a] != 1:
                diff = 1 - P_rand.sum(axis=2)[s, a]
                index_NO = np.argsort(P_rand[s, a])
                if diff >= 0:
                    P_rand[s, a, index_NO[-2]] += diff
                else:
                    P_rand[s, a, index_NO[-1]] += diff
    print("EstMat")
    print(P_rand)
    #print("Sum for each state-action pair:", P_rand.sum(axis=2))
    return P_rand


#################################
#### Calculate Policy Value #####
#################################
# Bellman Equation
def PolVal(ENV, policy, P, max_iteration = 5000, tol=1e-3):
    nS = ENV.nS
    nA = ENV.nA
    R_sa = ENV.R_sa
    V = np.zeros((nS,), dtype=float)
    V.fill(1000)
    if policy.ndim == 1:
        pi = np.zeros((policy.size, ENV.nA))
        pi[np.arange(policy.size), policy.astype(int)] = 1
    else:
        pi = policy
    for iter_count in range(max_iteration):
        newV = np.zeros(nS)
        for state in range(nS):
            newV[state] = pi[state]@(R_sa[state] + P[state] @ (gamma * V))  # BellmanOp
        Vdiff = np.max(np.abs(newV - V))
        V = newV
        if Vdiff < tol:
            break
    return V, V.sum()/ENV.nS


##################################################################
#### Observe (ExpRet - PreRet)'s relationship with P-\bar{P} #####
##################################################################
# calculate (ExpRet - PreRet) and norm(P-\bar{P})
def EVA_Obs(ENV, P_emp, TestSize, policy, PredictedReturn, norm = 1):
    Y_record = np.array([])  # -1: Predicted Val, 0~-2: experimental expected return
    X_record = np.array([])
    Sam_Ret = np.array([])
    MatID = np.arange(1, TestSize + 1, 1, dtype=int)
    for k in range(len(MatID)):
        P = np.load("./TOF_GenMat/Test/MatID" + str(MatID[k]) + ".npy")
        V, Exp_Ret = PolVal(ENV, policy, P)
        y = Exp_Ret - PredictedReturn
        Sam_Ret = np.append(Sam_Ret, Exp_Ret)
        Y_record = np.append(Y_record, y)
        x = np.linalg.norm((P_emp - P).reshape(ENV.nS*ENV.nA*ENV.nS), norm)
        X_record = np.append(X_record, x)
    return Y_record, X_record, Sam_Ret


#######################
####  Parameters  #####
#######################
ENV = ENV_MR()
ENV_name = "MR"
# the file directory for "well estimated transition kernel"
file_WETR = "Sampling_MR_TM-O_L1"
"""ENV = ENV_GW()
ENV_name = "GW"
file_WETR = "Sampling_GP_TM-M_L1" """
"""ENV = ENV_RiSw()
ENV_name = "RiSw"
file_WETR = "Sampling_RiSw_TM-O_L1" """


TrainingSize = 1  # General:2, General:1
TestSize = 1000  # plot 150
sam_size_WE = 1500
MaxStePE_WE = 10000
gamma = 0.85
nS = ENV.nS
nA = ENV.nA
R_sa = ENV.R_sa


###################
#### DRO_MAIN #####
###################
#### Generate transition matrix #####
mu = np.array\
    ([0,0,0,0,0, 0,0,1,1,2, 2.5,2.5,3,3,4]) #4
    # ([0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0]) #4
sigma = np.array\
    ([5e-3,1e-2,5e-2,0.1,0.2, 1,5,1,5,5, 1,5,1,2,5]) #5
StepSize = int(np.floor(TestSize/15)) # plot 10
TestSize_per = np.arange(TestSize, StepSize-1, -StepSize)
for i in range(15):
    MatGen(file_WETR, ENV, TrainingSize, TestSize_per[i], mu[i], sigma[i])
# place original true transition kernel in the train set file
shutil.copyfile("./" + file_WETR + "/sam_size_WE1500MaxStePE_WE10000.npy", "./TOF_GenMat/Train/MatID1.npy")
# place original true transition kernel in the Test set file, to observe the special case, p == \bar{p}
shutil.copyfile("./" + file_WETR + "/sam_size_WE1500MaxStePE_WE10000.npy", "./TOF_GenMat/Test/MatID1.npy")


################################################################
########################### TABLE ##############################
######## Policy Value - PredictedReturn V.S. Distance ##########
################################################################

######## RMDP ##########
# train the model with original true transition kernel without perturbation
Record_Pre = np.array([])
Record_Dif = np.array([])
Record_Sam = np.array([])
P_est = np.load("./TOF_GenMat/Train/MatID1.npy")
MaxStePE = 500
tol = 1e-3
radius = np.array([0.0, 0.3, 0.6, 0.9, 1.2, 1.5])
for Rad in radius:
    Model = "RMDP "+str(Rad)
    V_vi, R_policy = value_iteration_robust(P_est, ENV.nS, ENV.nA, ENV.R_sa, gamma, MaxStePE, tol, Rad)
    PredictedReturn = V_vi.sum()/ENV.nS
    print("Policy:",R_policy)
    y, x, Sam_Ret = EVA_Obs(ENV, P_est, TestSize, R_policy, PredictedReturn, norm = 1)
    print("x:",x)
    print("y:",y)
    Record_Pre = np.append(Record_Pre, PredictedReturn)
    Record_Sam = np.append(Record_Sam, np.median(Sam_Ret))
    Record_Dif = np.append(Record_Dif, np.median(y))
print("Record_Pre:",np.around(Record_Pre,1))
print("Record_Sam:",np.around(Record_Sam,1))
print("Record_Dif:",np.around(Record_Dif,1))

######## RSMDP ##########
# train the model with original true transition kernel without perturbation
Record_Pre = np.array([])
Record_Dif = np.array([])
Record_Sam = np.array([])
P_est = np.load("./TOF_GenMat/Train/MatID1.npy")
print(P_est)
Target_COEF = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
Target, policy = Dual(ENV.nS, ENV.nA, P_est, ENV.R_sa, gamma)
for TAU in Target_COEF:
    Model = "RSMDP "+str(TAU)
    TAU_real, Probs, RS_Policy = RSMDP(ENV.nS, ENV.nA, P_est, ENV.R_sa, TAU * Target, gamma)
    print("Policy:",Probs)
    PredictedReturn = TAU_real
    y, x, Sam_Ret = EVA_Obs(ENV, P_est, TestSize, Probs, PredictedReturn, norm = 1)
    print("x:",x)
    print("y:",y)
    Record_Pre = np.append(Record_Pre, PredictedReturn)
    Record_Sam = np.append(Record_Sam, np.median(Sam_Ret))
    Record_Dif = np.append(Record_Dif, np.median(y))
print("Record_Pre:", np.around(Record_Pre, 1))
print("Record_Sam:", np.around(Record_Sam, 1))
print("Record_Dif:", np.around(Record_Dif, 1))
print("Test")


################################################################
######################## SCATTER PLOT ##########################
######## Policy Value - PredictedReturn V.S. Distance ##########
################################################################

plt.rcParams["font.family"] = "Times New Roman"
fig, ax = plt.subplots(1,2,figsize=(16,7))

######## RMDP ##########
# train the model with original true transition kernel without perturbation
P_est = np.load("./TOF_GenMat/Train/MatID1.npy")
MaxStePE = 500
tol = 1e-3
# radius #
radius = np.array([0.05, 0.1]) #RiSw, MR, GW
for Rad in radius:
    Model = "RMDPs "+str(Rad)
    V_vi, R_policy = value_iteration_robust(P_est, ENV.nS, ENV.nA, ENV.R_sa, gamma, MaxStePE, tol, Rad)
    PredictedReturn = V_vi.sum()/ENV.nS
    # test obtained policy on the transition kernel with perturbation
    y, x, Sam_Ret = EVA_Obs(ENV, P_est, TestSize, R_policy, PredictedReturn, norm = 1)
    print("Policy:",R_policy)
    print("x:",x)
    print("y:",y)
    ax[0].scatter(x, y, label = Model, alpha = 0.3, marker = "^")
ax[0].legend(fontsize= 22, loc='lower left')
ax[0].set_title('RMDPs',fontsize= 24)
ax[0].hlines(0, 0, np.max(x), colors="r", linestyles='solid')
ax[0].tick_params(axis='both', which='major', labelsize=22)
ax[0].set_xlabel('Level of Contamination', fontsize=22)
ax[0].set_ylabel('Sample Return - Predicted Return', fontsize=22)
#ax[0].set(xlim=(-2, 35), ylim=(-10, 15))

######## RSMDP ##########
"""P_sum = np.zeros([ENV.nS, ENV.nA, ENV.nS])
MatID = np.arange(1, TrainingSize+1, 1, dtype=int)
for k in range(len(MatID)):
    P = np.load("./TOF_GenMat/Train/MatID" + str(MatID[k]) + ".npy")
    P_sum += P
P_est = MatProcess(ENV, P_sum)"""
# train the model with original true transition kernel without perturbation
P_est = np.load("./TOF_GenMat/Train/MatID1.npy")
print(P_est)
# Target_COEF #
# Target_COEF = np.array([0.8, 0.85]) #RiSw
Target_COEF = np.array([0.9, 0.85]) #MR, GW
Target, policy = Dual(ENV.nS, ENV.nA, P_est, ENV.R_sa, gamma)
for TAU in Target_COEF:
    Model = "RSMDPs "+str(TAU)
    TAU_real, Probs, RS_Policy = RSMDP(ENV.nS, ENV.nA, P_est, ENV.R_sa, TAU * Target, gamma)
    PredictedReturn = TAU_real
    # test obtained policy on the transition kernel with perturbation
    y, x, Sam_Ret = EVA_Obs(ENV, P_est, TestSize, Probs, PredictedReturn, norm = 1)
    print("x:",x)
    print("y:",y)
    ax[1].scatter(x, y, label = Model, alpha = 0.3)
ax[1].legend(fontsize= 22)
ax[1].set_title('RSMDPs',fontsize= 24)
ax[1].hlines(0, 0, np.max(x), colors="r", linestyles='solid')
ax[1].tick_params(axis='both', which='major', labelsize=22)
ax[1].set_xlabel('Level of Contamination', fontsize=22)
ax[1].set_ylabel('Sample Return - Predicted Return', fontsize=22)
#ax[1].set(xlim=(-2, 35), ylim=(-10, 15))

######## Nominal MDP ##########
"""# train the model with original true transition kernel without perturbation
P_est = np.load("./TOF_GenMat/Train/MatID1.npy")
Model = "NMDP"
V_vi, R_policy = value_iteration(P_est, ENV.nS, ENV.nA, ENV.R_sa, gamma)
PredictedReturn = V_vi.sum()/ENV.nS
y, x = EVA_Obs(ENV, P_est, TestSize, R_policy, PredictedReturn, norm = 1)
print("x:",x)
print("y:",y)
ax.scatter(x, y, label = Model, alpha = 0.3, marker = "2")
ax.legend()"""

######## Save Plot ##########
plt.savefig('./Figure/ExpRet-PreRet-Deviation_'+ENV_name+'.pdf', bbox_inches = 'tight')
plt.show()






#########################
######## Backup #########
#########################

################################################################
######## Policy Value - PredictedReturn V.S. Distance ##########
################################################################
"""
fig, ax = plt.subplots(figsize=(12,6))

######## RSMDP ##########

P_sum = np.zeros([ENV.nS, ENV.nA, ENV.nS])
MatID = np.arange(1, TrainingSize+1, 1, dtype=int)
for k in range(len(MatID)):
    P = np.load("./TOF_GenMat/Train/MatID" + str(MatID[k]) + ".npy")
    P_sum += P
P_est = MatProcess(ENV, P_sum)
# train the model with original true transition kernel without perturbation
P_est = np.load("./TOF_GenMat/Train/MatID1.npy")
print(P_est)

Target_COEF = np.array([1.0, 0.90, 0.85])
Target, policy = Dual(ENV.nS, ENV.nA, P_est, ENV.R_sa, gamma)
for TAU in Target_COEF:
    Model = "RSMDP "+str(TAU)
    TAU_real, Probs, RS_Policy = RSMDP(ENV.nS, ENV.nA, P_est, ENV.R_sa, TAU * Target, gamma)
    PredictedReturn = TAU_real
    y, x = EVA_Obs(ENV, P_est, TestSize, Probs, PredictedReturn, norm = 1)
    print("x:",x)
    print("y:",y)
    ax.scatter(x, y, label = Model, alpha = 0.3)
    ax.legend()

######## Robust MDP ##########
# train the model with original true transition kernel without perturbation
P_est = np.load("./TOF_GenMat/Train/MatID1.npy")

MaxStePE = 500
tol = 1e-3
radius = np.array([0.05, 0.1, 0.2])
for Rad in radius:
    Model = "RMDP "+str(Rad)
    V_vi, R_policy = value_iteration_robust(P_est, ENV.nS, ENV.nA, ENV.R_sa, gamma, MaxStePE, tol, Rad)
    PredictedReturn = V_vi.sum()/ENV.nS
    y, x = EVA_Obs(ENV, P_est, TestSize, R_policy, PredictedReturn, norm = 1)
    print("x:",x)
    print("y:",y)
    ax.scatter(x, y, label = Model, alpha = 0.3, marker = "^")
    ax.legend()

######## Nominal MDP ##########
# train the model with original true transition kernel without perturbation
P_est = np.load("./TOF_GenMat/Train/MatID1.npy")
Model = "NMDP"
V_vi, R_policy = value_iteration(P_est, ENV.nS, ENV.nA, ENV.R_sa, gamma)
PredictedReturn = V_vi.sum()/ENV.nS
y, x = EVA_Obs(ENV, P_est, TestSize, R_policy, PredictedReturn, norm = 1)
print("x:",x)
print("y:",y)
ax.scatter(x, y, label = Model, alpha = 0.3, marker = "2")
ax.legend()

######## DROMDP ##########
P_record = np.array([])
MatID = np.arange(1, TrainingSize+1, 1, dtype=int)
for k in range(len(MatID)):
    if k:
        P = np.load("./TOF_GenMat/Train/MatID" + str(MatID[k]) + ".npy")
        P = np.expand_dims(P, axis=0)
        P_record = np.concatenate((P_record, P), axis = 0)
    else:
        P_record = np.load("./TOF_GenMat/Train/MatID" + str(MatID[k]) + ".npy")
        P_record = np.expand_dims(P_record, axis=0)
P_est = P_record

Theta_list = np.array([0.05, 0.1, 0.2])
for Theta in Theta_list:
    Model = "DROMDP" + str(Theta)
    ##
    V_vi, policy = value_iteration_DRO(P_est, ENV.nS, ENV.nA, ENV.R_sa, gamma, Theta)
    np.save("./DRO_EXP_Results/Policy_DROMDP" + str(Theta), policy)
    ##
    #policy = np.load(".//DRO_EXP_Results/Policy_DROMDP" + str(Theta) + ".npy")
    ##
    PredictedReturn = V_vi.sum()/ENV.nS
    y, x = EVA_Obs(ENV, P_est, TestSize, policy, PredictedReturn, norm = 1)
    print("x:",x)
    print("y:",y)
    ax.scatter(x, y, label = Model, alpha = 0.3)
    ax.legend()

ax.set_xlabel('Deviation', fontsize=15)
ax.set_ylabel('Sample Return - Predicted Return', fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.hlines(0, 0, np.max(x), colors="r", linestyles='solid')
plt.savefig('./Figure/ExpRet-PreRet-Deviation_'+ENV_name+'.pdf', bbox_inches = 'tight')
plt.show()

"""

