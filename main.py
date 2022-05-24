from AGENT import run_single, run_single_random
from ENV import ENV_MR, ENV_GW, ENV_RiSw
from SAMpling import sampling,sampling_PriorKnowledge,sampling_WE
from VI import value_iteration, value_iteration_robust
from MOD import Dual, RSMDP
import pandas as pd
import numpy as np

###### Evaluation #######
# Collect experimental return based on the given policy
def EVA(Model,eva_size, ENV, nS, nA, policy, sam_size, MaxStePE, gamma, probs = []):
    R_record = np.array([])
    for i in range(eva_size):
        episode_reward = run_single(ENV, nS, nA,  policy, MaxStePE, gamma, probs)
        R_record = np.append(R_record, episode_reward)
    R_record = pd.DataFrame(R_record, columns=[Model])
    if len(probs):
        print("SamSiz" + str(sam_size) + "MaxSte" + str(MaxStePE), Model, probs)
    else:
        print("SamSiz" + str(sam_size) + "MaxSte" + str(MaxStePE), Model, policy)
    print(Model, "MEAN:", np.mean(R_record), "STD:", np.std(R_record))
    print("np.percentile:", np.percentile(R_record, [10, 20, 30]))
    #Record = pd.concat([Record, R_record], axis=1)
    R_record.to_csv("./EXP_Results/%s_SAM%s_MAXST%s_EVA%s.csv" % (Model, sam_size, MaxStePE, eva_size), index=False)
    return

##### Parameter Selection #####
def EVA_cv(eva_size, ENV, nS, nA, policy, MaxStePE, gamma, probs = []):
    R_record = np.array([])
    for i in range(eva_size):
        episode_reward = run_single(ENV, nS, nA,  policy, MaxStePE, gamma, probs)
        R_record = np.append(R_record, episode_reward)
    print("np.percentile:", np.percentile(R_record, [10, 20, 30]))
    return np.percentile(R_record, [10])
# RSMDP
def ParSel_RSMDP(ENV, sam_size, MaxStePE_Sam, MaxStePE, gamma, Target_COEF_List, cv_size):
    P_est = np.load("./Sampling/SamSiz" + str(sam_size) + "MaxSte" + str(MaxStePE_Sam) + ".npy")
    Target, policy = Dual(ENV.nS, ENV.nA, P_est, ENV.R_sa, gamma)
    Target_COEF = Target_COEF_List
    Target_list = Target_COEF*Target
    ## RSMDP_OS ##
    Best_TAU = 0
    Best_Return = 0
    for j in range(len(Target_list)):
        # Obtain policy via solving robust satisficing model using estimated transition matrix
        TAU_real, Probs, RS_Policy = RSMDP(ENV.nS, ENV.nA, P_est, ENV.R_sa, Target_list[j], gamma)
        Model = "RSMDP" + str(Target_COEF[j])
        # Randomized policy
        Return = EVA_cv(cv_size, ENV, ENV.nS, ENV.nA, policy, MaxStePE, gamma, Probs)
        if Return > Best_Return:
            Best_Return = Return
            Best_TAU = Target_COEF[j]
        print("Current Best_Return:", Best_Return, "Current Best_TAU:", Best_TAU)
    return Best_TAU
def ParSel_RMDP(ENV, sam_size, MaxStePE_Sam, MaxStePE, gamma, Radius_List, cv_size):
    P_est = np.load("./Sampling/SamSiz" + str(sam_size) + "MaxSte" + str(MaxStePE_Sam) + ".npy")
    Best_Return = 0
    Best_RAD = 0
    for j in range(len(Radius_List)):
        # Obtain policy via solving robust satisficing model using estimated transition matrix
        V_vi, R_policy = value_iteration_robust(P_est, ENV.nS, ENV.nA, ENV.R_sa, gamma, MaxStePE, tol, Radius_List[j])
        Model = "RMDP" + str(Radius_List[j])
        # Deterministic policy
        Return = EVA_cv(cv_size, ENV, ENV.nS, ENV.nA, R_policy, MaxStePE, gamma)
        if Return > Best_Return:
            Best_Return = Return
            Best_RAD = Radius_List[j]
        print("Current Best_Return:", Best_Return, "Current Best_RAD:", Best_RAD)
    return Best_RAD


#######################
####  Parameters  #####
#######################
sam_size_WE = 1500
MaxStePE_WE = 10000
MaxStePE_Sam = 10  #GP:100; RiSw:10,; MR:10; Predicted Value Record:100
MaxStePE = 100
MaxEPS = 26
sampling_seed = 2 
gamma = 0.85
eva_size = 5000
tol = 1e-3
ENV = ENV_RiSw()  ####


#####################
####  sampling  #####
#####################
# provide a well-estimated transition kernel (with large sampling size)
"""sampling_WE(ENV, ENV.nS, ENV.nA, gamma, sampling_seed, sam_size_WE, MaxStePE_WE)
"""
# estimate transition kernels without prior knowledge[MR,RiSw]
"""sampling(ENV, ENV.nS, ENV.nA, gamma, sampling_seed, MaxStePE_Sam, MaxEPS)
"""
# estimate transition kernels with prior knowledge[GW]
# (each sa pair's next state space is given to the agent at the beginning)[GW]
"""MatrixWE = np.load("./Sampling/sam_size_WE" + str(sam_size_WE) + "MaxStePE_WE" + str(MaxStePE_WE) + ".npy")
print(MatrixWE)
sampling_PriorKnowledge(ENV, MatrixWE,ENV.nS, ENV.nA, gamma, sampling_seed, MaxStePE_Sam, MaxEPS)"""


#####################
#### Evaluation #####
#####################

## Parameter Selection ##
############################
# Target-Oriented MDP
sam_size = 5
Target_COEF_List = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
cv_size = 100
Best_TAU = ParSel_RSMDP(ENV, sam_size, MaxStePE_Sam, MaxStePE, gamma, Target_COEF_List, cv_size)
## Robust MDP ##
sam_size = 5
Radius_List = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8])
cv_size = 100
Best_RAD = ParSel_RMDP(ENV, sam_size, MaxStePE_Sam, MaxStePE, gamma, Radius_List, cv_size)


# Collect experimental return based on the given policy
############################
for sam_size in np.arange(1, MaxEPS, 1, dtype=int):
    P_est = np.load("./Sampling/SamSiz" + str(sam_size) + "MaxSte" + str(MaxStePE_Sam) + ".npy")

    ## NMDP ##
    Model = "NMDP"
    V_vi, policy = value_iteration(P_est, ENV.nS, ENV.nA, ENV.R_sa, gamma, MaxStePE, tol)
    EVA(Model, eva_size, ENV, ENV.nS, ENV.nA, policy, sam_size, MaxStePE, gamma)
    ## RSMDP ##
    Target, policy = Dual(ENV.nS, ENV.nA, P_est, ENV.R_sa, gamma)
    Target_COEF = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 0.97, 0.98, 0.99])
    Target_list = Target_COEF*Target
    for j in range(len(Target_list)):
        # Obtain policy via solving robust satisficing model using estimated transition matrix
        TAU_real, Probs, RS_Policy = RSMDP(ENV.nS, ENV.nA, P_est, ENV.R_sa, Target_list[j], gamma)
        print(Probs.sum(axis=1))
        Model = "RSMDP" + str(Target_COEF[j])
        # Deterministic policy
        EVA(Model, eva_size, ENV, ENV.nS, ENV.nA, RS_Policy, sam_size, MaxStePE, gamma, Probs)
    ## RMDP ##
    radius = np.array([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8])
    for j in range(len(radius)):
        V_vi, R_policy = value_iteration_robust(P_est, ENV.nS, ENV.nA, ENV.R_sa, gamma, MaxStePE, tol, radius[j])
        Model = "RMDP" + str(radius[j])
        EVA(Model, eva_size, ENV, ENV.nS, ENV.nA, R_policy, sam_size, MaxStePE, gamma)


# Collect experimental return with the well estimated transition kernel
############################
P_est = np.load("./Sampling/sam_size_WE" + str(sam_size_WE) + "MaxStePE_WE" + str(MaxStePE_WE) + ".npy")
print(P_est)

## NMDP ##
Model = "NMDP"
V_vi, policy = value_iteration(P_est, ENV.nS, ENV.nA, ENV.R_sa, gamma, MaxStePE, tol)
EVA(Model, eva_size, ENV, ENV.nS, ENV.nA, policy, sam_size_WE, MaxStePE, gamma)
## RSMDP ##
Target, policy = Dual(ENV.nS, ENV.nA, P_est, ENV.R_sa, gamma)
Target_COEF = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 0.97, 0.98, 0.99])
Target_list = Target_COEF*Target
for j in range(len(Target_list)):
    # Obtain policy via solving robust satisficing model using estimated transition matrix
    TAU_real, Probs, RS_Policy = RSMDP(ENV.nS, ENV.nA, P_est, ENV.R_sa, Target_list[j], gamma)
    Model = "RSMDP" + str(Target_COEF[j])
    # Deterministic policy
    EVA(Model, eva_size, ENV, ENV.nS, ENV.nA, RS_Policy, sam_size_WE, MaxStePE, gamma, Probs)
## RMDP ##
radius = np.array([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8])
for j in range(len(radius)):
    V_vi, R_policy = value_iteration_robust(P_est, ENV.nS, ENV.nA, ENV.R_sa, gamma, MaxStePE, tol, radius[j])
    Model = "RMDP" + str(radius[j])
    EVA(Model, eva_size, ENV, ENV.nS, ENV.nA, R_policy, sam_size_WE, MaxStePE, gamma)



#########################
######## Backup #########
#########################

#####################
#### Record_TAU
#####################
"""R_record = np.array([])
for sam_size in np.arange(1, MaxEPS, 1, dtype=int):
    P_est = np.load("./Sampling/SamSiz" + str(sam_size) + "MaxSte" + str(MaxStePE_Sam) + ".npy")
    Target, policy = Dual(ENV.nS, ENV.nA, P_est, ENV.R_sa, gamma)
    R_record = np.append(R_record, Target)
np.save("./Sampling/TAU_Record", R_record)"""

##########################
#### Record_ValueFunction_NMDP
##########################
"""R_record = np.array([])
R_record_Comp = np.array([])
for sam_size in np.arange(1, MaxEPS, 1, dtype=int):
    P_est = np.load("./Sampling/SamSiz" + str(sam_size) + "MaxSte" + str(MaxStePE_Sam) + ".npy")

    V_vi, policy = value_iteration(P_est, ENV.nS, ENV.nA, ENV.R_sa, gamma, MaxStePE, tol)
    R_record = np.append(R_record, V_vi.sum() / ENV.nS)

    Target, policy = Dual(ENV.nS, ENV.nA, P_est, ENV.R_sa, gamma)
    R_record_Comp = np.append(R_record_Comp, Target)

print("Predicted Return by VI", R_record)
print("Predicted Return by R*U", R_record_Comp)
np.save("./Sampling/PreRet_NMDP", R_record)"""

##########################
#### Record_(Robust)ValueFunction_RMDP
##########################
"""radius = np.array([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8])
for Rad in radius:
    R_record = np.array([])
    for sam_size in np.arange(1, MaxEPS, 1, dtype=int):
        P_est = np.load("./Sampling/SamSiz" + str(sam_size) + "MaxSte" + str(MaxStePE_Sam) + ".npy")
        V_vi, R_policy = value_iteration_robust(P_est, ENV.nS, ENV.nA, ENV.R_sa, gamma, MaxStePE, tol, Rad)
        R_record = np.append(R_record, V_vi.sum()/ENV.nS)
    print("Predicted Return by RMDP" + str(Rad), R_record)
    np.save("./Sampling/PreRet_RMDP" + str(Rad), R_record)"""

##########################
#### Record_TAU-Real
##########################
"""Target_COEF = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 0.97, 0.98, 0.99])
for TAU in Target_COEF:
    R_record = np.array([])
    for sam_size in np.arange(1, MaxEPS, 1, dtype=int):
        P_est = np.load("./Sampling/SamSiz" + str(sam_size) + "MaxSte" + str(MaxStePE_Sam) + ".npy")
        Target, policy = Dual(ENV.nS, ENV.nA, P_est, ENV.R_sa, gamma)
        TAU_real, Probs, RS_Policy = RSMDP(ENV.nS, ENV.nA, P_est, ENV.R_sa, TAU*Target, gamma)
        R_record = np.append(R_record, TAU_real)
    print("Predicted Return by RSMDP" + str(TAU), R_record)
    np.save("./Sampling/PreRet_RSMDP" + str(TAU), R_record)"""


##########################
# Cross Validation
############################
"""for sam_size in [1,5,10,15,20,25]:

    P_est = np.load("./Sampling/SamSiz" + str(sam_size) + "MaxSte" + str(MaxStePE_Sam) + ".npy")

    Target, policy = Dual(ENV.nS, ENV.nA, P_est, ENV.R_sa, gamma)
    Target_COEF = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
    Target_list = Target_COEF*Target
    ## RSMDP_OS ##
    for j in range(len(Target_list)):
        # Obtain policy via solving robust satisficing model using estimated transition matrix
        TAU_real, Probs, RS_Policy = RSMDP(ENV.nS, ENV.nA, P_est, ENV.R_sa, Target_list[j], gamma)
        Model = "RSMDP" + str(Target_COEF[j])
        # Deterministic policy
        EVA(Model, eva_size, ENV, ENV.nS, ENV.nA, RS_Policy, sam_size, MaxStePE, gamma, Probs)

    radius = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8])
    for j in range(len(radius)):
        V_vi, R_policy = value_iteration_robust(P_est, ENV.nS, ENV.nA, ENV.R_sa, gamma, MaxStePE, tol, radius[j])
        Model = "RMDP" + str(radius[j])
        EVA(Model, eva_size, ENV, ENV.nS, ENV.nA, R_policy, sam_size, MaxStePE, gamma)"""