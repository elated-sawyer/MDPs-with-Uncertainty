# Robust Satisficing MDPs

## Experiments Files
**main.py**: conduct experiments on [Improvements on Percentile], save the estimated transition kernels in "Sampling", the results in "EXP_Results". <br>
**ResultsPlots.ipynb**: plot the results obtained from "main" file.<br>
**EXP2.py**: conduct experiments on [Target-oriented feature], plot the results, and save the plots in "Figure".<br>
*“Backup” section in different files* should be ignored.<br>

## Function Files
**MOD.py**: Optimization models: Dual MDP; RSMDP.<br>
**AGENT.py**: make agent take actions according to different requests (e.g., sampling, test policy).<br>
**ENV.py**: experimental environments, including ENV_GW, ENV_MR, ENV_RiSw.<br>
**VI.py**: dynamic programming methods, including value iteration, robust value iteration(RMDP).<br>
**SAMpling.py**: estimate transition kernel under different purposes.<br>

## Data
**EXP_Results_...**: experimental results in [Improvements on Percentile]<br>
>TM-O: estimate transition kernels without prior knowledge[MR,RiSw]; TM-M: estimate transition kernels with prior knowledge[GW]<br>
L1: L1-norm<br>
EVA-5000: test size 5000<br>
MSK: Mosek solver

**Sampling_...**: estimated transition kernels in [Improvements on Percentile]<br>
**TOF_GenMat**: generated transition kernels in [Target-oriented feature]<br>
**Figure**: plots in [Improvements on Percentile] and [Target-oriented feature]<br>
**Table**: experimental results in [Target-oriented feature].
