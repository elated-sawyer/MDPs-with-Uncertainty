# Robust MDPs
Despite being a fundamental building block for reinforcement learning, Markov decision processes (MDPs) often suffer from ambiguity in model parameters. Robust MDPs are proposed to overcome this challenge by optimizing the worst-case performance under ambiguity.<br>
The performances of NMDPs, RMDPs, DRMDPs are evaluated in three applications: river swim, machine replacement and grid world.

## Environments
We use a discounted factor $\gamma=0.85$ for all environments, and the objective is always maximizing (total discounted) rewards.

* Machine Replacement: we have 2 repair options constituting our action set ["repair", "do nothing"] and 10 states.
The rewards relate only to the states, which are [20, 20, 20, 20, 20, 20, 20,  0, 18, 10].

* River Swim: we have 2 swimming directions constituting our action set ["move left", "move right"] and 10 states, and the rewards relate only to the state, which are [5, 0, 0, 10, 10, 10, 10, 10, 10, 15]. 

* Grid World: the grid world has two rows and 12 columns, and the rewards relate to the column indices only, which are [0, 3, 21, 27,  6,  0,  0,  0,  0,  0, 15, 24]. There are four available actions, "move up" and "move down" for vertical moves (that decreases and increases the column index, respectively), as well as "move left", and "move right" for horizontal moves (that decreases and increases the row index, respectively). Horizontal moves have a chance of failure that only related to row indices (0.9 for the first row and 0.2 for the second). Failing a transfer or selecting a vertical move would generate the column index of the next state according to a Dirichlet distribution. After selecting a horizontal move, the agent will randomly go up, go down, or stay 
with probabilities $0.35$, $0.35$ and 0.3, respectively.

# Files Intro
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
