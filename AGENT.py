import numpy as np

# take action and collect episode return based on the given policy (deterministic or randomized)
def run_single(ENV, nS, nA, policy, iter_tot, gamma, probs = []):
    episode_reward = 0
    ob = np.random.choice(nS)
    for t in range(iter_tot):
        # deterministic or randomized policy
        if len(probs):
            a = np.random.choice(nA, p=probs[ob])   # policy
        else:
            a = policy[ob]
        ob_next, rew = ENV.Tran(ob, a)
        episode_reward += (gamma**t)*rew
        ob = ob_next
    return episode_reward

# mainly for sampling part, collect <s,a,s',r> to estimate transition kernel
def run_single_random(ENV, nS, nA, iter_tot, gamma, ob_g, probs = None):
    episode_reward = 0
    # continue with the last episode's final state
    ob = ob_g
    P_count = np.zeros([nS, nA, nS])
    for t in range(iter_tot):
        if probs == None:
            a = np.random.choice(nA)   # random policy
        else:
            a = np.random.choice(nA, p=probs[ob])
        ob_next, rew = ENV.Tran(ob, a)
        episode_reward += (gamma**t)*rew
        P_count[ob, a, ob_next] += 1
        ob = ob_next
    return episode_reward, P_count, ob