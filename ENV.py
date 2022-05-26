import numpy as np

class ENV_GW(object):
    def __init__(self):
        self.nS = 24
        self.nA = 4
        self.nJ = 2
        self.nI = 12
        self.R = np.array([ 0,  3, 21, 27,  6,  0,  0,  0,  0,  0, 15, 24])
        self.RR = np.array([0, 3, 21, 27, 6, 0, 0, 0, 0, 0, 15, 24, 0, 3, 21, 27, 6, 0, 0, 0, 0, 0, 15, 24])
        #self.R_sa = np.stack((RR, RR, RR, RR), axis=1)
        self.R_sa = np.array([[ 0,  0,  0,  0],[ 3,  3,  3,  3],[21, 21, 21, 21],[27, 27, 27, 27],[ 6,  6,  6,  6],
                               [ 0,  0,  0,  0],[ 0,  0,  0,  0],[ 0,  0,  0,  0],[ 0,  0,  0,  0],[ 0,  0,  0,  0],
                               [15, 15, 15, 15],[24, 24, 24, 24],[ 0,  0,  0,  0],[ 3,  3,  3,  3],[21, 21, 21, 21],
                               [27, 27, 27, 27],[ 6,  6,  6,  6],[ 0,  0,  0,  0],[ 0,  0,  0,  0],[ 0,  0,  0,  0],
                               [ 0,  0,  0,  0],[ 0,  0,  0,  0],[15, 15, 15, 15],[24, 24, 24, 24]])

    def Tran(self, ob, a):
        z = np.array([0.9, 0.2])
        P_ran = np.random.dirichlet([1, 1, 1, 1, 1, 1, 10, 20, 30, 3, 4, 5], 2)
        i = np.mod(ob, 12)
        j = np.floor_divide(ob, 12)

        if np.random.uniform(0, 1) > z[j]:
            if a == 0:  # right
                k = i + 1
            elif a == 1:  # left
                k = i - 1
            else:
                k = np.random.choice(12, p=P_ran[j])
            k = np.maximum(np.minimum(k,  self.nI - 1), 0)
        else:
            k = np.random.choice(12, p=P_ran[j])

        if a == 2:
            l = j + 1
        elif a == 3:
            l = j - 1
        else:
            e = np.random.uniform(0, 1)
            if e <= 0.35:
                l = j + 1
            elif e <= 0.7:
                l = j - 1
            else:
                l = j
        l = np.maximum(np.minimum(l,  self.nJ - 1), 0)
        rew = self.R[k]
        ob_next = k + l * 12
        return ob_next, rew

class ENV_MR(object):
    def __init__(self):
        self.nS = 10
        self.nA = 2
        self.R = np.array([20, 20, 20, 20, 20, 20, 20,  0, 18, 10])
        #self.R_sa = np.stack((R, R), axis=1)
        self.R_sa = np.array([[20, 20],[20, 20],[20, 20],[20, 20],[20, 20],
                               [20, 20],[20, 20],[ 0,  0],[18, 18],[10, 10]])

    def Tran(self, ob, a):
        if ob == 9:
            if a == 0:
                ob_next = ob
            else:
                if np.random.uniform(0, 1) >= 0.4:
                    ob_next = ob-1
                else:
                    ob_next = ob
        elif ob == 8:
            if a == 0:
                if np.random.uniform(0, 1) >= 0.2:
                    ob_next = 0
                else:
                    ob_next = ob
            else:
                ob_next = ob
        elif ob == 7:
            if a == 0:
                ob_next = ob
            else:
                temp = np.random.uniform(0, 1)
                if temp >= 0.9:
                    ob_next = 9
                elif temp >= 0.6:
                    ob_next = ob
                else:
                    ob_next = 8
        else:
            if a == 0:
                if np.random.uniform(0, 1) >= 0.2:
                    ob_next = ob+1
                else:
                    ob_next = ob
            else:
                temp = np.random.uniform(0, 1)
                if temp >= 0.9:
                    ob_next = 9
                elif temp >= 0.6:
                    ob_next = ob+1
                else:
                    ob_next = 8
        rew = self.R[ob_next]
        return ob_next, rew

class ENV_RiSw(object):
    def __init__(self):
        self.nS = 10
        self.nA = 2
        self.R = np.array([5, 0, 0, 10, 10, 10, 10, 10, 10, 15])
        self.R_sa = np.array([[5, 5], [0, 0], [0, 0], [10, 10], [10, 10], [10, 10], [10, 10], [10, 10], [10, 10], [15, 15]])

    def Tran(self, ob, a):
        if ob == 0:
            if a == 0:
                ob_next = ob
            else:
                if np.random.uniform(0, 1) >= 0.3:
                    ob_next = ob
                else:
                    ob_next = ob+1
        elif ob == self.nS-1:
            if a == 0:
                ob_next = ob-1
            else:
                if np.random.uniform(0, 1) >= 0.3:
                    ob_next = ob-1
                else:
                    ob_next = ob
        else:
            if a == 0:
                ob_next = ob-1
            else:
                temp = np.random.uniform(0, 1)
                if temp >= 0.9:
                    ob_next = ob-1
                elif temp >= 0.3:
                    ob_next = ob
                else:
                    ob_next = ob+1
        rew = self.R[ob_next]
        return ob_next, rew



#########################
######## Backup #########
#########################

"""class ENV_IC(object):

    def __init__(self):
        # Capacity: 10
        self.nS = 11  #-7~3
        self.nA = 4  # 0,1,2,3
        #Bsed on non-negative assumption
        self.R_sa = np.array([[ 0.,  0.,  0.,  0.],
       [ 4.,  4.,  4.,  4.],
       [ 8.,  8.,  8.,  8.],
       [16., 16., 16., 16.],
       [24., 24., 24., 24.],
       [27., 27., 27., 27.],
       [30., 30., 30., 30.],
       [30., 30., 30., 30.],
       [30., 30., 30.,  0.],
       [32., 32.,  0.,  0.],
       [34.,  0.,  0.,  0.]])

    def Tran(self, ob, a):
        P_ran = np.random.dirichlet([10, 20, 40, 30])
        sales = np.random.choice([0,2,4,6], p=P_ran)
        if ob + a >= self.nS - 1:
            ob_next = self.nS - 1 - sales
        elif ob + a >= sales:
            ob_next = ob + a - sales
        else:
            ob_next = ob + a
        rew = self.R_sa[ob_next][a]
        return ob_next, rew

    ## Calculate Reward Matrix
    def F(self, u):
    # F Returns expected revenue, u = s+a
        if u >= 6:
            output = 38
        elif u >= 4:
            output = 20
        elif u >= 2:
            output = 4
        else:
            output = 0
        return output

    def O(self, a):
    # O Returns cost of ordering units
        if a == 0:
            cost = 0
        elif a >= 1:
            cost = 4 + 2 * a
        return cost

    def R(self, s, a):
    # R Returns expected reward
        reward = self.F(s+a) - self.O(a) - (a+np.max([0,s]))
        return reward

    def Reward_saMatrix(self):
        Reward_sa_Matrix = np.zeros([self.nS, self.nA])
        R_state_sum = np.zeros([self.nS])
        R_state_count = np.zeros([self.nS])
        for s in np.arange(0, 0 + self.nS, 1, dtype=int):
            for a in np.arange(0, 0+self.nA, 1, dtype=int):
                if s+a <= 10:
                    R_state_sum[s] += self.R(s,a)
                    R_state_count[s] += 1
                else:
                    Reward_sa_Matrix[s, a] = -999
            for a in np.arange(0, 0+self.nA, 1, dtype=int):
                if s+a <= 10:
                    Reward_sa_Matrix[s,a] = np.around(R_state_sum[s]/R_state_count[s])
                else:
                    Reward_sa_Matrix[s, a] = 0
        return Reward_sa_Matrix"""

"""class ENV_IC_test(object):
    def __init__(self):
        # Capacity: 10
        self.nS = 11  #-7~3
        self.nA = 4  # 0,1,2,3
        #Bsed on non-negative assumption
        self.R_sa = np.array([[  16.,    3.,    0.,    3.],
                           [  15.,    2.,    5.,    2.],
                           [  14.,    7.,    4.,   21.],
                           [  19.,    6.,   23.,   20.],
                           [  18.,   25.,   22.,   19.],
                           [  37.,   24.,   21.,   18.],
                           [  36.,   23.,   20.,   17.],
                           [  35.,   22.,   19.,   16.],
                           [  34.,   21.,   18., -985.],
                           [  33.,   20., -985., -985.],
                           [  32., -985., -985., -985.]])

    def Tran(self, ob, a):
        if ob + a > self.nS-1:
            if np.random.uniform(0, 1) >= 0.1:
                temp = np.random.uniform(0, 1)
                if temp >= 0.75:
                    ob_next = 10
                elif temp >= 0.25:
                    ob_next = 9
                else:
                    ob_next = 8
            else:
                ob_next = np.random.choice(ob+1)
        elif ob + a == 0:
            ob_next = 0
        elif ob + a == 1:
            if np.random.uniform(0, 1) >= 0.5:
                temp = np.random.uniform(0, 1)
                if temp >= 0.75:
                    ob_next = 1
                else:
                    ob_next = 0
            else:
                ob_next = np.random.choice(ob + 1)
        elif ob + a == 2:
            if np.random.uniform(0, 1) >= 0.5:
                temp = np.random.uniform(0, 1)
                if temp >= 0.75:
                    ob_next = 2
                elif temp >= 0.25:
                    ob_next = 1
                else:
                    ob_next = 0
            else:
                ob_next = np.random.choice(ob+1)
        elif ob + a >= 8:
            if np.random.uniform(0, 1) >= 0.1:
                temp = np.random.uniform(0, 1)
                if temp >= 0.75:
                    ob_next = ob + a
                elif temp >= 0.25:
                    ob_next = ob + a - 1
                else:
                    ob_next = ob + a - 2
            else:
                ob_next = np.random.choice(ob+1)
        else:
            if np.random.uniform(0, 1) >= 0.5:
                temp = np.random.uniform(0, 1)
                if temp >= 0.75:
                    ob_next = ob + a
                elif temp >= 0.25:
                    ob_next = ob + a - 1
                else:
                    ob_next = ob + a - 2
            else:
                ob_next = np.random.choice(ob+1)
        rew = self.R_sa[ob][a]
        return ob_next, rew

    ## Calculate Reward Matrix
    def F(self, u):
    # F Returns expected revenue, u = s+a
        if u >= 5:
            output = 28
        elif u >= 3:
            output = 8
        else:
            output = 2
        return output

    def O(self, a):
    # O Returns cost of ordering units
        if a == 0:
            cost = 0
        elif a >= 1:
            cost = 10 + 2 * a
        return cost

    def R(self, s, a):
    # R Returns expected reward
        reward = self.F(s+a) - self.O(a) - (a+np.max([0,s]))
        return reward

    def Reward_saMatrix(self):
        Reward_sa_Matrix = np.zeros([self.nS, self.nA])
        for s in np.arange(0, 0 + self.nS, 1, dtype=int):
            for a in np.arange(0, 0+self.nA, 1, dtype=int):
                if s+a <= 10:
                    Reward_sa_Matrix[s,a] = self.R(s,a)
                else:
                    Reward_sa_Matrix[s, a] = -999
        return Reward_sa_Matrix"""


"""class ENV_IC(object):
    def __init__(self):
        self.nS = 11  #-7~3
        self.nA = 4  # 0,1,2,3
        #Bsed on non-negative assumption
        self.R_sa = np.array(
            [[0., 1., 6., 11.],
            [3., 4., 9., 14.],
            [6., 7., 12., 17.],
            [9., 10., 15., 20.],
            [12., 13., 18., 23.],
            [15., 16., 21., 24.],
            [18., 19., 22., 21.],
            [21., 20., 19., 16.],
            [26., 21., 18., -978.],
            [27., 20., -978., -978.],
            [26., -978., -978., -978.]])
    def Tran(self, ob, a):
        if ob - 7 + a > 3:
            temp = np.random.uniform(0, 1)
            if temp >= 0.75:
                ob_next = 10
            elif temp >= 0.25:
                ob_next = 9
            else:
                ob_next = 8
        elif ob - 7 + a == -7:
            ob_next = 0
        elif ob - 7 + a == -6:
            temp = np.random.uniform(0, 1)
            if temp >= 0.75:
                ob_next = 1
            else:
                ob_next = 0
        elif ob - 7 + a == -5:
            temp = np.random.uniform(0, 1)
            if temp >= 0.75:
                ob_next = 2
            elif temp >= 0.25:
                ob_next = 1
            else:
                ob_next = 0
        else:
            temp = np.random.uniform(0, 1)
            if temp >= 0.75:
                ob_next = ob + a
            elif temp >= 0.25:
                ob_next = ob + a - 1
            else:
                ob_next = ob + a - 2
        rew = self.R_sa[ob][a]
        return ob_next, rew

    ## Calculate Reward Matrix
    def F(self, u):
    # F Returns expected revenue, u = s+a
        if u <= 0:
            output = 0
        elif u == 1:
            output = 6
        elif u == 2:
            output = 8
        elif u >= 3:
            output = 8
        return output
    def O(self, a):
    # O Returns cost of ordering units
        if a == 0:
            cost = 0
        elif a >= 1:
            cost = 4 + 2 * a
        return cost
    def R(self, s, a):
    # R Returns expected reward
        d = np.absolute(np.min([0,s]))
        reward = self.F(s+a) + 8*(np.min([d,a])) - self.O(a) - (a+np.max([0,s])) - 3*d
        return reward
    def Reward_saMatrix(self):
        Reward_sa_Matrix = np.zeros([self.nS, self.nA])
        for s in np.arange(-7, -7 + self.nS, 1, dtype=int):
            for a in np.arange(0, 0+self.nA, 1, dtype=int):
                if np.max([s,0])+a <= 3:
                    Reward_sa_Matrix[s+7,a] = self.R(s,a)
                else:
                    Reward_sa_Matrix[s+7, a] = -999
        return Reward_sa_Matrix"""