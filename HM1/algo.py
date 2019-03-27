"""
Todo:
    Complete three algorithms. Please follow the instructions for each algorithm. Good Luck :)
"""
import numpy as np


class EpislonGreedy(object):
    """
    Implementation of epislon-greedy algorithm.
    """

    def __init__(self, NumofBandits=10, epsilon=0.1):
        """
        Initialize the class.
        Step 1: Initialize your Q-table and counter for each action
        """
        assert (0. <= epsilon <= 1.0), "[ERROR] Epsilon should be in range [0,1]"
        self._epsilon = epsilon
        self._nb = NumofBandits
        self._Q = np.zeros(self._nb, dtype=float)
        self._action_N = np.zeros(self._nb, dtype=int)

    def update(self, action, immi_reward):
        """
        Step 2: update your Q-table. No need to return any result.
        """
        self._Q[action] = self._Q[action] + (1 / self._action_N[action]) * (immi_reward - self._Q[action])

    def act(self, t):
        """
        Step 3: Choose the action via greedy or explore.
        Return: action selection
        """
        if self._epsilon > np.random.rand() or t is 0:
            a_max = np.random.randint(0, self._Q.size)
        else:
            a_max = np.argmax(self._Q)
        self._action_N[a_max] += 1

        return a_max


class UCB(object):
    """
    Implementation of upper confidence bound.
    """

    def __init__(self, NumofBandits=10, c=0):
        """
        Initailize the class.
        Step 1: Initialize your Q-table and counter for each action
        """
        self._nb = NumofBandits
        self._c = c
        self._Q = np.zeros(self._nb, dtype=float)
        self._action_N = np.ones(self._nb, dtype=int)

    def update(self, action, immi_reward):
        """
        Step 2: update your Q-table
        """
        self._Q[action] = self._Q[action] + (1 / self._action_N[action]) * (immi_reward - self._Q[action])

    def act(self, t):
        """
        Step 3: use UCB action selection. We'll pull all arms once first!
        HINT: Check out p.27, equation 2.8
        """
        const = self._c * np.sqrt(np.log(t + 1))
        n_action = np.sqrt(self._action_N)

        q_iter = np.divide(const, n_action)
        a_max = np.argmax(self._Q + q_iter)
        self._action_N[a_max] += 1
        return a_max


class Gradient(object):
    """
    Implementation of your gradient-based method
    """

    def __init__(self, NumofBandits=10, lda=0.5):
        """
        Initailize the class.
        Step 1: Initialize your Q-table and counter for each action
        """
        self._nb = NumofBandits
        self._Q = np.zeros(self._nb, dtype=float)
        self._action_N = np.zeros(self._nb, dtype=int)
        self._action_Pr = np.ones(self._nb, dtype=int)
        assert (0. <= lda <= 1.0), "[ERROR] Epsilon should be in range [0,1]"
        self._lda = lda

    def update(self, action, immi_reward):
        """
        Step 2: update your Q-table
        """
        self._Q[action] = self._Q[action] + (1 / self._action_N[action]) * (immi_reward - self._Q[action])

        q_mean = np.mean(self._Q)

        aux_action = self._action_Pr[action]
        self._action_Pr = self._action_Pr - self._lda * (self._Q[action] - q_mean) * self._action_Pr
        self._action_Pr[action] = aux_action + self._lda * (self._Q[action] - q_mean) * (1 - aux_action)

    def act(self, t):
        """
        Step 3: select action with gradient-based method
        HINT: Check out p.28, eq 2.9 in your textbook
        """

        self._action_Pr = np.exp(self._action_Pr) / np.sum(np.exp(self._action_Pr))
        a_max = np.argmax(self._action_Pr)
        self._action_N[a_max] += 1
        return a_max
