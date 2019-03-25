"""
Todo:
    Complete three algorithms. Please follow the instructions for each algorithm. Good Luck :)
"""
import numpy as np

class EpislonGreedy(object):
    """
    Implementation of epislon-greedy algorithm.
    """
    def __init__(self, NumofBandits=10, epislon=0.1):
        """
        Initialize the class.
        Step 1: Initialize your Q-table and counter for each action
        """
        assert (0. <= epislon <= 1.0), "[ERROR] Epislon should be in range [0,1]"
        self._epislon = epislon
        self._nb = NumofBandits
        self._Q = np.zeros(self._nb, dtype=float)
        self._action_N = np.zeros(self._nb, dtype=int)

    def update(self, action, immi_reward):
        """
        Step 2: update your Q-table. No need to return any result.
        """
        ################### Your code here #######################
        raise NotImplementedError('[EpislonGreedy] update function NOT IMPLEMENTED')
        ##########################################################

    def act(self, t):
        """
        Step 3: Choose the action via greedy or explore. 
        Return: action selection
        """
        ################### Your code here #######################
        raise NotImplementedError('[EpislonGreedy] act function NOT IMPLEMENTED')
        ##########################################################

class UCB(object):
    """
    Implementation of upper confidence bound.
    """
    def __init__(self, NumofBandits=10, c=2):
        """
        Initailize the class.
        Step 1: Initialize your Q-table and counter for each action
        """
        self._nb = NumofBandits
        self._c = c
        self._Q = np.zeros(self._nb, dtype=float)
        self._action_N = np.zeros(self._nb, dtype=int)

    def update(self, action, immi_reward):
        """
        Step 2: update your Q-table
        """
        ################### Your code here #######################
        raise NotImplementedError('[UCB] update function NOT IMPLEMENTED')
        ##########################################################

    def act(self, t):
        """
        Step 3: use UCB action selection. We'll pull all arms once first!
        HINT: Check out p.27, equation 2.8
        """
        ################### Your code here #######################
        raise NotImplementedError('[UCB] act function NOT IMPLEMENTED')
        ##########################################################

class Gradient(object):
    """
    Implementation of your gradient-based method
    """
    def __init__(self, NumofBandits=10, epislon=0.1):
        """
        Initailize the class.
        Step 1: Initialize your Q-table and counter for each action
        """
        self._nb = NumofBandits
        self._Q = np.zeros(self._nb, dtype=float)
        self._action_N = np.zeros(self._nb, dtype=int)

    def update(self, action, immi_reward):
        """
        Step 2: update your Q-table
        """
        ################### Your code here #######################
        raise NotImplementedError('[gradient] update function NOT IMPLEMENTED')
        ##########################################################

    def act(self, t):
        """
        Step 3: select action with gradient-based method
        HINT: Check out p.28, eq 2.9 in your textbook
        """
        ################### Your code here #######################
        raise NotImplementedError('[gradient] act function NOT IMPLEMENTED')
        ##########################################################
