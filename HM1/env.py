'''
Description:
    This is an implementation of multi-armed bandit environment.
    The rewards are modeled by two distributions:
        1. Normal distribution but with different mean and variance (follow Fig 2.1 in the textbook)
        2. Bernoulli distribution
    NOTE: Please DO NOT edit this file!!!
Author: 
    Chan-Wei Hu
'''
import random
import numpy as np


class Bernoulli_MAB(object):
    def __init__(self, NumofBandits=10):
        """
        Create multi-armed bandit with bernoulli distribution for success.
        
        Input:
            NumofBandits: number of bandits
        """
        self.NumofBandits = NumofBandits

        # define success probabilities of each arm
        self.prob = np.random.random(self.NumofBandits)

    def step(self, action):
        """
        Return reward for each bandit. We sample the reward from normal distribution.
        
        Input:
            action: which arm did you pull?

        """

        # handling error
        assert (action < self.NumofBandits), "[ERROR] un-identified arm"

        # return reward. return 1 for success while 0 for failure
        return int(np.random.random() < self.prob[action])


class Gaussian_MAB(object):
    def __init__(self, NumofBandits=10, MeanRange=[-5, 5], sigma=1):
        """
        Create multi-armed bandit with gaussian distribution reward.
        There is no need to specify the probability for each bandit, we generate it by random :)
        
        Input:
            NumofBandits: number of bandits
            MeanRange: mean value of reward for all bandit, we'll generate random value for it by given range
            sigma: standard deviation for all bandit, we'll set it to 0.1 by default
        """
        self.NumofBandits = NumofBandits
        self.sigma = sigma
        # Random generate the mean value of each action
        self.MeanList = np.random.uniform(MeanRange[0], MeanRange[1], self.NumofBandits)

    def step(self, action):
        """
        Return reward for each bandit. We sample the reward from normal distribution.
        
        Input:
            action: which arm did you pull?

        """

        # handling error
        assert (action < self.NumofBandits), "[ERROR] un-identified arm"

        # return reward via sampling from normal distribution
        return np.random.normal(self.MeanList[action], self.sigma, 1)[0]
