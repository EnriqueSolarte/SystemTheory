"""
Description:
    You are going to implement Dyna-Q, a integration of model-based and model-free methods. 
    Please follow the instructions to complete the assignment.
"""
import numpy as np
from copy import deepcopy


def choose_action(state, q_value, maze, epislon):
    """
    Description:
        choose the action using epislon-greedy policy
    """
    if np.random.random() < epislon:
        return np.random.choice(maze.actions)
    else:
        values = q_value[state[0], state[1], :]
        return np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])


def dyna_q(args, q_value, model, maze):
    """
    Description:
        Dyna-Q algorithm is here :)
    Inputs:
        args:    algorithm parameters
        q_value: Q table to maintain.
        model:   The internal model learned by Dyna-Q 
        maze:    Maze environment
    Return:
        steps:   Total steps taken in an episode.
    TODO:
        Complete the algorithm.
    """
    s_0 = maze.START_STATE
    steps = 0

    while s_0 not in maze.GOAL_STATES:

        # get action
        a = choose_action(s_0, q_value, maze, args.epislon)

        # take action
        s_1, reward = maze.step(s_0, a)

        # Q-Learning update rule
        target = reward + args.gamma * np.max(q_value[s_1[0], s_1[1], :])
        error = target - q_value[s_0[0], s_0[1], a]
        q_value[s_0[0], s_0[1], a] = q_value[s_0[0], s_0[1], a] + args.alpha * error

        # feed the internal model
        model.store(s_0, a, s_1, reward)

        # Planning from the internal model
        for t in range(0, args.plan_step):
            s_0i, a_i, s_1i, reward_i = model.sample()

            # Q-Learning update rule using internal Model
            target = reward_i + args.gamma * np.max(q_value[s_1i[0], s_1i[1], :])
            error = target - q_value[s_0i[0], s_0i[1], a_i]
            q_value[s_0i[0], s_0i[1], a_i] = q_value[s_0i[0], s_0i[1], a_i] + args.alpha * error

        steps += 1
        s_0 = s_1

    return steps 


class InternalModel(object):
    """
    Description:
        We'll create a tabular model for our simulated experience. Please complete the following code.
    """

    def __init__(self):
        self.model = dict()
        self.rand = np.random

    def store(self, state, action, next_state, reward):
        """
        TODO:
            Store the previous experience into the model.
        Return:
            NULL
        """
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        if tuple(state) not in self.model.keys():
            self.model[tuple(state)] = dict()
        self.model[tuple(state)][action] = [list(next_state), reward]

    def sample(self):
        """
        TODO:
            Randomly sample previous experience from internal model.
        Return:
            state, action, next_state, reward
        """
        state_index = self.rand.choice(range(len(self.model.keys())))
        state = list(self.model)[state_index]
        action_index = self.rand.choice(range(len(self.model[state].keys())))
        action = list(self.model[state])[action_index]
        next_state, reward = self.model[state][action]
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        return list(state), action, list(next_state), reward
