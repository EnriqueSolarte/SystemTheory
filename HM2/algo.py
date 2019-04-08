import sys
import gym
import itertools
import numpy as np
from collections import defaultdict


def epsilon_greedy_policy(Q, epsilon, num_of_action):
    """
    Description:
        Epsilon-greedy policy based on a given Q-function and epsilon.
        Don't need to modify this :) 
    """

    def policy_fn(obs):
        A = np.ones(num_of_action, dtype=float) * epsilon / num_of_action
        best_action = np.argmax(Q[obs])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control.

    Inputs:
        env: Environment object.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action.

    Returns:
        Q: the optimal action-value function, a dictionary mapping state -> action values.
        episode_rewards: reward array for every episode
        episode_lengths: how many time steps be taken in each episode
    """

    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    episode_lengths = np.zeros(num_episodes)
    episode_rewards = np.zeros(num_episodes)

    # The policy we're following
    policy = epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    # start training
    for i_episode in range(num_episodes):
        # Reset the environment and pick the first action
        state = env.reset()
        for t in itertools.count():
            # Take action from state S and the env environment
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            # Update rewards
            episode_rewards[i_episode] += reward
            episode_lengths[i_episode] = t

            # Dynamic Update
            best_next_action_idx = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action_idx]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if done:
                break

            state = next_state

    return Q, episode_rewards, episode_lengths


def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    SARSA algorithm: On-policy TD control.

    Inputs:
        env: environment object.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        episode_rewards: reward array for every episode
        episode_lengths: how many time steps be taken in each episode
    """
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    episode_lengths = np.zeros(num_episodes)
    episode_rewards = np.zeros(num_episodes)

    # The policy we're following
    policy = epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        # Reset the environment and pick the first action
        state = env.reset()
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        for t in itertools.count():
            # Take action from state S and the env environment
            next_state, reward, done, _ = env.step(action)

            # Next action
            n_act_probs = policy(next_state)
            n_act_idx = np.random.choice(np.arange(len(n_act_probs)), p=n_act_probs)

            # Update rewards
            episode_rewards[i_episode] += reward
            episode_lengths[i_episode] = t

            # TD Update
            td_target = reward + discount_factor * Q[next_state][n_act_idx]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if done:
                break

            action = n_act_idx
            state = next_state

    return Q, episode_rewards, episode_lengths
