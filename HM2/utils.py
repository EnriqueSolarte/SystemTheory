"""
Description:
    Some helper functions are implemented here.
    You can implement your own plotting function if you want to show extra results :).
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools


def plot(sum_reward, label, yxis_label=None):
    """
    Function to plot the results.
    
    Input:
        avg_reward: Reward averaged from multiple experiments. Size = [exps, timesteps]
        label: label of each line. Size = [exp_name]
    
    """
    assert len(label) == sum_reward.shape[0]

    # define the figure object
    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(111)

    # We show the reward curve
    steps = np.shape(sum_reward)[1]

    for i in range(len(label)):
        ax1.plot(range(steps), sum_reward[i], label=label[i])
    ax1.legend()
    ax1.set_xlabel("Episode")
    if yxis_label is None:
        ax1.set_ylabel("Sum of rewards during episode")
    else:
        ax1.set_ylabel(yxis_label)
    ax1.grid('k', ls='--', alpha=0.3)

    plt.show()


def my_render_trajectory(env, Q, idx, lb):
    state = env.reset()
    map = np.ones((4, 12, 3))

    for t in itertools.count():
        action = np.argmax(Q[state])
        next_state, reward, done, _ = env.step(action)
        env.render()
        draw_action(state, next_state, map, lb[idx])
        if done:
            plt.show()
            break
        state = next_state


def get_position(state):
    if (state + 1) % 12 == 0:
        v = 11
    else:
        v = (state + 1) % 12 - 1
    u = (state - v) / 12

    return int(u), int(v)


def draw_action(st, n_st, map, lb):
    u_0, v_0 = get_position(st)
    u_1, v_1 = get_position(n_st)

    # map[u_0, v_0, lb] = 0
    # map[u_1, v_1, lb] = 0
    map[3, 1:11, 1:3] = 0
    plt.imshow(map)
    plt.title("{} Trajectory".format(lb))
    x = 1
    y = 1
    delta_x = (v_1 - v_0) * x
    delta_y = (u_1 - u_0) * y

    plt.arrow(v_0 * x, u_0 * y, delta_x, delta_y, length_includes_head=True,
              head_width=0.5, head_length=0.5)
