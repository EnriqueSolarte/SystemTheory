import matplotlib.pyplot as plt
import numpy as np
import pickle


def plot(rewards, div=10):
    average_reward = []
    max_reward = []
    min_reward = []
    std_reward = []
    for i in range(0, len(rewards), div):
        average_reward.append(np.mean(rewards[i:i + div]))
        max_reward.append(np.max(rewards[i:i + div]))
        min_reward.append(np.min(rewards[i:i + div]))
        std_reward.append(np.std(rewards[i:i + div]))

    plt.plot(average_reward)
    # plt.plot(max_reward)
    # plt.plot(min_reward)
    plt.show()


def save_obj(filename, obj, verbose=True):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    if verbose:
        print("obj {} saved".format(filename))


def load_obj(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)


if __name__ == '__main__':
    # episodes, steps, rewards = load_obj("Q_learning_results")
    # plot(rewards)
    rewards = load_obj("A2C")
    plot(rewards)
