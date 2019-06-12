import numpy as np
import argparse
import matplotlib.pyplot as plt
import gym


class CartPole:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.env.seed(1)

        self.discrete_cart_pos = None
        self.discrete_cart_vel = None
        self.discrete_pole_pos = None
        self.discrete_pole_vel = None
        self.setting_discrete_observation()
        self.actions = range(self.env.action_space.n)
        self.q = {}

    def setting_discrete_observation(self, bins_pos=100, bins_vel=10000):
        '''
        Precomputed state observations
        '''
        self.discrete_cart_pos = np.linspace(-4.8, 4.8, bins_pos)
        self.discrete_cart_vel = np.linspace(-1000, -1000, bins_vel)
        self.discrete_pole_pos = np.linspace(-10, 10, 20)
        self.discrete_pole_vel = np.linspace(-1000, 1000, bins_vel)

    def build_state(self, observation):
        """
        Given an observation a state-code is built
        :param observation: observation [cart_pos, cart_vel, pole_ang, pole_vel]
        :return: state
        """
        cart_pos, cart_vel, pole_pos, pole_vel = observation

        cart_pos = self.__discretize(cart_pos, self.discrete_cart_pos)
        cart_vel = self.__discretize(cart_vel, self.discrete_cart_vel)
        pole_pos = self.__discretize(pole_pos, self.discrete_pole_pos)
        pole_vel = self.__discretize(pole_vel, self.discrete_pole_vel)

        features = [cart_pos, pole_pos, cart_vel, pole_vel]
        return int("".join(map(lambda feature: str(int(feature)), features)))

    def __discretize(self, value, discrete_array):
        return np.digitize(x=[value], bins=discrete_array)[0]

    def choose_action(self, state, epsilon):
        q = [self.evaluate_state(state, a) for a in self.actions]
        maxQ = max(q)

        if np.random.random() < epsilon:
            minQ = min(q);
            mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [q[i] + np.random.random() * mag - .5 * mag for i in range(len(self.actions))]
            maxQ = max(q)
            #
            # i = self.env.action_space.sample()
            # return self.actions[i]

        count = q.count(maxQ)

        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = np.random.choice(best)
        else:
            i = q.index(maxQ)

        return self.actions[i]

    def evaluate_state(self, state, action):
        '''
        Return the reward associated to both state and action
        :param state: state-code from the env
        :param action: env action
        :return: reward
        '''

        return self.q.get((state, action), 0.0)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()


class QLearning:
    def __init__(self, actions, epsilon, alpha, gamma, env):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = range(actions)
        self.env = env

        self.steps = []
        self.rewards = []
        self.episodes = []

    def run(self, args):
        for i_episode in range(args.episodes):
            current_observation = self.env.reset()
            state = self.env.build_state(current_observation)
            self.episodes.append(i_episode)
            self.rewards.append(0)
            for t in range(args.steps):
                # take an action given the current state following a pi policy
                action = self.env.choose_action(state, epsilon=self.epsilon)
                # if self.rewards[-1] > 100:
                #     self.env.env.render()
                # evaluating action in the environment
                next_obs, reward, done, info = self.env.step(action)
                next_state = self.env.build_state(next_obs)
                self.rewards[i_episode] += reward
                if done:
                    print(self.rewards[-1], i_episode)
                    reward = -100
                    self.update(state, action, reward, next_state)
                    self.steps.append(t)
                    break
                else:
                    self.update(state, action, reward, next_state)

                state = next_state

        return self.episodes, self.steps, self.rewards

    def update(self, state, action, reward, next_state):
        Q = self.env.q.get((state, action), None)
        if Q is None:
            self.env.q[(state, action)] = reward
        else:
            Q_max = max([self.env.evaluate_state(next_state, a) for a in self.actions])
            target = reward + self.gamma * Q_max
            error = target - Q
            self.env.q[(state, action)] = Q + self.alpha * error


if __name__ == '__main__':
    # define argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-episodes", "--episodes", default=2000,
                        help="Training episodes")
    parser.add_argument("-steps", "--steps", default=1000,
                        help="Max. number of steps")
    parser.add_argument("-render", "--render", action='store_true', default=False,
                        help="visualize the result of an algorithm")

    args = parser.parse_args()

    cPole = CartPole()
    algorithm = QLearning(actions=cPole.env.action_space.n,
                          alpha=0.1,
                          gamma=0.9,
                          epsilon=0.1,
                          env=cPole)
    episodes, steps, rewards = algorithm.run(args=args)
    plt.plot(episodes, rewards)
    plt.show()
    print("end")
