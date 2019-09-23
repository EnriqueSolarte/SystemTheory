import tensorflow as tf
import numpy as np
import gym
from plotting import *

env = gym.make('CartPole-v1')
env = env.unwrapped
env.seed(1)

## ENVIRONMENT Hyperparameters
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

## TRAINING Hyperparameters
max_episodes = 2000
learning_rate = 0.01
gamma = 0.95


def discount_and_normalize_rewards(episode_rewards):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative

    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

    return discounted_episode_rewards


with tf.name_scope("inputs"):
    input_ = tf.placeholder(tf.float32, [None, state_size], name="input_")
    actions = tf.placeholder(tf.int32, [None, action_size], name="actions")
    discounted_episode_rewards_ = tf.placeholder(tf.float32, [None, ], name="discounted_episode_rewards")

    with tf.name_scope("fc1"):
        fc1 = tf.contrib.layers.fully_connected(inputs=input_,
                                                num_outputs=5,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())

    with tf.name_scope("fc2"):
        fc2 = tf.contrib.layers.fully_connected(inputs=fc1,
                                                num_outputs=5,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())

    with tf.name_scope("fc3"):
        fc3 = tf.contrib.layers.fully_connected(inputs=fc2,
                                                num_outputs=action_size,
                                                activation_fn=None,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())

    with tf.name_scope("softmax"):
        action_distribution = tf.nn.softmax(fc3)

    with tf.name_scope("loss"):
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc3, labels=actions)
        loss = tf.reduce_mean(neg_log_prob * discounted_episode_rewards_)

    with tf.name_scope("train"):
        train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

allRewards = []
total_rewards = 0
maximumRewardRecorded = 0
episode = 0
episode_states, episode_actions, episode_rewards = [], [], []
mean_reward = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saved_rewards = []
    for episode in range(max_episodes):

        episode_rewards_sum = 0

        # Launch the game
        state = env.reset()

        while True:

            # Choose action a, remember WE'RE NOT IN A DETERMINISTIC ENVIRONMENT, WE'RE OUTPUT PROBABILITIES.
            action_probability_distribution = sess.run(action_distribution, feed_dict={input_: state.reshape([1, 4])})

            action = np.random.choice(range(action_probability_distribution.shape[1]), p=action_probability_distribution.ravel())  # select action w.r.t the actions prob

            # Perform a
            new_state, reward, done, info = env.step(action)

            # Store s, a, r
            episode_states.append(state)

            # For actions because we output only one (the index) we need 2 (1 is for the action taken)
            # We need [0., 1.] (if we take right) not just the index
            action_ = np.zeros(action_size)
            action_[action] = 1

            episode_actions.append(action_)

            episode_rewards.append(reward)

            # if mean_reward > 200:
            #     env.render()
            saved_rewards.append(reward)
            save_obj("Policy_gradient", saved_rewards, verbose=False)
            if done:
                # Calculate sum reward
                episode_rewards_sum = np.sum(episode_rewards)

                allRewards.append(episode_rewards_sum)

                total_rewards = np.sum(allRewards)

                # Mean reward
                mean_reward = np.divide(total_rewards, episode + 1)

                maximumRewardRecorded = np.amax(allRewards)

                print("==========================================")
                print("Episode: ", episode)
                print("Reward: ", episode_rewards_sum)
                print("Mean Reward", mean_reward)
                print("Max reward so far: ", maximumRewardRecorded)

                # Calculate discounted reward
                discounted_episode_rewards = discount_and_normalize_rewards(episode_rewards)

                # Feedforward, gradient and backpropagation
                loss_, _ = sess.run([loss, train_opt], feed_dict={input_: np.vstack(np.array(episode_states)),
                                                                  actions: np.vstack(np.array(episode_actions)),
                                                                  discounted_episode_rewards_: discounted_episode_rewards
                                                                  })

                # Reset the transition stores
                episode_states, episode_actions, episode_rewards = [], [], []

                break

            state = new_state
