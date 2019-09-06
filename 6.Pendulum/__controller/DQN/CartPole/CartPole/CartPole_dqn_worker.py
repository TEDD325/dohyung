import threading
from collections import deque
from logger import get_logger
import gym
import numpy as np
import random
import os

LAST_EPISODE_N_PATH = "./save_info/last_episode.txt"
MEMORY_LENGTH = 2000
BATCH_SIZE = 256

class Worker(threading.Thread):
    def __init__(self, idx, global_dqn, model, target_model, discount_factor, sess, series_size, feature_size, action_size, max_episodes, model_type):
        threading.Thread.__init__(self)

        self.state_list = []
        self.action_list = []
        self.reward_list = []
        self.new_state_list = []
        self.done_list = []

        # replay memory, maximum size is 2000
        self.memory = deque(maxlen=MEMORY_LENGTH)

        self.sess = sess

        self.global_dqn = global_dqn

        self.idx = idx
        self.model = model
        self.target_model = target_model
        self.discount_factor = discount_factor
        self.local_score_list = []
        self.local_logger = get_logger("./cartpole_a3c_" + str(self.idx))
        self.env = gym.make('CartPole-v0')

        self.series_size = series_size
        self.feature_size = feature_size
        self.action_size = action_size
        self.max_episodes = max_episodes
        self.model_type = model_type

        self.state_series = deque([], self.series_size)
        self.new_state_series = deque([], self.series_size)
        for _ in range(series_size):
            self.state_series.append(np.zeros(shape=(self.feature_size,)).tolist())
            self.new_state_series.append(np.zeros(shape=(self.feature_size,)).tolist())

        self.running = True

    # Thread interactive with environment
    def run(self):
        local_episode = 0

        if os.path.exists(LAST_EPISODE_N_PATH):
            f = open(LAST_EPISODE_N_PATH, 'r')
            local_episode = int(f.readline())
            f.close()

        while local_episode < self.max_episodes and self.running:
            state = self.env.reset()
            self.state_series.append(state.tolist())

            local_score = 0
            local_step = 0

            while self.running:
                action = self.get_action(self.state_series)
                new_state, reward, done, info = self.env.step(action)

                # self.local_logger.info("{0} - policy: {1}|{2}, Action: {3} --> State: {3}, Reward: {4}, Done: {5}, Info: {6}".format(
                #     self.idx, policy, argmax, action, new_state, reward, done, info
                # ))
                # print("{0} - policy: {1}|{2}, Action: {3} --> State: {3}, Reward: {4}, Done: {5}, Info: {6}".format(
                #     self.idx, policy, argmax, action, new_state, reward, done, info
                # ))

                local_score += reward
                local_step += 1

                self.append_sample(state, action, reward, new_state, done)

                state = new_state

                if local_step % 5 == 0 and self.running:
                    loss = self.train_episode()
                    self.global_dqn.save_model()
                    self.remove_memory()

                if done and self.running:
                    if len(self.state_list) > 0:
                        loss = self.train_episode()
                        self.global_dqn.save_model()
                        self.remove_memory()

                    local_episode += 1

                    self.local_score_list.append(local_score)
                    mean_local_score = np.mean(self.local_score_list)

                    self.local_logger.info("{0:>5}-Episode {1:>3d}: SCORE {2:.6f}, MEAN SCORE {3:.6f}".format(
                        self.idx,
                        local_episode,
                        local_score,
                        mean_local_score
                    ))
                    print("{0:>5}-Episode {1:>3d}: SCORE {2:.6f}, MEAN SCORE {3:.6f}".format(
                        self.idx,
                        local_episode,
                        local_score,
                        mean_local_score
                    ))

                    self.global_dqn.append_global_score_list(self.idx, local_episode, local_score, loss)
                    break

    # save sample <s, a, r, s'>into replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def remove_memory(self):
        self.memory.clear()

    # update policy network and value network every episode
    def train_episode(self):
        mini_batch = random.sample(self.memory, BATCH_SIZE)

        states = np.zeros(
            (BATCH_SIZE, self.feature_size, self.series_size, 1)
        )

        next_states = np.zeros(
            (BATCH_SIZE, self.feature_size, self.series_size, 1)
        )

        actions, rewards, dones = [], [], []

        for i in range(BATCH_SIZE):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        # model Q-function about current state
        # Q-function of target model about next state
        target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)

        # update target using bellman optimal equation
        for i in range(BATCH_SIZE):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (np.amax(target_val[i]))

        hist = self.model.fit(states, target, batch_size=BATCH_SIZE, epochs=1, verbose=0)
        self.loss = np.mean(hist.history['loss'])

        return self.loss

    # select action based on epsilon greedy policy
    def get_action(self, state):
        # if np.random.rand() <= self.epsilon:
        #     return random.randrange(self.action_size)
        # else:
        #     q_value = self.model.predict(state)
        state = np.array(state)
        q_value = self.model.predict(state)
        return np.argmax(q_value[0])

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
