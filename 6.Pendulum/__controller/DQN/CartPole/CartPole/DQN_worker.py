import threading
from collections import deque
from logger import get_logger
import numpy as np
import os
import sys
import time
import random
import itertools


STEP_THRESHOLD = 5 # minimum 15
MEMORY_LENGTH = 2000
ENV_COUNT = 0
LAST_EPISODE_N_PATH = "./save_info/last_episode.txt"
BATCH_SIZE = 200
TRAINING_ITER = 100

class Worker(threading.Thread):
    def __init__(self, idx, global_a3c, actor, critic, optimizer, discount_factor, sess, series_size, feature_size,
                 action_size, max_episodes, model_type):
        threading.Thread.__init__(self)
        global ENV_COUNT
        # ENV_COUNT = THREADS_NUM
        # self.env_1 = env[0]
        # self.env_2 = env[1]

        self.thread_id = idx

        # self.state_list = []
        # self.action_list = []
        # self.reward_list = []
        # self.new_state_list = []
        # self.done_list = []
        # self.discount_reward = []
        self.state_list = deque(maxlen=MEMORY_LENGTH)
        self.action_list = deque(maxlen=MEMORY_LENGTH)
        self.reward_list = deque(maxlen=MEMORY_LENGTH)
        self.new_state_list = deque(maxlen=MEMORY_LENGTH)
        self.done_list = deque(maxlen=MEMORY_LENGTH)
        self.discount_reward_list = deque(maxlen=MEMORY_LENGTH)

        self.sess = sess

        self.global_a3c = global_a3c

        self.idx = idx
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        self.local_score_list = []
        self.local_logger = get_logger("./" + str(self.idx))
        global ENV_COUNT
        # env = ENV_1()
        env = None
        if ENV_COUNT == 0:
            # env = self.env_1
            env = ENV_1()
            ENV_COUNT += 1
        elif ENV_COUNT == 1:
            # env = self.env_2
            env = ENV_2()
            ENV_COUNT += 1
        elif ENV_COUNT > 1:
            env = None

        self.env = env
        print("env>", self.env)

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
            state, _, _, _, _ = self.env.reset()
            # print("state> ", state[0])
            # print(type(state[0]))
            state = [state[0] * 100, state[1], state[2] * 100, state[3]]
            state = np.array(state)
            # print(state.shape)
            self.state_series.append(state.tolist())
            # print("state_sereis> ", self.state_series)
            # print(type(self.state_series))

            local_score = 0
            local_step = 0

            # previous_time = time.perf_counter()
            while self.running:
                previous_time = time.perf_counter()

                # current_time = time.perf_counter()
                # print(self.thread_id, " - ", current_time - previous_time)
                # previous_time = current_time

                policy, argmax, action = self.get_action(self.state_series)
                new_state, reward, done, info = self.env.step(action)
                new_state = [new_state[0] * 100, new_state[1], new_state[2] * 100, new_state[3]]
                self.local_logger.info(
                    "{0} - policy: {1}|{2}, Action: {3} --> State: {3}, Reward: {4}, Done: {5}, Info: {6}".format(
                        self.idx, policy, argmax, action, new_state, reward, done, info
                    ))

                local_score += reward
                local_step += 1
                success_cnt = 0

                self.append_memory(state, action, reward, new_state, done)

                state = new_state

                if not done:
                    while True:
                        current_time = time.perf_counter()
                        if current_time - previous_time >= 6 / 1000:
                            # print("TIME EXCEEDS 6 ms.")
                            break

                # if serial successes are persisted, system will be shutted down.
                if local_step >= 5000:
                    success_cnt += 1
                    done = True
                else:
                    success_cnt = 0

                if success_cnt >= 5:
                    self.env.close()

                if done and self.running:
                    self.env.wait()

                    if local_step > STEP_THRESHOLD:
                        print("id:{0}, episode:{1}  score:{2}  step:{3}  info:{4}".format(
                            self.thread_id, local_episode, local_score, local_step, info)
                            )

                        sys.stdout.flush()

                        local_episode += 1

                        actor_loss, critic_loss = self.train_episode()
                        # print("actor_loss:{0}, critic_loss:{1}".format(actor_loss, critic_loss))

                        self.global_a3c.save_model()
                        # self.remove_memory()

                        self.local_score_list.append(local_score)
                        mean_local_score = np.mean(self.local_score_list)

                        self.local_logger.info("{0:>5}-Episode {1:>3d}: SCORE {2:.6f}, MEAN SCORE {3:.6f}".format(
                            self.idx,
                            local_episode,
                            local_score,
                            mean_local_score
                        ))

                        self.global_a3c.append_global_score_list(self.idx, local_episode, local_score, actor_loss,
                                                                 critic_loss)

                        f = open(LAST_EPISODE_N_PATH, 'w')
                        f.write(str(local_episode) + "\n")
                        f.close()

                    break

    def append_memory(self, state, action, reward, new_state, done):
        self.state_series.append(state)
        self.state_list.append(self.state_series.copy())

        act = np.zeros(self.action_size)
        act[action] = 1
        self.action_list.append(act)

        self.reward_list.append(reward)

        self.new_state_series.append(new_state)
        self.new_state_list.append(self.new_state_series.copy())

        self.done_list.append(done)

    def remove_memory(self):
        self.state_list.clear()
        self.action_list.clear()
        self.reward_list.clear()
        self.new_state_list.clear()
        self.done_list.clear()
        self.discount_reward_list.clear()

    # update policy network and value network every episode
    def train_episode(self):
        discount_rewards = []
        if self.model_type == "MLP":
            # v_r = self.critic.predict(
            #     np.reshape(self.new_state_list, (len(self.new_state_list), self.series_size * self.feature_size))
            # )[0]
            if self.done_list[-1]:
                reward = 0
            else:
                reward = self.critic.predict(
                    np.reshape(self.new_state_list, (len(self.new_state_list), self.series_size * self.feature_size))
                )[0]
        elif self.model_type == 'LSTM':
            if self.done_list[-1]:
                reward = 0
            else:
                reward = self.critic.predict(
                    np.reshape(self.new_state_list, (len(self.new_state_list), self.series_size, self.feature_size))
                )[0]
        else:
            if self.done_list[-1]:
                reward = 0
            else:
                reward = self.critic.predict(
                    np.reshape(self.new_state_list, (len(self.new_state_list), self.series_size, self.feature_size, 1))
                )[0]

        for t in reversed(range(0, len(self.reward_list))):
            reward = self.reward_list[t] + self.discount_factor * reward
            discount_rewards.append(reward)

        discount_rewards.reverse()
        for discount_reward in discount_rewards:
            self.discount_reward_list.append(discount_reward)

        mean_actor_loss = []
        mean_critic_loss = []


        if len(self.state_list) < BATCH_SIZE:
            if self.model_type == "MLP":
                local_input = np.reshape(self.state_list, (len(self.state_list), self.series_size * self.feature_size))
                value = self.critic.predict(local_input)[0]
            elif self.model_type == 'LSTM':
                local_input = np.reshape(self.state_list, (len(self.state_list), self.series_size, self.feature_size))
                value = self.critic.predict(local_input)[0]
            else:
                local_input = np.reshape(self.state_list, (len(self.state_list), self.series_size, self.feature_size, 1))
                value = self.critic.predict(local_input)[0]

            advantage = (np.array(discount_rewards) - value).tolist()

            actor_loss, policy, weights = self.optimizer[0]([
                local_input,
                np.reshape(self.action_list, (len(self.action_list), self.action_size)),
                advantage
            ])

            critic_loss, value, weights_0, weights_1, weights_2, weights_3 = self.optimizer[1]([
                local_input,
                discount_rewards
            ])

            mean_actor_loss = np.mean(actor_loss)
            mean_critic_loss = np.mean(critic_loss)
            return mean_actor_loss, mean_critic_loss
        else:
            for _ in range(TRAINING_ITER):
                actor_loss_list = []
                critic_loss_list = []
                init_batch_index = random.randrange(0, len(self.state_list)-BATCH_SIZE)
                state_list = list(itertools.islice(self.state_list, init_batch_index, init_batch_index + BATCH_SIZE))
                action_list = list(itertools.islice(self.action_list, init_batch_index, init_batch_index + BATCH_SIZE))
                discount_reward_list = list(itertools.islice(self.discount_reward_list, init_batch_index, init_batch_index + BATCH_SIZE))

                if self.model_type == "MLP":
                    local_input = np.reshape(state_list, (len(state_list), self.series_size * self.feature_size))
                    value = self.critic.predict(local_input)[0]
                elif self.model_type == 'LSTM':
                    local_input = np.reshape(state_list, (len(state_list), self.series_size, self.feature_size))
                    value = self.critic.predict(local_input)[0]

                else:
                    local_input = np.reshape(state_list,
                                             (len(state_list), self.series_size, self.feature_size, 1))
                    value = self.critic.predict(local_input)[0]

                advantage = (np.array(discount_reward_list) - value).tolist()

                actor_loss_list, policy, weights = self.optimizer[0]([
                    local_input,
                    np.reshape(action_list, (len(action_list), self.action_size)),
                    advantage
                ])

                critic_loss_list, value, weights_0, weights_1, weights_2, weights_3 = self.optimizer[1]([
                    local_input,
                    discount_reward_list
                ])

                mean_actor_loss = np.mean(actor_loss_list)
                mean_critic_loss = np.mean(critic_loss_list)
            return mean_actor_loss, mean_critic_loss


    def get_action(self, state):
        state = np.asarray(state)
        if self.model_type == "MLP":
            # print("state> ", state)
            # print(type(state))
            # state = np.array(state[0])
            state = np.reshape(state, (1, self.series_size * self.feature_size))
        elif self.model_type == "LSTM":
            state = np.reshape(state, (1, self.series_size, self.feature_size))
        else:
            state = np.reshape(state, (1, self.series_size, self.feature_size, 1))

        logits = self.actor.predict(state)[0]
        policy = self.softmax(logits)
        action = np.random.choice(self.action_size, 1, p=policy)[0]

        return policy, np.argmax(policy), action

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)