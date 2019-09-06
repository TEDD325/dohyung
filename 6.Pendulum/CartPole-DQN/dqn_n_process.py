# -*- coding: utf-8 -*-
# https://github.com/openai/gym/wiki/CartPole-v0
# https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947

import math
import threading
import time
import zlib

import tensorflow as tf
from logger import get_logger
import sys
import json

from tensorflow.python.keras.layers import Dropout
#from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers import LeakyReLU

print(tf.__version__)
tf.config.gpu.set_per_process_memory_fraction(0.4)

from tensorflow.python.keras.layers import Dense, Input

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from collections import deque
import numpy as np
import random
import gym
from multiprocessing import Process
import zmq
import matplotlib.pyplot as plt
import os
import pickle
import socket

from numpy.random import seed
# seed(1)
# tf.random.set_seed(2)

import warnings
warnings.filterwarnings("ignore")

ddqn = True
num_hidden_layers = 3
num_weight_transfer_hidden_layers = 4
num_workers = 4
score_based_transfer = False
loss_based_transfer = False
soft_transfer = True
soft_transfer_fraction = 0.3
verbose = False



def exp_moving_average(values, window):
    """ Numpy implementation of EMA
    """
    if window >= len(values):
        sma = np.mean(np.asarray(values))
        a = [sma] * len(values)
    else:
        weights = np.exp(np.linspace(-1., 0., window))
        weights /= weights.sum()
        a = np.convolve(values, weights, mode='full')[:len(values)]
        a[:window] = a[window]
    return a


class DQNAgent:
    def __init__(self, worker_idx, env_id, win_reward, loss_trials, max_episodes):
        self.worker_idx = worker_idx
        self.env = gym.make(env_id)
        self.env.seed(0)

        self.action_space = self.env.action_space

        # experience buffer
        self.memory = deque(maxlen=2000)

        # discount rate
        self.gamma = 0.9

        # initially 90% exploration, 10% exploitation
        self.epsilon = 0.5

        # iteratively applying decay til 10% exploration/90% exploitation
        # self.epsilon_min = 0.0001
        self.epsilon_min = 0.001

        # coordinate the speed of epsilon decaying
        self.epsilon_coor = 0.3

        # soft update target network
        self.tau = soft_transfer_fraction

        # learning rate
        self.learning_rate = 0.001

        self.win_reward = win_reward

        self.loss_trials = loss_trials

        # Q Network weights filename
        self.weights_file = './dqn_cartpole.h5'

        # Q Network for training
        self.n_inputs = int(self.env.observation_space.shape[0] / 2)
        self.n_outputs = self.action_space.n

        self.q_model = self.build_model(self.n_inputs, self.n_outputs)
        self.q_model.compile(
            loss='mse',
            optimizer=Adam(lr=self.learning_rate)
        )

        # target Q Network
        self.target_q_model = self.build_model(self.n_inputs, self.n_outputs)

        # copy Q Network params to target Q Network
        self.update_target_model_weights()

        self.replay_counter = 0

        print("----------Worker {0}: {1}:--------".format(
            self.worker_idx,
            "Double DQN" if ddqn else "DQN",
        ))

        if score_based_transfer:
            print("SCORE-based TRANSFER!!!")
        elif loss_based_transfer:
            print("LOSS-based TRANSFER!!!")
        else:
            print("NO TRANSFER")

        self.logger = get_logger("cartpole_worker_" + str(self.worker_idx))

        self.max_episodes = max_episodes

        self.global_max_ema_score = 0
        self.global_min_ema_loss = 1000000000

        self.local_scores = []
        self.local_losses = []

        self.score_dequeue = deque(maxlen=100)
        self.loss_dequeue = deque(maxlen=100)

    # Q Network is 256-256-256-2 MLP
    def build_model(self, n_inputs, n_outputs):
        inputs = Input(shape=(n_inputs,), name='state_' + str(self.worker_idx))
        x = Dense(
            units=128,
            kernel_initializer='he_normal',
            bias_initializer='zero',
            name="layer_0_" + str(self.worker_idx)
        )(inputs)

        x = LeakyReLU(alpha=0.05)(x)

        x = Dense(
            units=128,
            kernel_initializer='he_normal',
            bias_initializer='zero',
            name="layer_1_" + str(self.worker_idx)
        )(x)

        x = LeakyReLU(alpha=0.05)(x)

        x = Dense(
            units=128,
            kernel_initializer='he_normal',
            bias_initializer='zero',
            name="layer_2_" + str(self.worker_idx)
        )(x)

        x = LeakyReLU(alpha=0.05)(x)

        x = Dense(
            units=n_outputs,
            kernel_initializer='he_normal',
            bias_initializer='zero',
            activation='linear',
            name='layer_3_' + str(self.worker_idx)
        )(x)

        model = Model(inputs, x)
        model.summary()
        return model

    # save Q Network params to a file
    def save_weights(self):
        self.q_model.save_weights(self.weights_file)

    # copy trained Q Network params to target Q Network
    def update_target_model_weights(self, soft=False):
        if soft:
            pars_behavior = self.q_model.get_weights()  # these have form [W1, b1, W2, b2, ..], Wi = weights of layer i
            pars_target = self.target_q_model.get_weights()  # bi = biases in layer i

            ctr = 0
            for par_behavior, par_target in zip(pars_behavior, pars_target):
                par_target = par_target * (1 - self.tau) + par_behavior * self.tau
                pars_target[ctr] = par_target
                ctr += 1

            self.target_q_model.set_weights(pars_target)
        else:
            self.target_q_model.set_weights(
                self.q_model.get_weights()
            )

    # eps-greedy policy
    def act(self, state):
        if np.random.rand() < self.epsilon:
            # explore - do random action
            return self.action_space.sample(), True
        else:
            # exploit
            q_values = self.q_model.predict(state)

            # select the action with max Q-value
            return np.argmax(q_values[0]), False

    # store experiences in the replay buffer
    def remember(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        self.memory.append(item)

    # compute Q_max
    # use of target Q Network solves the non-stationarity problem
    def get_target_q_value(self, next_state, reward):
        # max Q value among next state's actions
        if ddqn:
            # DDQN
            # current Q Network selects the action
            # a'_max = argmax_a' Q(s', a')
            action = np.argmax(self.q_model.predict(next_state)[0])

            # target Q Network evaluates the action
            # Q_max = Q_target(s', a'_max)
            q_max = self.target_q_model.predict(next_state)[0][action]
        else:
            # DQN chooses the max Q value among next actions
            # selection and evaluation of action is on the target Q Network
            # Q_max = max_a' Q_target(s', a')
            q_max = np.amax(self.target_q_model.predict(next_state)[0])

        # Q_value = reward + gamma * Q_max
        q_value = reward + self.gamma * q_max
        return q_value

    # experience replay addresses the correlation issue between samples
    def replay(self, batch_size, episode):
        # sars = state, action, reward, state' (next_state)
        sars_batch = random.sample(self.memory, batch_size)
        state_batch, q_values_batch = [], []

        # but easier to understand using a loop
        for state, action, reward, next_state, done in sars_batch:
            # policy prediction for a given state
            q_values = self.q_model.predict(state)

            # get Q_max
            q_value = self.get_target_q_value(next_state, reward)

            # correction on the Q value for the action used
            q_values[0][action] = reward if done else q_value

            # collect batch state-q_value mapping
            state_batch.append(state[0])
            q_values_batch.append(q_values[0])

        # train the Q-network
        self.q_model.fit(
            x=np.array(state_batch),
            y=np.array(q_values_batch),
            batch_size=batch_size,
            epochs=1,
            verbose=0
        )

        loss = self.q_model.evaluate(
            x=np.array(state_batch),
            y=np.array(q_values_batch),
            verbose=0
        )

        # update exploration-exploitation probability
        self.update_epsilon(episode)

        # copy new params on old target after every 10 training updates
        if self.replay_counter % 10 == 0:
            self.update_target_model_weights()

        self.replay_counter += 1

        return loss

    #decrease the exploration, increase exploitation
    def update_epsilon(self, episode):
        if episode != 0:
            #epsilon_decay = (self.epsilon_min / self.epsilon) ** (0.2 / float(episode))
            epsilon_decay = 0.99

            if self.epsilon > self.epsilon_min:
                self.epsilon *= epsilon_decay

    # def update_epsilon(self, episode):
    #     if episode != 0:
    #         epsilon_decay = (self.epsilon_coor ** (episode / 50) / self.epsilon) ** (1. / float(episode))
    #
    #         if self.epsilon > self.epsilon_min:
    #             self.epsilon *= epsilon_decay
    #
    #     if episode > 150:
    #         self.epsilon = 0.0001

    def start_rl(self, socket):
        # should be solved in this number of episodes
        state_size = int(self.env.observation_space.shape[0] / 2)
        batch_size = 64

        # by default, CartPole-v0 has max episode steps = 200
        # you can use this to experiment beyond 200
        # env._max_episode_steps = 4000

        # Q-Learning sampling and fitting
        for episode in range(self.max_episodes):
            state = self.env.reset()[2:4]
            state = np.reshape(state, [1, state_size])
            done = False
            total_reward = 0

            has_random_action = False
            while not done:
                # in CartPole-v0, action=0 is left and action=1 is right
                action, is_random = self.act(state)

                if is_random:
                    has_random_action = True

                next_state, reward, done, _ = self.env.step(action)

                next_state = next_state[2:4]

                # in CartPole-v0:
                # state = [pos, vel, theta, angular speed]
                next_state = np.reshape(next_state, [1, state_size])

                # store every experience unit in replay buffer
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            if len(self.memory) < batch_size:
                msg = "Worker {0}-Ep.{1:>2d}: Memory Length: {2}, Has_Rand_Act:{3}".format(
                    self.worker_idx,
                    episode,
                    len(self.memory),
                    has_random_action
                )
                self.logger.info(msg)
                if verbose: print(msg)
            else:
                # call experience relay
                loss = self.replay(batch_size, episode)
                score = total_reward

                self.local_losses.append(loss)
                self.local_scores.append(score)

                self.score_dequeue.append(score)
                self.loss_dequeue.append(loss)

                ema_loss = exp_moving_average(self.local_losses, 10)[-1]
                ema_score = exp_moving_average(self.local_scores, 10)[-1]

                mean_score_over_recent_100_episodes = np.mean(self.score_dequeue)
                mean_loss_over_recent_100_episodes = np.mean(self.loss_dequeue)

                send_weights = False
                # Worker에 의하여 더 높은 Score를 찾음
                if episode > 10 and score_based_transfer and self.global_max_ema_score < ema_score:
                    self.global_max_ema_score = ema_score
                    send_weights = True
                    self.update_target_model_weights()

                    msg = ">>> Worker {0}: Find New Global Max EMA Score: {1:>4.2f}".format(
                        self.worker_idx,
                        self.global_max_ema_score
                    )
                    self.logger.info(msg)
                    if verbose: print(msg)

                # Worker에 의하여 더 낮은 Loss를 찾음
                if episode > 10 and loss_based_transfer and ema_loss < self.global_min_ema_loss:
                    self.global_min_ema_loss = ema_loss
                    send_weights = True
                    self.update_target_model_weights()

                    msg = ">>> Worker {0}: Find New Global Min EMA Loss: {1:>6.4f}".format(
                        self.worker_idx,
                        self.global_min_ema_loss
                    )
                    self.logger.info(msg)
                    if verbose: print(msg)

                continue_loop = self.send_episode_info_and_adapt_best_weights(
                    socket,
                    episode,
                    loss,
                    ema_loss,
                    score,
                    ema_score,
                    send_weights=send_weights
                )

                if not continue_loop:
                    time.sleep(1)
                    break

                msg = "Worker {0}-Ep.{1:>2d}: Loss={2:6.4f} (EMA: {3:6.4f}, Mean: {4:6.4f}), Score={5:5.1f} (EMA: {" \
                      "6:>4.2f}, Mean: {7:>4.2f}), Epsilon: {8:>6.4f}, Has_Rand_Act:{9}".format(
                    self.worker_idx,
                    episode,
                    loss,
                    ema_loss,
                    mean_loss_over_recent_100_episodes,
                    score,
                    ema_score,
                    mean_score_over_recent_100_episodes,
                    self.epsilon,
                    has_random_action
                )

                self.logger.info(msg)
                if verbose: print(msg)

                if mean_score_over_recent_100_episodes >= self.win_reward:
                    msg = "******* Worker {0} - Solved in episode {1}: Mean score = {2} - Epsilon: {3}".format(
                        self.worker_idx,
                        episode,
                        mean_score_over_recent_100_episodes,
                        self.epsilon
                    )
                    self.logger.info(msg)
                    print(msg)

                    self.save_weights()

                    self.send_solve_info(
                        socket,
                        last_episode=episode
                    )

                    break

        # close the env and write monitor result info to disk
        self.env.close()

    def send_episode_info_and_adapt_best_weights(self, socket, episode, loss, ema_loss, score, ema_score, send_weights):
        if score_based_transfer or loss_based_transfer:
            if send_weights:
                weights = {}
                for layer_id in range(num_weight_transfer_hidden_layers):
                    layer_name = "layer_{0}_{1}".format(
                        layer_id,
                        worker_idx
                    )
                    weights[layer_id] = self.q_model.get_layer(name=layer_name).get_weights()
            else:
                weights = {}

            episode_msg = {
                "type": "episode",
                "episode": episode,
                "worker_idx": self.worker_idx,
                "loss": loss,
                "ema_loss": ema_loss,
                "score": score,
                "ema_score": ema_score,
                "weights": weights
            }
        else:
            episode_msg = {
                "type": "episode",
                "worker_idx": self.worker_idx,
                "loss": loss,
                "score": score
            }

        episode_msg = pickle.dumps(episode_msg, protocol=-1)
        episode_msg = zlib.compress(episode_msg)
        socket.send(episode_msg)

        episode_ack_msg = socket.recv()
        episode_ack_msg = zlib.decompress(episode_ack_msg)
        episode_ack_msg = pickle.loads(episode_ack_msg)

        continue_loop = True
        if episode_ack_msg["type"] == "episode_ack":
            if score_based_transfer:
                global_max_ema_score = episode_ack_msg["global_max_ema_score"]
                best_weights = episode_ack_msg["best_weights"]

                if best_weights is not None and len(best_weights) > 0 and not send_weights:
                    # Worker 스스로는 Best Score/Weights 를 못 찾았지만 서버로 부터 Best Score/weights 를 받은 경우
                    self.global_max_ema_score = global_max_ema_score

                    for layer_id in range(num_weight_transfer_hidden_layers):
                        layer_name = "layer_{0}_{1}".format(
                            layer_id,
                            worker_idx
                        )
                        self.q_model.get_layer(name=layer_name).set_weights(best_weights[layer_id])

                    self.update_target_model_weights()

                    msg = ">>> Worker {0}: Receive New Best Weights from Worker {1} and Set Them to Local Model!!! - " \
                          "global_max_ema_score: {2}".format(
                        self.worker_idx,
                        episode_ack_msg["best_found_worker"],
                        self.global_max_ema_score
                    )
                    self.logger.info(msg)
                    if verbose: print(msg)

            if loss_based_transfer:
                global_min_ema_loss = episode_ack_msg["global_min_ema_loss"]
                best_weights = episode_ack_msg["best_weights"]

                if best_weights is not None and len(best_weights) > 0 and not send_weights:
                    # Worker 스스로는 Best Score/weights 를 못 찾았지만 서버로 부터 Best Score/weights 를 받은 경우
                    self.global_min_ema_loss = global_min_ema_loss

                    for layer_id in range(num_weight_transfer_hidden_layers):
                        layer_name = "layer_{0}_{1}".format(
                            layer_id,
                            worker_idx
                        )
                        self.q_model.get_layer(name=layer_name).set_weights(best_weights[layer_id])

                    self.update_target_model_weights()

                    msg = ">>> Worker {0}: Receive New Best Weights from Worker {1} and Set Them to Local Model!!! - " \
                          "global_min_ema_loss: {2:6.4f}".format(
                        self.worker_idx,
                        episode_ack_msg["best_found_worker"],
                        self.global_min_ema_loss
                    )
                    self.logger.info(msg)
                    if verbose: print(msg)
            else:
                pass
        elif episode_ack_msg["type"] == "solved_ack":
            if score_based_transfer or loss_based_transfer:
                msg = "Solved by Other Worker"
                self.logger.info(msg)
                if verbose: print(msg)
                continue_loop = False
        else:
            pass

        return continue_loop

    def send_solve_info(self, socket, last_episode):
        solve_msg = {
            "type": "solved",
            "worker_idx": self.worker_idx,
            "last_episode": last_episode
        }
        solve_msg = pickle.dumps(solve_msg, protocol=-1)
        solve_msg = zlib.compress(solve_msg)
        socket.send(solve_msg)

    def transfer_update(self, best_weights):
        for layer_id in range(num_weight_transfer_hidden_layers):
            layer_name = "layer_{0}_{1}".format(
                layer_id,
                worker_idx
            )
            if soft_transfer:
                par_original = self.q_model.get_layer(name=layer_name).get_weights()
                par_updated = par_original * self.tau + best_weights[layer_id] * (1 - self.tau)
                self.q_model.get_layer(name=layer_name).set_weights(par_updated)
            else:
                self.q_model.get_layer(name=layer_name).set_weights(best_weights[layer_id])

        self.update_target_model_weights()


def worker_func(worker_idx, env_id, win_reward, loss_trials, max_episodes, port):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect('tcp://127.0.0.1:' + str(port))

    dqn_agent = DQNAgent(worker_idx, env_id, win_reward, loss_trials, max_episodes)
    dqn_agent.start_rl(socket)


class MultiDQN:
    def __init__(self):
        # stores the reward per episode
        self.scores = {}
        self.losses = {}
        self.weight_update_episodes = []

        self.global_max_ema_score = 0
        self.global_min_ema_loss = 0
        self.best_weights = None
        self.best_found_worker = -1

        for worker_idx in range(num_workers):
            self.scores[worker_idx] = []
            self.losses[worker_idx] = []

        self.global_logger = get_logger("cartpole_global")

        if not os.path.exists("./graphs/"):
            os.makedirs("./graphs/")

        self.continue_loop = True

    def update_loss(self, worker_idx, loss):
        # self.global_logger.info("Worker {0} sends its loss value: {1}".format(worker_idx, loss))
        self.losses[worker_idx].append(loss)
        self.save_graph()

    def update_score(self, worker_idx, score):
        # self.global_logger.info("Worker {0} sends its score value: {1}".format(worker_idx, score))
        self.scores[worker_idx].append(score)
        self.save_graph()

    def save_graph(self):
        plt.clf()

        f, axarr = plt.subplots(nrows=num_workers, ncols=2, sharex=True)
        f.subplots_adjust(hspace=0.25)

        for worker_idx in range(num_workers):
            axarr[worker_idx][0].plot(
                range(len(self.losses[worker_idx])),
                self.losses[worker_idx],
                'b'
            )
            axarr[worker_idx][0].plot(
                range(len(self.losses[worker_idx])),
                exp_moving_average(self.losses[worker_idx], 10),
                'r'
            )

            axarr[worker_idx][1].plot(
                range(len(self.scores[worker_idx])),
                self.scores[worker_idx],
                'b'
            )
            axarr[worker_idx][1].plot(
                range(len(self.scores[worker_idx])),
                exp_moving_average(self.scores[worker_idx], 10),
                'r'
            )

        plt.savefig("./graphs/loss_score.png")
        plt.close('all')

    # def save_graph(self):
    #     plt.clf()
    #
    #     f, axarr = plt.subplots(nrows=num_workers, ncols=2, sharex=True)
    #     f.subplots_adjust(hspace=0.25)
    #
    #     min_size = sys.maxsize
    #
    #     for worker_idx in range(num_workers):
    #         if len(self.losses[worker_idx]) < min_size:
    #             min_size = len(self.losses[worker_idx])
    #         if len(self.scores[worker_idx]) < min_size:
    #             min_size = len(self.scores[worker_idx])
    #
    #     for worker_idx in range(num_workers):
    #         losses = self.losses[worker_idx][0:min_size]
    #
    #         axarr[worker_idx][0].plot(range(min_size), losses, 'b')
    #         axarr[worker_idx][0].plot(range(min_size), exp_moving_average(self.losses[worker_idx], 10)[0:min_size],
    #                                   'r')
    #         # axarr[worker_idx][0].legend(["Loss", "Loss_EMA"])
    #
    #         scores = self.scores[worker_idx][0:min_size]
    #         axarr[worker_idx][1].plot(range(min_size), scores, 'b')
    #         axarr[worker_idx][1].plot(range(min_size), exp_moving_average(self.scores[worker_idx], 10)[0:min_size],
    #                                   'r')
    #         # axarr[worker_idx][1].legend(["Score", "Score_EMA"])
    #
    #         # axarr[worker_idx][1].scatter(
    #         #     x=self.weight_update_episodes,
    #         #     y=np.ndarray(scores)[self.weight_update_episodes],
    #         #     s=10
    #         # )
    #
    #     plt.savefig("./graphs/loss_score.png")
    #     plt.close('all')

    def log_info(self, msg):
        self.global_logger.info(msg)


def server_func(multi_dqn):
    context = zmq.Context()
    sockets = {}

    for worker_idx in range(num_workers):
        sockets[worker_idx] = context.socket(zmq.REP)
        sockets[worker_idx].bind('tcp://127.0.0.1:' + str(10000 + worker_idx))
        #sockets[worker_idx].setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    notification_per_workers = 0
    solved_notification_per_workers = 0

    solved_workers = []

    while True:
        for worker_idx in range(num_workers):
            if worker_idx in solved_workers:
                continue

            try:
                episode_msg = sockets[worker_idx].recv()
            except zmq.error.ZMQError as e:
                print("zmq.error.ZMQError!")
                break
            episode_msg = zlib.decompress(episode_msg)
            episode_msg = pickle.loads(episode_msg)

            if episode_msg["type"] == "episode":
                multi_dqn.update_loss(episode_msg["worker_idx"], loss=episode_msg["loss"])
                multi_dqn.update_score(episode_msg["worker_idx"], score=episode_msg["score"])

                if multi_dqn.continue_loop:
                    if score_based_transfer or loss_based_transfer:
                        episode = episode_msg["episode"]
                        ema_score = episode_msg["ema_score"]
                        ema_loss = episode_msg["ema_loss"]

                        if len(episode_msg["weights"]) > 0: # Worker로 부터 Best Weights를 수신 받음
                            multi_dqn.weight_update_episodes.append(episode)
                            if score_based_transfer:
                                msg = ">>> Best Weights Found by Worker {0} at Episode {1}!!! - Last Global Max EMA " \
                                      "Score: {2:4.1f}, New Global Max EMA Score: {3:4.1f}".format(
                                    worker_idx,
                                    episode,
                                    multi_dqn.global_max_ema_score,
                                    ema_score
                                )
                                multi_dqn.global_max_ema_score = ema_score

                            if loss_based_transfer:
                                msg = ">>> Best Weights Found by Worker {0} at Episode {1}!!! - Last Global Min EMA " \
                                      "Loss: {2:6.4f}, New Global Min EMA Loss: {3:6.4f}".format(
                                    worker_idx,
                                    episode,
                                    multi_dqn.global_min_ema_loss,
                                    ema_loss
                                )
                                multi_dqn.global_min_ema_loss = ema_loss

                            multi_dqn.log_info(msg)
                            if verbose: print(msg)

                            multi_dqn.best_weights = episode_msg["weights"]
                            multi_dqn.best_found_worker = worker_idx
                            notification_per_workers = 1

                            if score_based_transfer:
                                episode_ack_msg = {
                                    "type": "episode_ack",
                                    "global_max_ema_score": multi_dqn.global_max_ema_score,
                                    "best_weights": multi_dqn.best_weights,
                                    "best_found_worker": multi_dqn.best_found_worker
                                }
                            else:
                                episode_ack_msg = {
                                    "type": "episode_ack",
                                    "global_min_ema_loss": multi_dqn.global_min_ema_loss,
                                    "best_weights": multi_dqn.best_weights,
                                    "best_found_worker": multi_dqn.best_found_worker
                                }

                            send_to_worker(sockets[worker_idx], episode_ack_msg)

                        else: # Worker로 부터 Best Weights를 수신 받지 못함
                            notification_per_workers += 1

                            if score_based_transfer:
                                episode_ack_msg = {
                                    "type": "episode_ack",
                                    "global_max_ema_score": multi_dqn.global_max_ema_score,
                                    "best_weights": multi_dqn.best_weights,
                                    "best_found_worker": multi_dqn.best_found_worker
                                }
                            else:
                                episode_ack_msg = {
                                    "type": "episode_ack",
                                    "global_min_ema_loss": multi_dqn.global_min_ema_loss,
                                    "best_weights": multi_dqn.best_weights,
                                    "best_found_worker": multi_dqn.best_found_worker
                                }
                            send_to_worker(sockets[worker_idx], episode_ack_msg)

                            if notification_per_workers == num_workers:
                                notification_per_workers = 0
                                multi_dqn.best_weights = {}
                                multi_dqn.best_found_worker = -1
                    else:
                        episode_ack_msg = {
                            "type": "episode_ack"
                        }
                        send_to_worker(sockets[worker_idx], episode_ack_msg)
                else:
                    solved_notification_per_workers += 1
                    episode_ack_msg = {
                        "type": "solved_ack"
                    }
                    send_to_worker(sockets[worker_idx], episode_ack_msg)
            elif episode_msg["type"] == "solved":

                solved_workers.append(int(episode_msg["worker_idx"]))

                msg = "SOLVED!!! - Last Episode: {0} by {1} {2}".format(
                    episode_msg["last_episode"],
                    "DDQN" if ddqn else "DQN",
                    "with Transfer" if score_based_transfer or loss_based_transfer else "Without Transfer"
                )
                multi_dqn.log_info(msg)
                if verbose: print(msg)

                if score_based_transfer or loss_based_transfer:
                    solved_notification_per_workers = 1
                multi_dqn.continue_loop = False
            else:
                pass

        if solved_notification_per_workers == num_workers:
            break


def send_to_worker(socket, episode_ack_msg):
    episode_ack_msg = pickle.dumps(episode_ack_msg, protocol=-1)
    episode_ack_msg = zlib.compress(episode_ack_msg)
    socket.send(episode_ack_msg)


if __name__ == '__main__':
    # the CartPole-v0 is considered solved if for 100 consecutive trials,
    # the cart pole has not fallen over and it has achieved an average reward of 195.0.
    # a reward of +1 is provided for every timestep the pole remains
    # upright
    win_reward = 195.0

    # loss
    loss_trials = 10

    max_episodes = 3000

    env_id = "CartPole-v0"

    num_experiments = 1

    for _ in range(num_experiments):
        # instantiate the DQN/DDQN agent
        multi_dqn = MultiDQN()

        server = Process(target=server_func, args=(multi_dqn,))
        server.start()

        clients = []
        for worker_idx in range(num_workers):
            client = Process(target=worker_func, args=(
                worker_idx,
                env_id,
                win_reward,
                loss_trials,
                max_episodes,
                10000 + worker_idx
            ))

            clients.append(client)

            client.start()

        while True:
            is_anyone_alive = True
            for client in clients:
                is_anyone_alive = client.is_alive()
            is_anyone_alive = server.is_alive()

            if not is_anyone_alive:
                break

            time.sleep(1)