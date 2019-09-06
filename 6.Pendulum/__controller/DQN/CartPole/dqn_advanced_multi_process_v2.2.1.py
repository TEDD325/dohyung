"""
x = Dense(512, activation='relu', name="hidden_layer_0_"+str(self.worker_idx))(inputs)
x = Dense(512, activation='relu', name="hidden_layer_1_"+str(self.worker_idx))(x)
x = Dense(512, activation='relu', name="hidden_layer_2_"+str(self.worker_idx))(x)
learning_rate = 0.001
self.epsilon_min = 0.1
"""

import math
import threading
import time
import zlib
import tensorflow as tf
from logger import get_logger
import sys
import json
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
import random
import gym
from multiprocessing import Process
import zmq
import matplotlib.pyplot as plt
import os
import pickle
import warnings

print(tf.__version__)
warnings.filterwarnings("ignore")

ddqn = True
num_workers = 4
num_hidden_layers = 3
transfer = True
num_weight_transfer_hidden_layers = 3
verbose = False
learning_rate = 0.001
experiment_count = 1
filename = "dqn_advanced_multi_process_v2.2.1"

def exp_moving_average(values, window): #?
    """
    :param values:
    :param window:
    :return:
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
    def __init__(self, worker_idx, env_id, win_trials, win_reward, loss_trials, max_episodes):
        """
        :param worker_idx: int
        :param env_id: str
        :param win_trials: int
        :param win_reward: float
        :param loss_trials: int
        :param max_episodes: int
        """
        self.worker_idx = worker_idx
        self.env = gym.make(env_id)
        self.env.seed(0)

        self.action_space = self.env.action_space

        # experience buffer
        self.memory = []

        # discount rate
        self.gamma = 0.9

        # initially 90% exploration, 10% exploitation
        self.epsilon = 1.0

        # iteratively applying decay til 10% exploration/90% exploitation
        self.epsilon_min = 0.1 / num_workers

        self.win_trials = win_trials
        self.win_reward = win_reward

        self.loss_trials = loss_trials

        # Q Network weights filename
        self.weights_file = './dqn_cartpole.h5'

        # Q Network for training
        self.n_inputs = self.env.observation_space.shape[0]
        self.n_outputs = self.action_space.n

        self.q_model = self.build_model(self.n_inputs, self.n_outputs)
        self.q_model.compile(loss='mse', optimizer=Adam(lr=learning_rate))

        # target Q Network
        self.target_q_model = self.build_model(self.n_inputs, self.n_outputs)

        # copy Q Network params to target Q Network
        self.update_target_model_weights()

        self.replay_counter = 0

        if ddqn:
            print("----------Worker {0}-{1}: Double DQN--------".format(
                self.worker_idx,
                "With Transfer" if transfer else "Without Transfer",
            ))
        else:
            print("-------------Worker {0}-{1}: DQN------------".format(
                self.worker_idx,
                "With Transfer" if transfer else "Without Transfer",
            ))

        self.logger = get_logger("cartpole_worker_" + str(self.worker_idx))

        self.max_episodes = max_episodes

        self.global_max_mean_score = 0
        self.global_mean_loss = 0.0

        self.local_scores = []
        self.local_losses = []

    # Q Network is 256-256-256-2 MLP
    def build_model(self, n_inputs, n_outputs):
        """
        :param n_inputs: int, 4
        :param n_outputs: int, 2
        :return: tensorflow.keras.Model
        """
        inputs = Input(shape=(n_inputs,), name="state_"+str(self.worker_idx))
        x = Dense(512, activation='relu', name="hidden_layer_0_"+str(self.worker_idx))(inputs)
        x = Dense(512, activation='relu', name="hidden_layer_1_"+str(self.worker_idx))(x)
        x = Dense(512, activation='relu', name="hidden_layer_2_"+str(self.worker_idx))(x)
        x = Dense(n_outputs, activation='linear', name="output_layer_"+str(self.worker_idx))(x)
        model = Model(inputs, x)
        # model.summary()
        return model

    # save Q Network params to a file
    def save_weights(self):
        self.q_model.save_weights(self.weights_file)

    # copy trained Q Network params to target Q Network
    def update_target_model_weights(self):
        self.target_q_model.set_weights(
            self.q_model.get_weights()
        )

    # copy trained other worker's Q Network params to current Q Network
    def update_layer_weights_from_other_worker(self, layer, weights):
        """
        :param layer:
        :param weights:
        """
        self.q_model.get_layer(name="layer_"+str(layer)).set_weights(weights)

    # eps_greddy policy
    def act(self, state):
        """
        :param state:
        :return:
        """
        if np.random.rand() < self.epsilon:
            # explore - do random action
            return self.action_space.sample()
        # exploit
        q_values = self.q_model.predict(state)

        return np.argmax(q_values[0])

    # select experieces in the replay buffer
    def remember(self, state, action, reward, next_state, done):
        """
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        """
        item = (state, action, reward, next_state, done)
        self.memory.append(item)

    # compute Q_max
    # use of target Q Network solves the non-stationarity problem
    def get_target_q_value(self, next_state, reward):
        """
        :param next_state:
        :param reward:
        :return:
        """
        # max Q value among next state's actions
        if ddqn:
            # DDQN
            # current Q Network selects the action
            # a'_max = argmax_a'(s', a'_max)
            action = np.argmax(self.q_model.predict(next_state)[0]) #?

            # target Q Network evaluates the action
            # Q_max = Q_targets(s', a'_max)
            q_max = self.target_q_model.predict(next_state)[0][action] #?
        else:
            # DQN chooses the max Q value among next actions
            # selection and evaluation of action is on the target Q Network
            # Q_max = max_a' Q target(s', a')
            q_max = np.amax(self.target_q_model.predict(next_state)[0]) #?

        # Q_value = reward + gamma * Q_max
        q_value = reward + self.gamma * q_max
        return q_value

    # experience replay addresses the correlation issue between samples
    def replay(self, batch_size, episode):
        """
        :param batch_size:
        :param episode:
        :return:
        """
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

    # decrease the exploration, increase exploitation
    def update_epsilon(self, episode):
        """
        :param episode:
        :return:
        """
        if episode != 0:
            epsilon_decay = (self.epsilon_min / self.epsilon) ** (1. / float(episode))

            if self.epsilon > self.epsilon_min:
                self.epsilon *= epsilon_decay
        # print("epsilon:", self.epsilon)

    def start_rl(self, socket):
        """
        :param socket:
        :return:
        """
        # should be solved in this number of episodes
        state_size = self.env.observation_space.shape[0]
        batch_size = 64

        # by default, CartPole-v0 has max episode steps = 200
        # you can use this to experiment beyond 200
        # env._max_episode_steps = 4000

        # Q-Learning sampling and fitting
        for episode in range(self.max_episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, state_size])
            done = False
            total_reward = 0
            while not done:
                # in CartPole-v0, action=0 is left and action=1 is right
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                # in CartPole-v0:
                # state = [pos, vel, theta, angular speed]
                next_state = np.reshape(next_state, [1, state_size])
                # store every experience unit in replay buffer
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            if len(self.memory) < batch_size:
                continue

            # call experience replay
            loss = self.replay(batch_size, episode)
            score = total_reward

            self.local_losses.append(loss)
            self.local_scores.append(score)

            mean_loss = exp_moving_average(self.local_losses, 10)[-1] #?
            mean_score = exp_moving_average(self.local_scores, 10)[-1]

            if self.global_max_mean_score < mean_score: #!
                self.global_max_mean_score = mean_score #!
                send_weights = True

                msg = ">>> Worker {0}: Find New Best Weights!!! - global_max_score: {1}".format(
                    self.worker_idx,
                    self.global_max_mean_score
                )
                self.logger.info(msg)
                if verbose: print(msg)

            else:
                send_weights = False

            continue_loop = self.send_episode_info_and_adapt_best_weights(
                socket,
                loss,
                score,
                mean_score,
                send_weights=send_weights
            )

            if not continue_loop:
                time.sleep(1)
                break

            msg = "Worker {0} - Episode {1:>2d} Loss = {2:6.4f} (Mean: {3:6.4f}), Score = {4:5.1f} (Mean: {5:4.2f} in {6} episodes".format(
                self.worker_idx,
                episode,
                loss,
                mean_loss,
                score,
                mean_score,
                self.win_trials
            )
            self.logger.info(msg)
            if verbose: print(msg)

            if mean_score >= self.win_reward and episode >= self.win_trials:#?
                msg = "******* Worker {0} - Solved in episode {1}: Mean score = {2} in {3} episodes: Epsilon: {4}".format(
                    self.worker_idx,
                    episode,
                    mean_score,
                    self.win_trials,
                    self.epsilon
                )
                self.logger.info(msg)
                print(msg)

                self.save_weights()#?
                self.send_solve_info(#?
                    socket,
                    last_episode=episode
                )

                with open('dqn_advanced_multi_process_v2.2_experiment_result.txt', 'a+') as f:
                    exp_result = "\n\n "+str(experiment_count)+" \n *** Worker "+ str(self.worker_idx)+" - Solved in episode " \
                                 + str(episode)+ ": Mean score = " + str(mean_score) + " in " + str(self.win_trials)+ " episodes"
                    f.write(exp_result)

                break

        # close the env and write monitor result info to disk
        self.env.close()

    def send_episode_info_and_adapt_best_weights(self, socket, loss, score, mean_score, send_weights):
        if transfer:
            # weights = {} #? line 379, 399
            if send_weights:
                weights = {}
                if num_weight_transfer_hidden_layers > num_hidden_layers: #?
                    layer_id = 0
                    for layer_id in range(num_hidden_layers): #? indentation
                        layer_name = "hidden_layer_{0}_{1}".format(
                            layer_id,
                            self.worker_idx
                        )
                        weights[layer_id] = self.q_model.get_layer(name=layer_name).get_weights()

                    weights[layer_id + 1] = self.q_model.get_layer(name="output_layer_{0}".format(self.worker_idx)).get_weights()
                else:
                    for layer_id in range(num_weight_transfer_hidden_layers):
                        layer_name = "hidden_layer_{0}_{1}".format(
                            layer_id,
                            self.worker_idx
                        )
                        weights[layer_id] = self.q_model.get_layer(name=layer_name).get_weights()

            else:
                weights = {}

            episode_msg = {
                "type": "episode",
                "worker_idx": self.worker_idx,
                "loss": loss,
                "score": score,
                "mean_score": mean_score,
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

        # ACK: 응답 문자(acknowledgement code)
        episode_ack_msg = socket.recv()
        episode_ack_msg = zlib.decompress(episode_ack_msg)
        episode_ack_msg = pickle.loads(episode_ack_msg)

        continue_loop = True #?
        if episode_ack_msg["type"] == "episode_ack":
            if transfer:
                global_max_mean_score = episode_ack_msg["global_max_mean_score"]
                best_weights = episode_ack_msg["best_weights"]

                if len(best_weights) > 0 and not send_weights:
                    self.global_max_mean_score = global_max_mean_score

                    if num_weight_transfer_hidden_layers > num_hidden_layers: #?
                        layer_id = 0
                        for layer_id in range(num_hidden_layers): #! line 382-387과 중복되는 코드 부분
                            layer_name = "hidden_layer_{0}_{1}".format(
                                layer_id,
                                self.worker_idx
                            )
                            self.q_model.get_layer(name=layer_name).set_weights(best_weights[layer_id])
                        self.q_model.get_layer(name="output_layer_{0}".format(self.worker_idx)).set_weights(
                            best_weights[layer_id + 1] #?
                        )
                    else:
                        for layer_id in range(num_weight_transfer_hidden_layers):
                            layer_name = "hidden_layer_{0}_{1}".format(
                                layer_id,
                                self.worker_idx
                            )
                            self.q_model.get_layer(name=layer_name).set_weights(best_weights[layer_id])

                    msg = ">>> Worker {0}: Set New Best Weights to Local Model!!! - global_max_score: {1}".format(
                        self.worker_idx,
                        self.global_max_mean_score
                    )
                    self.logger.info(msg)
                    if verbose: print(msg)

        elif episode_ack_msg["type"] == "solved":
            msg = "Solved by Other Worker"
            self.logger.info(msg)
            if verbose: print(msg)
            continue_loop = False
        else:
            pass

        return continue_loop

    def send_solve_info(self, socket, last_episode):
        """
        :param socket:
        :param last_episode:
        :return:
        """
        solve_msg = {
            "type": "solved",
            "last_episode": last_episode
        }
        solve_msg = pickle.dumps(solve_msg, protocol=-1)
        solve_msg = zlib.compress(solve_msg)
        socket.send(solve_msg)

def worker_func(worker_idx, env_id, win_trials, win_reward, loss_trials, max_episodes, port):
    context = zmq.Context()
    socket = context.socket(zmq.REQ) # REQUEST
    socket.connect("tcp://127.0.0.1:"+str(port))

    dqn_agent = DQNAgent(worker_idx, env_id, win_trials, win_reward, loss_trials, max_episodes)
    dqn_agent.start_rl(socket)

class MultiDQN:
    def __init__(self):
        # stores the reward per episode
        self.scores = {}
        self.losses = {}
        self.global_max_mean_score = 0
        self.best_weights = None

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

    def save_graph(self): #?
        plt.clf()

        f, axarr = plt.subplots(nrows=num_workers, ncols=2, sharex=True)
        f.subplots_adjust(hspace=0.25)

        min_size = sys.maxsize

        for worker_idx in range(num_workers):
            if len(self.losses[worker_idx]) < min_size:
                min_size = len(self.losses[worker_idx])
            if len(self.scores[worker_idx]) < min_size:
                min_size = len(self.scores[worker_idx])

        for worker_idx in range(num_workers):
            axarr[worker_idx][0].plot(range(min_size), self.losses[worker_idx][0:min_size], 'b')
            axarr[worker_idx][0].plot(range(min_size), exp_moving_average(self.losses[worker_idx], 10)[0:min_size], 'r')
            # axarr[worker_idx][0].legend(["Loss", "Loss_EMA"])

            axarr[worker_idx][1].plot(range(min_size), self.scores[worker_idx][0:min_size], 'b')
            axarr[worker_idx][1].plot(range(min_size), exp_moving_average(self.scores[worker_idx], 10)[0:min_size]), 'r'
            # axarr[worker_idx][1].legend(["Score", "Score_EMA"])

        plt.savefig("./graphs/loss_score.png")
        plt.close('all')

    def log_info(self, msg):
        self.global_logger.info(msg)

def server_func(multi_dqn):
    context = zmq.Context()
    sockets = {}

    for worker_idx in range(num_workers):
        sockets[worker_idx] = context.socket(zmq.REP) # RESPONSE
        sockets[worker_idx].bind('tcp://127.0.0.1:'+str(5000+worker_idx))

    max_mean_score_notification_per_workers = 0
    solved_notification_per_workers = 0

    while True:
        for worker_idx in range(num_workers):
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
                    if transfer:
                        mean_score = episode_msg["mean_score"]

                        if len(episode_msg["weights"]) > 0:
                            msg = ">>> Best Weights Found by Worker {0}!!! - Last Global Mean Max Score: {1}, New Global Mean Max Score: {2}".format(
                                worker_idx,
                                multi_dqn.global_max_mean_score,
                                mean_score
                            )
                            multi_dqn.log_info(msg)
                            if verbose: print(msg)

                            multi_dqn.global_max_mean_score = mean_score
                            multi_dqn.best_weights = episode_msg["weights"]
                            max_mean_score_notification_per_workers = 1
                        else:
                            if max_mean_score_notification_per_workers == num_workers: #?
                                max_mean_score_notification_per_workers = 0
                                multi_dqn.best_weights = {}
                            else:
                                max_mean_score_notification_per_workers += 1
                        episode_ack_msg = {
                            "type": "episode_ack",
                            "global_max_mean_score": multi_dqn.global_max_mean_score,
                            "best_weights": multi_dqn.best_weights
                        }
                    else:
                        episode_ack_msg = {
                            "type": "episode_ack"
                        }
                else:
                    solved_notification_per_workers += 1
                    episode_ack_msg = {
                        "type": "solved"
                    }

                episode_ack_msg = pickle.dumps(episode_ack_msg, protocol=-1)
                episode_ack_msg = zlib.compress(episode_ack_msg)
                sockets[worker_idx].send(episode_ack_msg)

            elif episode_msg["type"] == "solved":
                msg = "SOLVED!!! - Last Episode: {0} by {1} {2}".format(
                    episode_msg["last_episode"],
                    "DDQN" if ddqn else "DQN",
                    "with transfer" if transfer else "Without Transfer"
                )
                multi_dqn.log_info(msg)
                if verbose: print(msg)

                solved_notification_per_workers = 1
                multi_dqn.continue_loop = False
            else:
                pass

        if solved_notification_per_workers == num_workers: #?
            break

if __name__ == "__main__":
    # the number of trials without falling over
    win_trials = 100

    # the CartPole-v0 is considered solved if for 100 consecutive trials,
    # the cart pole has not fallen over and it has achieved an average
    # reward of 195.0
    # a reward of +1 is provided for every timestep the pole remains upright
    win_reward = 195.0

    # loss
    loss_trials = 10

    max_episodes = 3000

    env_id = "CartPole-v0"

    for _ in range(10):
        # instantiate the DQN/DDQN agent
        multi_dqn = MultiDQN()

        server = Process(target=server_func, args=(multi_dqn,))
        server.start()

        clients = []
        for worker_idx in range(num_workers):
            client = Process(target=worker_func, args=(
                worker_idx,
                env_id,
                win_trials,
                win_reward,
                loss_trials,
                max_episodes,
                5000+worker_idx
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